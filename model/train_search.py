import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('cuda is not available')
        sys.exit(1)

    # 优化网络model，即w的优化,alpha的优化在下面的architect部分
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # 架构搜索时使用CiFAR-10作为代理任务
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # 划分训练集、验证集
    # 论文里是一半训练集，一半验证集,即train_portion为0.5
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    # 优化w时的学习率调整
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    # 初始化整个架构，在architect中定义, 即优化alpha
    architect = Architect(model, args)

    for epoch in range(args.epoches):
        # 每个epoch优化一次model的lr
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        # 保存模型
        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    """

    :param train_queue: 存放mini-batch
    :param valid_queue:
    :param model:
    :param architect:
    :param criterion:
    :param optimizer:
    :param lr:
    :return:
    """
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        # step对应的是enumerate产生的索引下标
        # input张量的形状是[batch_size, input_channels, input_height, input_width]
        # target此处是指标签
        model.train()
        # 获取batch的(train_queue中存放的是一个个batch)
        n = input.size(0)

        # input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        # 上方的两行代码进行如下改写，因为Variable已经被废弃了
        input = input.clone().detach()
        target = Variable(target, requires_grad=False)

        # 更新架构参数alpha是用验证集更新的，所以从valid_queue中取出batch传人architect.step()
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cpu()
        target_search = target_search.cpu()

        # 下面是DARTS的优化部分。DARTS是交替优化的，先优化alpha，再优化w
        # 更新alpha（unrolled为True时）
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # 前向传播，计算loss
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        # 更新w
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)  # 梯度裁剪
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        n = input.size(0)

        input = Variable(input, volatile=True)
        target = Variable(target, volatile=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(loss.data[0], n)
        top5.update(loss.data[0], n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
