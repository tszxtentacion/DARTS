import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:  # PRIMITIVES中就是8个操作
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))  # 给池化操作后面加一个batch_normalization
            self._ops.append(op)  # 把这些op都放在预先定义好的ModuleList里

    def forward(self, x, weights):
        # op(x)指对输入x进行一个相应操作；此处即进行每个操作并且加权求和
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        """
        architecture的基本组成构件（两个基本nodes为cell k-2、k-1的输出）
        :param steps: cell中节点连接状态待确定的数量
        :param multiplier: Cell的输出边的数量，即输出特征图的数量（一个边对应一个输出特征图）
        :param C_prev_prev: cell k-2的输出通道数
        :param C_prev: cell k-1的输出通道数
        :param C: cell k（当前）的输入通道数
        :param reduction: bool, 当前单元是否采样
        :param reduction_prev: bool, 前一个单元是否采样
        """
        super(Cell, self).__init__()
        self.reduction = reduction
        # input nodes的结构不变，不参与搜索
        # 第一个input node结构 (看前一个Cell是否reduction)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        # 第二个input node结构
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps     # 为intermediate节点的个数(此处为4)
        self._multiplier = multiplier

        # 构建operation的ModuleList
        self._ops = nn.ModuleList()
        # 构建batch_normal的ModuleList
        self._bns = nn.ModuleList()

        # 遍历intermediate nodes(4个)的所有混合操作 (假设2个input，4个intermediate，1个output)
        for i in range(self._steps):
            for j in range(2 + i):  # 类似于二重循环的遍历
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)  # op是构建两个节点之间的混合操作
                self._ops.append(op)    # 所有边的混合操作添加到ops，list的len为2+3+4+5=14[[],[],...,[]]

    # cell前向传播时自动调用
    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # states中为[s0,s1,b1,b2,b3,b4] b1,b2,b3,b4分别是四个intermediate output的输出
        # 对于一个特征图（张量），它的维度通常为4，即(batch_size, num_channels, height, width)
        states = [s0, s1]
        offset = 0
        # 遍历每个intermediate nodes, 得到各自的output
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        # 对intermediate的output进行concat作为当前cell的输出；#dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍
        # 取后_multiplier个特征图，不取输入特征图防止特征冗余
        return torch.cat(states[-self._multiplier:], dim=1)
        # 特征图沿着通道维度（dim = 1）进行拼接。通过在通道维度上拼接特征图，我们可以整合来自不同边的输出特征（因为不同方案的通道数不一样），形成一个更丰富的表示。


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        """

        :param C: 初始的通道数量
        :param num_classes: 分类任务的类别数量，决定最后的神经元数量
        :param layers: 模型的层数（标量，数量）
        :param criterion: 一个布尔值，用于确定是否在模型中添加辅助分类器。辅助分类器可以帮助训练过程更加稳定
        :param steps: 每个cell中的节点数量
        :param multiplier: 输出通道数量 = multiplier * C
        :param stem_multiplier: 前一层输出通道数和现在的输出通道数的倍数关系。在模型的起始部分，stem_multiplier用于决定输入数据的处理方式。
        它会影响起始层的通道数量，这个数量等于初始通道数量C乘以stem_multiplier
        """
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        # 当前的输入
        C_curr = stem_multiplier * C
        # stem指初始部分
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    # 根据新的架构参数alpha产生新的模型
    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            # s0和s1分别表示上一个cell和当前cell的输出特征
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # 可搜索操作的数量，即一个cell中边的总数。这里的2表示的是两个输入节点，steps是中间节点的数量
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        # 可选操作的数量
        num_ops = len(PRIMITIVES)

        self.alphas_normal = torch.tensor(1e-3 * torch.randn(k, num_ops), requires_grad=True).to('cuda')
        self.alphas_reduce = torch.tensor(1e-3 * torch.randn(k, num_ops), requires_grad=True).to('cuda')
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def arch_parameters(self):
        return self._arch_parameters

    # 将架构参数alpha转换为一个具体的网络架构。即从alphas_normal和alphas_reduce中选取权重最大的几个操作，
    # 作为普通cell和reduction cell中的主要操作。
    def genotype(self):
        # 为给定的alpha值（alphas_normal或alphas_reduce）找到具有最大权重的操作。（此处为操作权重alpha。节点权重是神经网络本身的参数）
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                # 输入当前节点的所有可能操作，即当前节点的所有输入边的权重
                W = weights[start:end].copy()
                # 根据最大权重对节点进行降序排序，[:2]表示找到与当前中间节点相连的两个具有最大权重的输入节点。
                # range(i+2)表示生成当前节点之前的所有节点;降序排列操作的权重;k用于遍历W;排除表示没有连接的None操作）
                edges = sorted(range(i+2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu.numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        # 计算要进行特征拼接的节点序列，在DARTS中，输出节点通常是将所有中间节点的特征进行拼接，以生成最终的输出特征。
        # multiplier表示最终要拼接的中间节点的输出的个数，一般小于steps，为了防止特征冗余。此处意为拼接最后multiplier个中间节点
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        # 创建一个Genotype对象，用于表示最终神经网络架构
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        # 通过创建Genotype对象，我们可以将学习到的权重（alphas）转换为一个具体的神经网络架构。
        # 在训练过程结束后，我们可以使用这个神经网络架构进行实际的训练和推理任务。
        return genotype
