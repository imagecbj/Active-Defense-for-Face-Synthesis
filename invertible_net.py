import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utlz
import torch.nn.init as init

clock_global = 1.0
augment_global = True


def initialize_weights(net_l, scale=1):
    """
    初始化网络权重

    参数:
        net_l (list or nn.Module): 网络模型或模型列表
        scale (float): 权重缩放因子，默认为1

    注意:
        该函数将会初始化卷积层和全连接层的权重，
        采用 Kaiming 正态分布初始化方法（fan_in），
        并可选地对权重进行缩放。此外，对 Batch Normalization 层的权重进行初始化。
    """
    # 如果输入的不是列表，将其转换为列表
    if not isinstance(net_l, list):
        net_l = [net_l]

    # 对每个网络进行权重初始化
    for net in net_l:
        # 遍历网络中的每个模块
        for m in net.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 正态分布初始化权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # 对权重进行缩放（用于残差块）
                m.weight.data *= scale
                # 如果存在偏置项，则将其初始化为零
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 使用 Kaiming 正态分布初始化权重
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # 对权重进行缩放
                m.weight.data *= scale
                # 如果存在偏置项，则将其初始化为零
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果是 Batch Normalization 层
            elif isinstance(m, nn.BatchNorm2d):
                # 将 Batch Normalization 层的权重初始化为1
                init.constant_(m.weight, 1)
                # 将 Batch Normalization 层的偏置项初始化为零
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    """
    使用 Xavier 初始化网络权重

    参数:
        net_l (list or nn.Module): 网络模型或模型列表
        scale (float): 权重缩放因子，默认为1

    注意:
        该函数将会使用 Xavier 初始化方法对卷积层和全连接层的权重进行初始化，
        并可选地对权重进行缩放。此外，对 Batch Normalization 层的权重进行初始化。
    """
    # 如果输入的不是列表，将其转换为列表
    if not isinstance(net_l, list):
        net_l = [net_l]

    # 对每个网络进行权重初始化
    for net in net_l:
        # 遍历网络中的每个模块
        for m in net.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用 Xavier 初始化方法初始化权重
                init.xavier_normal_(m.weight)
                # 对权重进行缩放（用于残差块）
                m.weight.data *= scale
                # 如果存在偏置项，则将其初始化为零
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 使用 Xavier 初始化方法初始化权重
                init.xavier_normal_(m.weight)
                # 对权重进行缩放
                m.weight.data *= scale
                # 如果存在偏置项，则将其初始化为零
                if m.bias is not None:
                    m.bias.data.zero_()
            # 如果是 Batch Normalization 层
            elif isinstance(m, nn.BatchNorm2d):
                # 将 Batch Normalization 层的权重初始化为1
                init.constant_(m.weight, 1)
                # 将 Batch Normalization 层的偏置项初始化为零
                init.constant_(m.bias.data, 0.0)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, rev):
        """
        前向传播或反向传播操作

        参数:
            x (Tensor): 输入张量
            rev (bool): 表示正向传播还是反向传播

        返回:
            list: 处理后的张量列表
        """
        x = x[0]  # 获取输入张量
        if not rev:  # 如果是正向传播
            B, C, H, W = x.shape  # 获取张量形状
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # 将张量按空间维度分解
            x = x.permute(0, 1, 3, 5, 2, 4)  # 转置操作，使通道维度变为第二维度
            x = x.reshape(B, 4 * C, H // 2, W // 2)  # 将空间维度因子聚合到通道维度中
            return [x]  # 返回处理后的张量
        else:  # 如果是反向传播
            B, C, H, W = x.shape  # 获取张量形状
            x = x.reshape(B, C // 4, 2, 2, H, W)  # 将张量按通道维度分解
            x = x.permute(0, 1, 4, 2, 5, 3)  # 转置操作，使空间维度变为第二维度
            x = x.reshape(B, C // 4, 2 * H, 2 * W)  # 将通道维度因子聚合到空间维度中
            return [x]  # 返回处理后的张量



class Unsqueeze(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, rev):
        x = x[0]  # 获取输入张量
        if not rev:  # 如果是正向传播
            B, C, H, W = x.shape  # 获取张量形状
            x = x.reshape(B, C // 4, 2, 2, H, W)  # 将张量按通道维度分解
            x = x.permute(0, 1, 4, 2, 5, 3)  # 转置操作，使空间维度变为第二维度
            x = x.reshape(B, C // 4, 2 * H, 2 * W)  # 将通道维度因子聚合到空间维度中
            return [x]  # 返回处理后的张量
        else:  # 如果是反向传播
            B, C, H, W = x.shape  # 获取张量形状
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # 将张量按空间维度分解
            x = x.permute(0, 1, 3, 5, 2, 4)  # 转置操作，使通道维度变为第二维度
            x = x.reshape(B, 4 * C, H // 2, W // 2)  # 将空间维度因子聚合到通道维度中
            return [x]  # 返回处理后的张量




class RNVPCouplingBlock(nn.Module):
    # 耦合块遵循RealNVP设计

    def __init__(self, dims_in, subnet_constructor=None, clamp=1.0, clock=1):
        super().__init__()  # 初始化父类
        self.clock = clock  # 计时器或其他追踪机制
        channels = dims_in[0][0]  # 输入的通道数
        self.ndims = len(dims_in[0])  # 输入数据的维数
        self.split_len1 = channels // 2  # 将通道分成两部分，这是第一部分的长度
        self.split_len2 = channels - channels // 2  # 第二部分的长度

        self.clamp = clamp  # 用于乘性组件的软夹紧，限制每个输入维度的放大或衰减
        self.affine_eps = 0.0001  # 用于避免除以0的小常数

        # 初始化四个子网络
        self.s1 = subnet_constructor(self.split_len1, self.split_len2)
        self.t1 = subnet_constructor(self.split_len1, self.split_len2)
        self.s2 = subnet_constructor(self.split_len2, self.split_len1)
        self.t2 = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        # 计算放大系数
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps

    def forward(self, x, rev=False):
        x = x[0]  # 取出输入数据的第一个元素

        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))


        if not rev:
            # 正向传播
            x2_c = x2
            s2, t2 = self.s2(x2_c), self.t2(x2_c)
            y1 = self.e(s2) * x1 + t2  # 应用变换
            y1_c = y1
            s1, t1 = self.s1(y1_c), self.t1(y1_c)
            y2 = self.e(s1) * x2 + t1  # 再次应用变换
            self.last_s = [s1, s2]  # 存储s值
        else:
            # 反向传播
            # 交换x1和y1的命名
            x1_c = x1
            s1, t1 = self.s1(x1_c), self.t1(x1_c)
            y2 = (x2 - t1) / self.e(s1)  # 应用逆变换
            y2_c = y2
            s2, t2 = self.s2(y2_c), self.t2(y2_c)
            y1 = (x1 - t2) / self.e(s2)  # 再次应用逆变换
            self.last_s = [s1, s2]  # 存储s值

        return torch.cat((y1, y2), 1)  # 将输出合并并返回


class HaarDownsampling(nn.Module):
    # Haar下采样模块

    def __init__(self, dims_in, order_by_wavelet=False, rebalance=1.):
        super().__init__()  # 初始化父类

        # 初始化参数
        self.in_channels = dims_in[0][0]  # 输入通道数
        self.fac_fwd = 0.5 * rebalance  # 正向缩放因子
        self.fac_rev = 0.5 / rebalance  # 反向缩放因子

        # 初始化Haar小波权重
        self.haar_weights = torch.ones(4, 1, 2, 2)
        # 调整权重以实现Haar小波变换
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1
        # 复制权重以匹配输入通道数
        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        # 设置权重为不可训练的参数
        self.haar_weights = nn.Parameter(self.haar_weights, requires_grad=False)

        self.permute = order_by_wavelet  # 是否按小波顺序排列输出
        self.last_jac = None  # 用于存储最后的雅可比矩阵

        if self.permute:
            # 如果需要按小波顺序排列，生成排列索引
            permutation = [i + 4 * j for j in range(self.in_channels) for i in range(4)]
            self.perm = torch.LongTensor(permutation)  # 排列索引
            self.perm_inv = torch.LongTensor(permutation)  # 反排列索引
            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def forward(self, x, rev=False):
        # 定义正向和反向传播
        if not rev:
            # 正向传播：使用Haar权重进行下采样
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels)
            if self.permute:
                # 如果需要排列，按照指定索引排列输出
                return [out[:, self.perm] * self.fac_fwd]
            else:
                return [out * self.fac_fwd]
        else:
            # 反向传播：使用Haar权重进行上采样
            if self.permute:
                x_perm = x[0][:, self.perm_inv]
            else:
                x_perm = x[0]
            return [F.conv_transpose2d(x_perm * self.fac_rev, self.haar_weights, bias=None, stride=2,
                                       groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # 计算并返回雅可比矩阵，此处未具体实现
        return self.last_jac

    def output_dims(self, input_dims):
        # 计算输出的维度
        assert len(input_dims) == 1, "HaarDownsampling只能处理一个输入"
        c, w, h = input_dims[0]
        # 输出的通道数、宽度和高度
        c2, w2, h2 = c * 4, w // 2, h // 2
        self.elements = c * w * h  # 输入的元素总数
        assert c * h * w == c2 * h2 * w2, "输入维度不均匀"
        return [(c2, w2, h2)]


class HaarUpsampling(nn.Module):
    def __init__(self, dims_in):
        super().__init__()

        # 输入通道数
        self.in_channels = dims_in[0][0] // 4
        # Haar小波权重
        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1
        self.haar_weights *= 0.5
        # 复制权重以适应通道数
        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        # 将权重设置为可训练参数
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if rev:
            # 反向传播时的操作
            return [F.conv2d(x[0], self.haar_weights,
                             bias=None, stride=2, groups=self.in_channels)]
        else:
            # 正向传播时的操作
            return [F.conv_transpose2d(x[0], self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # 返回雅可比矩阵
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "只能使用一个输入"
        c, w, h = input_dims[0]
        # 输出通道数、宽度和高度
        c2, w2, h2 = c // 4, w * 2, h * 2
        assert c * h * w == c2 * h2 * w2, "不均匀的输入维度"
        return [(c2, w2, h2)]




def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    flow = flow.permute(0,2,3,1)
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output




class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResidualDenseBlock_out_att(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out_att, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.ca = ChannelAttention(input + 4 * 32)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        # 应用通道注意力和空间注意力
        # x5 = self.ca(x5) * x5
        # x5 = self.sa(x5) * x5
        return x5
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class INV_block_att(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out_att, clamp=2.0, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)

class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=2.0, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class Model_att(nn.Module):
    def __init__(self):
        super(Model_att, self).__init__()

        self.model = Hinet_att()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)


class Hinet_att(nn.Module):

    def __init__(self):
        super(Hinet_att, self).__init__()

        self.inv1 = INV_block_att()
        self.inv2 = INV_block_att()
        self.inv3 = INV_block_att()
        self.inv4 = INV_block()
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()

        self.inv9 = INV_block()
        self.inv10 = INV_block()
        self.inv11 = INV_block()
        self.inv12 = INV_block()
        self.inv13 = INV_block()
        self.inv14 = INV_block_att()
        self.inv15 = INV_block_att()
        self.inv16 = INV_block_att()

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)

        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)

            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out



if __name__ == "__main__":
     import cv2  # 导入OpenCV库


