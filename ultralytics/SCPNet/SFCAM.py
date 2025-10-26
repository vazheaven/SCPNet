import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        """
        初始化基础卷积模块。

        参数:
        in_channel (int): 输入通道数。
        out_channel (int): 输出通道数。
        kernel_size (int): 卷积核大小。
        stride (int): 步长。
        bias (bool): 是否使用偏置，默认为 True。
        norm (bool): 是否使用批量归一化，默认为 False。
        relu (bool): 是否使用 GELU 激活函数，默认为 True。
        transpose (bool): 是否使用转置卷积，默认为 False。
        """
        super(BasicConv, self).__init__()
        # 如果同时使用偏置和批量归一化，不使用偏置
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            # 添加转置卷积层
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            # 添加普通卷积层
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            # 添加批量归一化层
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            # 添加 GELU 激活函数
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        """
        初始化残差块。

        参数:
        in_channel (int): 输入通道数。
        out_channel (int): 输出通道数。
        filter (bool): 是否使用注意力滤波器，默认为 False。
        """
        super(ResBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        # 第二个卷积层
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        if filter:
            # 11 核的立方注意力模块
            self.cubic_11 = cubic_attention(in_channel // 2, group=1, kernel=11)
            # 7 核的立方注意力模块
            self.cubic_7 = cubic_attention(in_channel // 2, group=1, kernel=7)
            # 光谱注意力模块
            self.pool_att = SpecAtte(in_channel)

        self.filter = filter

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        out = self.conv1(x)
        if self.filter:
            out = self.pool_att(out)
            # 将特征图在通道维度上分割成两部分
            out = torch.chunk(out, 2, dim=1)
            out_11 = self.cubic_11(out[0])
            out_7 = self.cubic_7(out[1])
            # 拼接两部分特征图
            out = torch.cat((out_11, out_7), dim=1)

        out = self.conv2(out)
        # 残差连接
        return out + x


class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        """
        初始化立方注意力模块。

        参数:
        dim (int): 输入特征图的通道数。
        group (int): 分组数。
        kernel (int): 卷积核大小。
        """
        super().__init__()
        # 水平空间条带注意力模块
        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        # 垂直空间条带注意力模块
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        # 可学习的缩放因子
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        # 可学习的偏移因子
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        """
        初始化空间条带注意力模块。

        参数:
        dim (int): 输入特征图的通道数。
        kernel (int): 卷积核大小，默认为 5。
        group (int): 分组数，默认为 2。
        H (bool): 是否为水平方向，默认为 True。
        """
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        # 根据 H 确定卷积核形状
        self.kernel = (1, kernel) if H else (kernel, 1)
        # 根据 H 确定填充形状
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        # 根据 H 确定填充方式
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        # 1x1 卷积层
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        # 自适应平均池化层
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # Sigmoid 激活函数
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 自适应平均池化
        filter = self.ap(x)
        # 1x1 卷积
        filter = self.conv(filter)
        n, c, h, w = x.shape
        # 填充并展开特征图
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        # 调整滤波器形状
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        # 应用 Sigmoid 激活函数
        filter = self.filter_act(filter)

        # 计算注意力加权输出
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class GlobalPoolStripAttention(nn.Module):
    def __init__(self, k) -> None:
        """
        初始化全局池化条带注意力模块。

        参数:
        k (int): 输入通道数。
        """
        super().__init__()

        self.channel = k
        # 垂直方向低通参数
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        # 垂直方向高通参数
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        # 水平方向低通参数
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        # 水平方向高通参数
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        # 水平自适应平均池化
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直自适应平均池化
        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))

        # 可学习的缩放因子
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        # 可学习的偏移因子
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 水平低通特征
        hori_l = self.hori_pool(x)
        # 水平高通特征
        hori_h = x - hori_l

        # 水平方向加权输出
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        # 垂直低通特征
        vert_l = self.vert_pool(hori_out)
        # 垂直高通特征
        vert_h = hori_out - vert_l

        # 垂直方向加权输出
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma


class LocalPoolStripAttention(nn.Module):
    def __init__(self, k, kernel=7) -> None:
        """
        初始化局部池化条带注意力模块。

        参数:
        k (int): 输入通道数。
        kernel (int): 池化核大小，默认为 7。
        """
        super().__init__()

        self.channel = k
        # 垂直方向低通参数
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        # 垂直方向高通参数
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        # 水平方向低通参数
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        # 水平方向高通参数
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        # 水平平均池化
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)
        # 垂直平均池化
        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)

        pad_size = kernel // 2
        # 水平填充
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))
        # 垂直填充
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))

        # 可学习的缩放因子
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        # 可学习的偏移因子
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 水平低通特征
        hori_l = self.hori_pool(self.pad_hori(x))
        # 水平高通特征
        hori_h = x - hori_l

        # 水平方向加权输出
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h

        # 垂直低通特征
        vert_l = self.vert_pool(self.pad_vert(hori_out))
        # 垂直高通特征
        vert_h = hori_out - vert_l

        # 垂直方向加权输出
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma


class SpecAtte(nn.Module):
    def __init__(self, k) -> None:
        """
        初始化光谱注意力模块。

        参数:
        k (int): 输入通道数。
        """
        super().__init__()
        # 全局池化条带注意力模块
        self.global_att = GlobalPoolStripAttention(k)
        # 7 核局部池化条带注意力模块
        self.local_att_7 = LocalPoolStripAttention(k, kernel=7)
        # 11 核局部池化条带注意力模块
        self.local_att_11 = LocalPoolStripAttention(k, kernel=11)
        # 1x1 卷积层
        self.conv = nn.Conv2d(k, k, 1)

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        # 全局注意力输出
        global_out = self.global_att(x)
        # 7 核局部注意力输出
        local_7_out = self.local_att_7(x)
        # 11 核局部注意力输出
        local_11_out = self.local_att_11(x)

        # 合并注意力输出
        out = global_out + local_7_out + local_11_out

        return self.conv(out)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_ResBlock(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ResBlock(c_,c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_ResBlock(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_LLSKM特征融合，为false的时候我们使用普通的Bottleneck提取特征
class SFCAM(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    # 输入通道数
    in_channel = 64
    # 输出通道数
    out_channel = 64
    # 创建残差块实例
    res_block = ResBlock(in_channel, out_channel, filter=True)
    # 生成随机输入张量
    x = torch.randn(1, in_channel, 27, 32)
    # 前向传播
    output = res_block(x)
    print("输出张量形状:", output.shape)
