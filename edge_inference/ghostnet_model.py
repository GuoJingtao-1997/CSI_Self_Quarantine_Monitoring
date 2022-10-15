# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import math
import torch
import torch.nn as nn

__all__ = ['BranchyGhostnet']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.0) / 6.0


class hard_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = hard_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn()
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GlobalAvgPool1d(nn.Module):
    """
    Reduce mean over one dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dim: int):
        return x.mean(dim=dim, keepdim=True)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super().__init__()
        self.pool = GlobalAvgPool1d()

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = hard_swish()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x_h = self.pool(x, dim=-1)
        x_w = self.pool(x, dim=-2).permute(0, 1, 3, 2)

        x_cat = torch.cat([x_h, x_w], dim=2)
        x_cat = self.relu(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        #x_h = x_h.expand(-1, -1, h, w)
        #x_w = x_w.expand(-1, -1, h, w)

        return identity * x_h * x_w


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        act_layer=nn.ReLU,
        se_ratio=0.0,
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        self.conv_dw1 = nn.Conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            padding=(dw_kernel_size - 1) // 2,
            groups=mid_chs,
            bias=False,
        )
        self.bn_dw1 = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            #self.se = SqueezeExcite(in_chs, se_ratio=se_ratio)
            self.se = CoordAtt(mid_chs, mid_chs)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw1(x)
            x = self.bn_dw1(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(
        self,
        cfgs_fe,
        cfgs_fo,
        branch_classes=1000,
        final_classes=1000,
        #exit_threshold=0.5,
        width=1.0,
        dropout=0.2,
    ):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs_fe = cfgs_fe
        self.cfgs_fo = cfgs_fo
        self.dropout = dropout
        self.fast_inference_mode = False
        #self.exit_threshold = torch.tensor([exit_threshold], dtype=torch.float32)
        self.backbone = nn.ModuleList()
        self.exit_branch = nn.ModuleList()

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        conv_bn_act = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        # first backbone
        stages_fe = []
        for cfg in self.cfgs_fe:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                    )
                )
                input_channel = output_channel
            stages_fe.append(nn.Sequential(*layers))
        output_channel = hidden_channel
        blocks_fe = nn.Sequential(*stages_fe)
        self.backbone.append(nn.Sequential(conv_bn_act, blocks_fe))

        branch_channel = 1280
        # early exit branch
        early_exit_branch = nn.Sequential(
            ConvBnAct(input_channel, output_channel, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(output_channel, branch_channel, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(branch_channel, branch_classes)
        )

        self.exit_branch.append(early_exit_branch)

        # second backbone
        stages_fo = []
        for cfg in self.cfgs_fo:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                    )
                )
                input_channel = output_channel
            stages_fo.append(nn.Sequential(*layers))

        output_channel = hidden_channel
        stages_fo.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        blocks_fo = nn.Sequential(*stages_fo)
        self.backbone.append(blocks_fo)

        output_channel = 1280
        # building last several layers for final exit
        final_exit = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(output_channel, final_classes)
        )

        self.exit_branch.append(final_exit)

        self._initialize_weights()

    def exit_criterion_loss(self, x):
        with torch.no_grad():
            y = nn.functional.softmax(x, dim=1)
            loss = -torch.sum(y * torch.log(y), dim=1)
            return loss < self.exit_threshold

    def exit_criterion_acc(self, x):
        with torch.no_grad():
            one_idx = torch.tensor([1], dtype=torch.float32)
            _, idx = torch.max(nn.functional.softmax(x, dim=-1), dim=-1)
            return torch.ne(idx, one_idx) 

    # @torch.jit.unused
    # def _forward_train(self, x):
    #     res = []
    #     for bone, branch in zip(self.backbone, self.exit_branch):
    #             x = bone(x)
    #             res.append(branch(x))
    #     return res

    def forward(self, x):
        res = []
        if self.fast_inference_mode:
            for i, (bone, branch) in enumerate(zip(self.backbone, self.exit_branch)):
                x = bone(x)
                y = branch(x)
                if i == 0 and self.exit_criterion_acc(y):
                    return [y, torch.tensor(0.)]
                res.append(y)
            return res
        else:
            for bone, branch in zip(self.backbone, self.exit_branch):
                x = bone(x)
                res.append(branch(x))
        return res

    def set_fast_infer_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def BranchyGhostnet(**kwargs):
    """
    Constructs a GhostNet Encoder
    """
    # early exit backbone
    cfgs_fe = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
        ],
    ]
    # final exit backbone
    cfgs_fo = [
        # k, t, c, SE, s
        # stage4
        [
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
        ],
    ]
    return GhostNet(cfgs_fe, cfgs_fo, **kwargs)
