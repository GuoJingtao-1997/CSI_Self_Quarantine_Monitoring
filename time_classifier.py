'''
Author: Guo Jingtao
Date: 2022-04-21 11:44:53
LastEditTime: 2022-06-04 22:13:55
LastEditors: Guo Jingtao
Description: 
FilePath: /undefined/Users/guojingtao/Documents/CSI_DataProcess/HumanActivityCode/time_classifier.py

'''
import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = [
    'TUNet',
    'BiLSTMClassifier',
    'LSTMClassifier',
    'ConvBiLSTMClassifier',
    'AttentionBiLSTMClassifier',
    'InceptionModel',
    'BasicBlock',
    'ResNet',
    'FCNBaseline',
    'LSTM_FCN',
    'GRU_FCN',
    'MLSTM_FCN',
    'MGRU_FCN',
    'gMLP',
]

# ====================================================================================================================

# sub-parts of the TU-Net model


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, linear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = F.interpolate(x1, scale_factor=2, mode='linear', align_corners=True)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class TUNet(nn.Module):
    def __init__(self, n_channels=234, n_classes_act=5, n_classes_per=3):
        super().__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc_act = outconv(64, n_classes_act)
        self.outc_per = outconv(64, n_classes_per)

    def forward(self, x):
        x1 = self.inc(x.squeeze_(dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_act = self.outc_act(x)
        x_per = self.outc_per(x)
        return x_per.mean(dim=-1), x_act.mean(dim=-1)


# ====================================================================================================================


# WiFi CSI Based Passive Human Activity Recognition Using Attention Based BLSTM
class AttentionBiLSTMClassifier(nn.Module):
    def __init__(
        self, emb_features, n_classes_act=5, n_classes_per=3, dropout=0.0
    ) -> None:
        super().__init__()
        self.dropout = dropout
        # self.emb_features = emb_features

        hidden_channel = 200
        self.blstm = nn.LSTM(
            input_size=emb_features,
            hidden_size=hidden_channel,
            num_layers=1,
            # dropout=dropout,
            bidirectional=True,
        )

        self.c_act = nn.Linear(hidden_channel * 2, n_classes_act)
        self.c_per = nn.Linear(hidden_channel * 2, n_classes_per)

    def attention_net(self, blstm_out, final_state):
        # blstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(blstm_out.permute(1, 0, 2), hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        # scaling = self.emb_features ** (1 / 2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        # attn_out: [batch_size, n_hidden * num_directions(=2)]
        attn_out = torch.bmm(
            blstm_out.permute(1, 2, 0), soft_attn_weights.unsqueeze(2)
        ).squeeze(2)

        return attn_out

    def forward(self, x):
        x = x.squeeze_(dim=1).permute(
            2, 0, 1
        )  # input : [seq_len, batch_size, embedding_dim]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # output : [seq_len, batch_size, n_hidden * num_directions(=2)]
        blstm_out, (final_hidden_state, final_cell_state) = self.blstm(x)
        # out = out[-1, :, :]
        attn_out = self.attention_net(blstm_out, final_hidden_state)
        if self.dropout > 0.0:
            blstm_out = F.dropout(blstm_out, p=self.dropout, training=self.training)
        out_act = self.c_act(attn_out)
        out_per = self.c_per(attn_out)
        return out_per, out_act


# ====================================================================================================================


# WiFi CSI based HAR
class BiLSTMClassifier(nn.Module):
    def __init__(
        self, encoder_features, n_classes_act=5, n_classes_per=3, dropout=0.2
    ) -> None:
        super().__init__()
        self.dropout = dropout

        hidden_channel = 512
        self.blstm = nn.LSTM(
            input_size=encoder_features,
            hidden_size=hidden_channel // 2,
            num_layers=1,
            # dropout=dropout,
            bidirectional=True,
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channel, hidden_channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channel // 2, hidden_channel // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.c_act = nn.Linear(hidden_channel // 4, n_classes_act)
        self.c_per = nn.Linear(hidden_channel // 4, n_classes_per)

    def forward(self, x):
        x = x.squeeze_(dim=1).permute(2, 0, 1)
        blstm_out, (final_hidden_state, final_cell_state) = self.blstm(x)
        blstm_out = torch.cat([final_hidden_state[0], final_hidden_state[1]], dim=-1)
        linear_out = self.linear(blstm_out)
        out_act = self.c_act(linear_out)
        out_per = self.c_per(linear_out)
        return out_per, out_act


# ====================================================================================================================


# A Survey on Behavior Recognition Using Wi-Fi CSI
class LSTMClassifier(nn.Module):
    def __init__(
        self, encoder_features, n_classes_act=5, n_classes_per=3, dropout=0.0
    ) -> None:
        super().__init__()
        self.dropout = dropout

        hidden_channel = 200
        self.lstm = nn.LSTM(
            input_size=encoder_features,
            hidden_size=hidden_channel,
            num_layers=1,
            # dropout=dropout,
            bidirectional=False,
        )

        self.c_act = nn.Linear(hidden_channel, n_classes_act)
        self.c_per = nn.Linear(hidden_channel, n_classes_per)

    def forward(self, x):
        x = x.squeeze_(dim=1).permute(2, 0, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[-1, :, :]
        if self.dropout > 0.0:
            lstm_out = F.dropout(lstm_out, p=self.dropout, training=self.training)
        out_act = self.c_act(lstm_out)
        out_per = self.c_per(lstm_out)
        return out_per, out_act


# ====================================================================================================================


# Wifi-based human activity recognition using raspberry pi
class ConvBiLSTMClassifier(nn.Module):
    def __init__(
        self, encoder_features, n_classes_act=5, n_classes_per=3, dropout=0.0
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convmax = nn.Sequential(
            nn.Conv1d(encoder_features, 128, 5),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        hidden_channel = 200
        self.blstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_channel,
            num_layers=4,
            dropout=dropout,
            bidirectional=True,
        )
        self.c_act = nn.Linear(hidden_channel * 2, n_classes_act)
        self.c_per = nn.Linear(hidden_channel * 2, n_classes_per)

    def forward(self, x):
        x = x.squeeze_(dim=1)
        x = self.convmax(x)
        x = x.permute(2, 0, 1)
        blstm_out, (final_hidden_state, final_cell_state) = self.blstm(x)
        blstm_out = torch.cat([final_hidden_state[0], final_hidden_state[1]], dim=-1)
        if self.dropout > 0.0:
            blstm_out = F.dropout(blstm_out, p=self.dropout, training=self.training)
        out_act = self.c_act(blstm_out)
        out_per = self.c_per(blstm_out)
        return out_per, out_act


# ====================================================================================================================


# Fully Convolutional Neural Networks (FCNs)
class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"

    def __init__(self, out_size=1):
        super().__init__()
        self.ap1d = nn.AdaptiveAvgPool1d(out_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.ap1d(x))


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        act=nn.ReLU,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            act(inplace=True) if act is not None else nn.Sequential(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)


class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/pdf/1611.06455.pdf
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, n_classes_act=5, n_classes_per=3) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'n_classes_act': n_classes_act,
            'n_classes_per': n_classes_per,
        }

        self.layers = nn.Sequential(
            *[
                ConvBlock(in_channels, 128, 8, 1),
                ConvBlock(128, 256, 5, 1),
                ConvBlock(256, 128, 3, 1),
            ]
        )
        self.gap = GAP1d(1)
        self.final_act = nn.Linear(128, n_classes_act)
        self.final_per = nn.Linear(128, n_classes_per)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x.squeeze_(dim=1)
        x = self.layers(x)
        # return self.final(x.mean(dim=-1))
        # print("X: ", x.shape)
        # print("X: ", x.mean(dim=-1).shape)
        # print("X: ", x.mean(dim=0).shape)
        # print("X: ", x.mean(dim=1).shape)
        x = self.gap(x)
        out_act = self.final_act(x)
        out_per = self.final_per(x)
        return out_per, out_act


# ====================================================================================================================


class SqueezeExciteBlock(nn.Module):
    def __init__(self, ni, reduction=16):
        super().__init__()
        self.avg_pool = GAP1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ni, ni // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(ni // reduction, ni, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y).unsqueeze(2)
        return x * y.expand_as(x)


class _RNN_FCN_Base(nn.Module):
    def __init__(
        self,
        c_in,
        n_classes_act=5,
        n_classes_per=3,
        hidden_size=100,
        rnn_layers=1,
        bias=True,
        cell_dropout=0,
        rnn_dropout=0.8,
        bidirectional=False,
        fc_dropout=0.0,
        conv_layers=[128, 256, 128],
        kss=[8, 5, 3],
        se=0,
    ):
        super().__init__()
        # RNN
        self.rnn = self._cell(
            c_in,
            hidden_size,
            num_layers=rnn_layers,
            bias=bias,
            dropout=cell_dropout,
            bidirectional=bidirectional,
        )
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else nn.Sequential()

        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0], 1)
        self.se1 = (
            SqueezeExciteBlock(conv_layers[0], se) if se != 0 else nn.Sequential()
        )
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1], 1)
        self.se2 = (
            SqueezeExciteBlock(conv_layers[1], se) if se != 0 else nn.Sequential()
        )
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2], 1)
        self.gap = GAP1d(1)

        # Common
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Sequential()
        self.fc_per = nn.Linear(
            hidden_size * (1 + bidirectional) + conv_layers[-1], n_classes_per
        )
        self.fc_act = nn.Linear(
            hidden_size * (1 + bidirectional) + conv_layers[-1], n_classes_act
        )

    def forward(self, x):
        # RNN
        x = x.squeeze_(dim=1)
        rnn_input = x.permute(
            2, 0, 1
        )  # permute --> (seq_len, batch_size, n_vars) when batch_first=False
        output, _ = self.rnn(rnn_input)
        last_out = output[-1, :, :]  # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)

        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)

        # Concat
        x = torch.cat([last_out, x], dim=1)
        x = self.fc_dropout(x)
        out_per = self.fc_per(x)
        out_act = self.fc_act(x)
        return out_per, out_act


class LSTM_FCN(_RNN_FCN_Base):
    _cell = nn.LSTM


class GRU_FCN(_RNN_FCN_Base):
    _cell = nn.GRU


class MLSTM_FCN(_RNN_FCN_Base):
    _cell = nn.LSTM

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=se, **kwargs)


class MGRU_FCN(_RNN_FCN_Base):
    _cell = nn.GRU

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=se, **kwargs)


# ====================================================================================================================

# InceptionTime

# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019).
# InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()
        ks = [ks // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = (
            nn.Conv1d(ni, nf, 1, 1, 0, bias=False) if bottleneck else nn.Sequential()
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    nf if bottleneck else ni,
                    nf,
                    k,
                    1,
                    k // 2,
                    bias=False,
                )
                for k in ks
            ]
        )
        self.maxconvpool = nn.Sequential(
            *[
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(ni, nf, 1, 1, 0, bias=False),
            ]
        )

        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = torch.cat(
            [l(x) for l in self.convs] + [self.maxconvpool(input_tensor)], dim=1
        )
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=32, residual=True, depth=6, **kwargs):
        super().__init__()
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(
                InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs)
            )
            if self.residual and d % 3 == 1:
                n_in, n_out = ni if d == 1 else nf * 4, nf * 4
                self.shortcut.append(
                    nn.BatchNorm1d(n_in)
                    if n_in == n_out
                    else ConvBlock(n_in, n_out, 1, 1, act=None)
                )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        for d in range(self.depth):
            if d == 2:
                res = x
            x = self.inception[d](x)
            if self.residual and d % 3 == 1:
                x = self.act(x + self.shortcut[d // 3](res))
        return x


class InceptionModel(nn.Module):
    def __init__(
        self, c_in, n_classes_act=5, n_classes_per=3, nf=32, nb_filters=None, **kwargs
    ):
        super().__init__()
        nf = nb_filters if nf is None else nf  # for compatibility
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        self.gap = GAP1d(1)
        self.fc_per = nn.Linear(nf * 4, n_classes_per)
        self.fc_act = nn.Linear(nf * 4, n_classes_act)

    def forward(self, x):
        x = self.inceptionblock(x.squeeze_(dim=1))
        x = self.gap(x)
        out_per = self.fc_per(x)
        out_act = self.fc_act(x)
        return out_per, out_act


# ====================================================================================================================

# ResNet-1D


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, inchannel=234, activity_num=5, person_num=3):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(
            inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(
                512 * block.expansion,
                512 * block.expansion,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, activity_num)

        self.PERClassifier = nn.Sequential(
            nn.Conv1d(
                512 * block.expansion,
                512 * block.expansion,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.per_fc = nn.Linear(512 * block.expansion, person_num)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        x = self.conv1(x.squeeze_(dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        act = self.ACTClassifier(c4)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        per = self.PERClassifier(c4)
        per = per.view(per.size(0), -1)
        per1 = self.per_fc(per)

        return per1, act1


# ====================================================================================================================

# GatedMLP


class _SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class _gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = _SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class _gMLPBackbone(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, depth=6):
        super().__init__()
        self.model = nn.Sequential(
            *[_gMLPBlock(d_model, d_ffn, seq_len) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)


class gMLP(_gMLPBackbone):
    def __init__(
        self,
        c_in,
        seq_len,
        n_classes_act=5,
        n_classes_per=3,
        patch_size=1,
        d_model=256,
        d_ffn=512,
        depth=6,
    ):
        assert seq_len % patch_size == 0, "`seq_len` must be divisibe by `patch_size`"
        super().__init__(d_model, d_ffn, seq_len // patch_size, depth)
        self.patcher = nn.Conv1d(
            c_in, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.head_per = nn.Linear(d_model, n_classes_per)
        self.head_act = nn.Linear(d_model, n_classes_act)

    def forward(self, x):
        patches = self.patcher(x.squeeze_(dim=1))
        batch_size, num_channels, _ = patches.shape
        patches = patches.permute(0, 2, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out_per = self.head_per(embedding)
        out_act = self.head_act(embedding)
        return out_per, out_act
