import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.ops import DeformConv2d
from torch.nn.modules.utils import _pair

class DCN_align_method(BaseModule):
    def __init__(self,inchannel: int):
        super().__init__()
        self.inchannel = inchannel
        self.xi_channel= int(inchannel/2)
        self.kernel_size=3

        # self.conv_offset1 = nn.Conv2d(
        #     in_channels=self.inchannel,
        #     out_channels=int(self.inchannel/2),
        #     kernel_size=3,
        #     stride=_pair(1),
        #     padding=_pair(1),
        #     dilation=_pair(1),
        #     bias=True)
        # self.conv_offset1.weight.data.zero_()
        # self.conv_offset1.bias.data.zero_()
        #
        # self.conv_offset2 = nn.Conv2d(
        #     in_channels= int(self.inchannel/2),
        #     out_channels=2*self.kernel_size * self.kernel_size,
        #     kernel_size=3,
        #     stride=_pair(1),
        #     padding=_pair(1),
        #     dilation=_pair(1),
        #     bias=True)
        # self.conv_offset2.weight.data.zero_()
        # self.conv_offset2.bias.data.zero_()

        self.conv_offset = nn.Conv2d(
            in_channels=self.inchannel ,
            out_channels=2 * self.kernel_size * self.kernel_size,
            kernel_size=3,
            stride=_pair(1),
            padding=_pair(1),
            dilation=_pair(1),
            bias=True)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

        self.xi_dconv = DeformConv2d(
            self.xi_channel,  # 输入通道
            self.xi_channel,  # 输出通道
            kernel_size=self.kernel_size,  # 卷积核大小
            stride=1,  # stride
            padding=1)  # padding

        # self.bn = build_norm_layer(dict(type='BN', momentum=0.03, eps=0.001),self.xi_channel)[1]
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:

        # offset1 = self.conv_offset1(torch.cat((x, x_i), dim=1))
        # offset = self.conv_offset2(offset1)

        offset = self.conv_offset(torch.cat((x, x_i), dim=1))

        # x_i = self.act(self.bn(self.xi_dconv(x_i, offset)))
        x_i = self.act(self.xi_dconv(x_i, offset))
        return x_i