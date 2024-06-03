from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear, build_activation_layer, build_conv_layer,build_norm_layer
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import CSPLayerWithTwoConv, SPPFBottleneck
from ..utils import make_divisible, make_round
from .base_backbone import BaseBackbone
from ..detectors.align_method import DCN_align_method
from ..plugins.rgb_t_channel_cross_att import yolov8_dcn_channel_abs_com_dc
@MODELS.register_module()
class YOLOv8_dcn_channel_att(BaseBackbone):
    """CSP-Darknet backbone used in YOLOv8.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv8CSPDarknet
        >>> import torch
        >>> model = YOLOv8CSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 inchannel=None,
                 embed_dim=None,
                 feature_scale=None,
                 channel_scale=None,
                 pool_h=None,
                 pool_w=None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

        self.stem_i = self.build_stem_layer()
        self.layers_i = ['stem_i']
        for idx, setting in enumerate(self.arch_settings[arch]):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            if plugins is not None:
                stage += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f'stage{idx + 1}_i', nn.Sequential(*stage))
            self.layers_i.append(f'stage{idx + 1}_i')
        self.channel_att=nn.ModuleList()
        for i in range(len(inchannel)):
            self.channel_att.append(yolov8_dcn_channel_abs_com_dc(inchannel=inchannel[i],embed_dim=embed_dim[i],pool_h=pool_h[i],pool_w=pool_w[i],channel_scale=channel_scale,feature_scale=feature_scale))
        self.max_conv_x = nn.ModuleList()
        self.max_conv_xi = nn.ModuleList()
        self.norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # requires_grad=True
        self.act_cfg = dict(type='SiLU', inplace=True)
        jiangwei_channnel_in = [inchannel[0]+2*(inchannel[0]//channel_scale),inchannel[1]+2*(inchannel[1]//channel_scale),inchannel[2]+2*(inchannel[2]//channel_scale)]

        for i in range(len(inchannel)):   #cat模式  avg  max 原特征拼接
            self.max_conv_x.append(ConvModule(
                jiangwei_channnel_in[i],
                inchannel[i],
                kernel_size=3,  # 3
                stride=1,
                padding=1,  # 1
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
            self.max_conv_xi.append(ConvModule(
                jiangwei_channnel_in[i],
                inchannel[i],
                kernel_size=3,  # 3
                stride=1,
                padding=1,  # 1
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def DCN_align(self):
        align_method = nn.ModuleList()
        # inchannel = [128,256, 512]
        inchannel = [256, 512, 1024]
        for i in range(len(inchannel)):
            align_method.append(DCN_align_method(inchannel=inchannel[i]))
        return align_method


    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()


    def forward(self, x: torch.Tensor,x_i: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        outs_i = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            layer_name_i = layer_name + '_i'
            layer_i=getattr(self, layer_name_i)

            x = layer(x)
            x_i = layer_i(x_i)

            if i in self.out_indices:
                outs_i.append(x_i)
                outs.append(x)

            if i > 0 and i < 4:
                max_v_com, max_i_com, avg_v_com, avg_i_com = self.channel_att[i-1](x, x_i)   #拼接模式
                x=self.max_conv_x[i-1](torch.cat([x,max_i_com,avg_i_com],dim=1))         #1:avg+max
                x_i=self.max_conv_xi[i-1](torch.cat([x_i,max_v_com,avg_v_com],dim=1))    #1:avg+max

        return tuple(outs),tuple(outs_i)




