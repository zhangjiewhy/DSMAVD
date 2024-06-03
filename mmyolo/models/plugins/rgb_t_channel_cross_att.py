from mmengine.model import BaseModule
import torch
import torch.nn as nn
from mmdet.models.utils import multi_apply
from mmcv.cnn import ConvModule
from torch.nn.modules.utils import _pair
from mmcv.cnn import Linear, build_activation_layer, build_conv_layer,build_norm_layer
from mmcv.cnn import Linear, build_activation_layer, build_conv_layer,build_norm_layer

class yolov8_dcn_channel_abs_com_dc(BaseModule):
    def __init__(self,inchannel,embed_dim,pool_h,pool_w,channel_scale,feature_scale):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # requires_grad=True
        self.act_cfg = dict(type='SiLU', inplace=True)
        print("channel_scale---cs:")
        print(channel_scale)
        print("feature_scale---ss:")
        print(feature_scale)

        self.pool_h = pool_h//feature_scale
        self.pool_w = pool_w//feature_scale
        self.channel_scale = channel_scale
        self.feature_scale = feature_scale
        if self.channel_scale != 1:
            self.x_conv = ConvModule(
                in_channels=inchannel,
                out_channels=inchannel//channel_scale,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.xi_conv = ConvModule(
                in_channels=inchannel,
                out_channels=inchannel//channel_scale,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        if self.feature_scale != 1:
            self.max_pool = nn.AdaptiveMaxPool2d((self.pool_h,self.pool_w))
            self.avg_pool = nn.AdaptiveAvgPool2d((self.pool_h,self.pool_w))

        self.ln_norm_cfg = dict(type='LN')
        self.LN_max_x = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        self.LN_max_xi = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        self.LN_avg_x = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        self.LN_avg_xi = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]

    def forward(self, x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor: #forward
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        if self.channel_scale != 1:
            x = self.x_conv(x)
            xi = self.xi_conv(xi)

        if self.feature_scale != 1:
            x_max= self.max_pool(x)
            xi_max = self.max_pool(xi)
        else:
            x_max = x
            xi_max = xi

        b, c, h, w = xi_max.size()
        x_max = x_max.view(m_batchsize, c,-1)
        xi_max =xi_max.view(m_batchsize, c,-1)

        att=xi_max.unsqueeze(dim=2)-x_max.unsqueeze(dim=1) #(XI-X)
        energy_v = (att.abs().sum(dim=-1))/(self.pool_h*self.pool_w)

        attention_v_com = self.softmax(energy_v)
        proj_value_v = x.view(m_batchsize, -1, height*width)
        max_v_com = torch.bmm(attention_v_com, proj_value_v)

        energy_i = energy_v.permute(0, 2, 1)
        attention_i_com = self.softmax(energy_i)
        proj_value_i = xi.view( m_batchsize, -1, height*width)
        max_i_com = torch.bmm(attention_i_com, proj_value_i)

#########avg  pool
        if self.feature_scale != 1:
            x_avg = self.avg_pool(x)
            xi_avg = self.avg_pool(xi)
        else:
            x_avg = x
            xi_avg = xi

        x_avg = x_avg.view(m_batchsize, c, -1)
        xi_avg = xi_avg.view(m_batchsize, c, -1)
        att_avg = xi_avg.unsqueeze(dim=2) - x_avg.unsqueeze(dim=1)  # (XI-X)
        energy_v_avg = (att_avg.abs().sum(dim=-1)) /(self.pool_h*self.pool_w)

        attention_v_com_avg = self.softmax(energy_v_avg)
        avg_v_com = torch.bmm(attention_v_com_avg, proj_value_v)

        energy_i_avg = energy_v_avg.permute(0, 2, 1)
        attention_i_com_avg = self.softmax(energy_i_avg)
        avg_i_com = torch.bmm(attention_i_com_avg, proj_value_i)


        max_v_com = self.LN_max_x(max_v_com).view(m_batchsize, -1, height, width)
        max_i_com = self.LN_max_xi(max_i_com).view(m_batchsize, -1, height, width)
        avg_v_com = self.LN_avg_x(avg_v_com).view(m_batchsize, -1, height, width)
        avg_i_com = self.LN_avg_xi(avg_i_com).view(m_batchsize, -1, height, width)

        return  max_v_com, max_i_com, avg_v_com, avg_i_com



