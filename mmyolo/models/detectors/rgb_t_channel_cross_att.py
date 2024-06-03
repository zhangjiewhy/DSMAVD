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
            x = self.x_conv(x)    #降维度
            xi = self.xi_conv(xi) #降维度

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
        energy_v = (att.abs().sum(dim=-1))/(self.pool_h*self.pool_w) # (b,c,c）
                              # energy_new_v = torch.max(energy_v, -1, keepdim=True)[0].expand_as(energy_v) - energy_v  # 得到（b,c,c）

        attention_v_com = self.softmax(energy_v)
                             # attention_v_dc = self.softmax(energy_new_v)

        proj_value_v = x.view(m_batchsize, -1, height*width)  # 得到（b,c,w*h）
        max_v_com = torch.bmm(attention_v_com, proj_value_v) # 得到（b,c,w*h） 与红外差异较大的可见光特征  .view(m_batchsize, -1, height, width)
                             # max_v_dc = torch.bmm(attention_v_dc, proj_value_v)  # 得到（b,c,w*h）   与红外差异较小的可见光特征

        energy_i = energy_v.permute(0, 2, 1)
                             # energy_new_i = torch.max(energy_i, -1, keepdim=True)[0].expand_as(energy_i) - energy_i  # 得到（b,c,c）
        attention_i_com = self.softmax(energy_i)
                             # attention_i_dc = self.softmax(energy_new_i)
        #
        proj_value_i = xi.view( m_batchsize, -1, height*width)  # 得到（b,c,w*h）
        max_i_com = torch.bmm(attention_i_com, proj_value_i)  # 得到（b,c,w*h） 与可见光差异较大的红外特征
                         # max_i_dc = torch.bmm(attention_i_dc, proj_value_i) # 得到（b,c,w*h）    可见光差异较小的红外特征

#########avg  pool
        if self.feature_scale != 1:
            x_avg = self.avg_pool(x)
            xi_avg = self.avg_pool(xi)
        else:
            x_avg = x
            xi_avg = xi
        # b, c, h, w = xi_avg.size()

        x_avg = x_avg.view(m_batchsize, c, -1)
        xi_avg = xi_avg.view(m_batchsize, c, -1)
        att_avg = xi_avg.unsqueeze(dim=2) - x_avg.unsqueeze(dim=1)  # (XI-X)

        energy_v_avg = (att_avg.abs().sum(dim=-1)) /(self.pool_h*self.pool_w) # (b,c,c）
                       # energy_new_avg = torch.max(energy_v_avg, -1, keepdim=True)[0].expand_as(energy_v_avg) - energy_v_avg  # 得到（b,c,c）

        attention_v_com_avg = self.softmax(energy_v_avg)
                      # attention_v_dc_avg = self.softmax(energy_new_avg)

        avg_v_com = torch.bmm(attention_v_com_avg, proj_value_v)  # 得到（b,c,w*h） 与红外差异较大的可见光特征
                         # avg_v_dc = torch.bmm(attention_v_dc_avg, proj_value_v)  # 得到（b,c,w*h）   与红外差异较小的可见光特征

        energy_i_avg = energy_v_avg.permute(0, 2, 1)
                          # energy_new_i_avg = torch.max(energy_i_avg, -1, keepdim=True)[0].expand_as(energy_i_avg) - energy_i_avg  # 得到（b,c,c）
        attention_i_com_avg = self.softmax(energy_i_avg)
                           # attention_i_dc_avg = self.softmax(energy_new_i_avg)

        avg_i_com = torch.bmm(attention_i_com_avg, proj_value_i)  # 得到（b,c,w*h） 与可见光差异较大的红外特征
                           # avg_i_dc = torch.bmm(attention_i_dc_avg, proj_value_i)  # 得到（b,c,w*h） 可见光差异较小的红外特征

        max_v_com = self.LN_max_x(max_v_com).view(m_batchsize, -1, height, width)
        max_i_com = self.LN_max_xi(max_i_com).view(m_batchsize, -1, height, width)
        avg_v_com = self.LN_avg_x(avg_v_com).view(m_batchsize, -1, height, width)
        avg_i_com = self.LN_avg_xi(avg_i_com).view(m_batchsize, -1, height, width)

        return  max_v_com, max_i_com, avg_v_com, avg_i_com

        # max_v_dc = self.LN_max_x(max_v_dc).view(m_batchsize, -1, height, width)
        # max_i_dc = self.LN_max_xi(max_i_dc).view(m_batchsize, -1, height, width)
        # avg_v_dc = self.LN_avg_x(avg_v_dc).view(m_batchsize, -1, height, width)
        # avg_i_dc = self.LN_avg_xi(avg_i_dc).view(m_batchsize, -1, height, width)
        # return max_v_dc, max_i_dc, avg_v_dc, avg_i_dc

class yolov8_COM_channel_ATT(BaseModule):
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
            self.x_Q = ConvModule(in_channels=inchannel,out_channels=inchannel//channel_scale,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
            self.xi_KT = ConvModule(in_channels=inchannel,out_channels=inchannel//channel_scale,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
            self.xi_V = ConvModule(in_channels=inchannel,out_channels=inchannel//channel_scale,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)

            self.xi_Q = ConvModule(in_channels=inchannel, out_channels=inchannel//channel_scale, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.x_KT = ConvModule(in_channels=inchannel, out_channels=inchannel//channel_scale, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.x_V = ConvModule(in_channels=inchannel, out_channels=inchannel//channel_scale, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

            self.x_conv_up = ConvModule(in_channels=inchannel//channel_scale, out_channels=inchannel, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.xi_conv_up = ConvModule(in_channels=inchannel//channel_scale, out_channels=inchannel, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        if self.feature_scale != 1:
            self.max_pool = nn.AdaptiveMaxPool2d((self.pool_h,self.pool_w))
            # self.avg_pool = nn.AdaptiveAvgPool2d((self.pool_h,self.pool_w))

        # self.ln_norm_cfg = dict(type='LN')
        # self.LN_max_x = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        # self.LN_max_xi = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        # self.LN_avg_x = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        # self.LN_avg_xi = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]

    def forward(self, x: torch.Tensor, xi: torch.Tensor,mask_i,mask_rgb,Dmask_abs) -> torch.Tensor: #forward
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        #得到新的X_i
        if self.channel_scale != 1:
            x_Q =  self.x_Q(x)      #降维度
            xi_KT = self.xi_KT(xi)   #降维度
            xi_V = self.xi_V(xi)
        else:
            x_Q = x  # 降维度
            xi_KT = xi  # 降维度
            xi_V = xi
        if self.feature_scale != 1:
            x_Q = self.max_pool(x_Q)
            xi_KT = self.max_pool(xi_KT)

            mask_i = self.max_pool(mask_i)
            mask_rgb = self.max_pool(mask_rgb)

        d_mask=[(mask_i.sigmoid()-mask_rgb.sigmoid()).abs() < Dmask_abs][0]

        b, c, h, w = x_Q.size()
        Q_rgb = (x_Q*d_mask).view(m_batchsize, c,-1)*((self.pool_h*self.pool_w)** -0.5)
        K_i =xi_KT.view(m_batchsize, c,-1)
        ATT=(torch.bmm(Q_rgb, K_i.transpose(1,2)))
        ATT = self.softmax(ATT)
        V_i = xi_V.view(m_batchsize, -1, height*width)  # 得到（b,c,w*h）
        x_i_new = torch.bmm(ATT, V_i).view(m_batchsize, -1, height, width) # 得到（b,c,w*h） 与红外差异较大的可见光特征  .view(m_batchsize, -1, height, width)


        # 得到新的X_rgb
        if self.channel_scale != 1:
            xi_Q = self.xi_Q(xi)  # 降维度
            x_KT = self.x_KT(x)  # 降维度
            x_V = self.x_V(x)
        else:
            xi_Q = xi  # 降维度
            x_KT = x  # 降维度
            x_V = x
        if self.feature_scale != 1:
            xi_Q = self.max_pool(xi_Q)
            x_KT = self.max_pool(x_KT)
        b, c, h, w = xi_Q.size()
        Q_i =  (xi_Q * d_mask).view(m_batchsize, c, -1)*((self.pool_h * self.pool_w) ** -0.5)
        K_rgb =  x_KT.view(m_batchsize, c, -1)
        ATT = (torch.bmm(Q_i, K_rgb.transpose(1, 2)))
        ATT = self.softmax(ATT)
        V_rgb = x_V.view(m_batchsize, -1, height * width)  # 得到（b,c,w*h）
        x_rgb_new = torch.bmm(ATT, V_rgb).view(m_batchsize, -1, height,width)  # 得到（b,c,w*h） 与红外差异较大的可见光特征  .view(m_batchsize, -1, height, width)

        if self.channel_scale != 1:
            x_i_new = self.xi_conv_up(x_i_new)   # 升维度
            x_rgb_new = self.x_conv_up(x_rgb_new)

        x = x + x_rgb_new
        xi = xi + x_i_new
        return  x, xi
class yolov8_COM_space_ATT(BaseModule):
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
            self.x_Q = ConvModule(in_channels=inchannel,out_channels=inchannel//channel_scale,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
            self.xi_KT = ConvModule(in_channels=inchannel,out_channels=inchannel//channel_scale,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
            self.xi_V = ConvModule(in_channels=inchannel,out_channels=inchannel//channel_scale,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)

            self.xi_Q = ConvModule(in_channels=inchannel, out_channels=inchannel//channel_scale, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.x_KT = ConvModule(in_channels=inchannel, out_channels=inchannel//channel_scale, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.x_V = ConvModule(in_channels=inchannel, out_channels=inchannel//channel_scale, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

            self.x_conv_up = ConvModule(in_channels=inchannel//channel_scale, out_channels=inchannel, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.xi_conv_up = ConvModule(in_channels=inchannel//channel_scale, out_channels=inchannel, kernel_size=1,stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        if self.feature_scale != 1:
            self.max_pool = nn.AdaptiveMaxPool2d((self.pool_h,self.pool_w))
            # self.avg_pool = nn.AdaptiveAvgPool2d((self.pool_h,self.pool_w))

        # self.ln_norm_cfg = dict(type='LN')
        # self.LN_max_x = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        # self.LN_max_xi = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        # self.LN_avg_x = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]
        # self.LN_avg_xi = build_norm_layer(self.ln_norm_cfg, embed_dim)[1]

    def forward(self, x: torch.Tensor, xi: torch.Tensor,mask_i,mask_rgb,Dmask_abs) -> torch.Tensor: #forward
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        #得到新的X_i
        if self.channel_scale != 1:
            x_Q =  self.x_Q(x)      #降维度
            xi_KT = self.xi_KT(xi)   #降维度
            xi_V = self.xi_V(xi)
        else:
            x_Q = x  # 降维度
            xi_KT = xi  # 降维度
            xi_V = xi
        if self.feature_scale != 1:
            xi_KT = self.max_pool(xi_KT)
            xi_V = self.max_pool(xi_V)

        d_mask=[(mask_i.sigmoid()-mask_rgb.sigmoid()).abs() < Dmask_abs][0]

        b, c, h, w = x_Q.size()
        Q_rgb = (x_Q*d_mask).view(m_batchsize, c,-1)
        Q_rgb = Q_rgb.transpose(1,2)

        K_i =xi_KT.view(m_batchsize, c,-1)
        ATT=(torch.bmm(Q_rgb,K_i))*((self.pool_h*self.pool_w)** -0.5)
        ATT = self.softmax(ATT)

        V_i = xi_V.view(m_batchsize, -1, height*width).transpose(1,2)  # 得到（b,w*h,c）
        x_i_new = torch.bmm(ATT, V_i).transpose(1,2)
        x_i_new=x_i_new.view(m_batchsize, -1, height, width) # 得到（b,c,w*h） 与红外差异较大的可见光特征  .view(m_batchsize, -1, height, width)


        # 得到新的X_rgb
        if self.channel_scale != 1:
            xi_Q = self.xi_Q(xi)  # 降维度
            x_KT = self.x_KT(x)  # 降维度
            x_V = self.x_V(x)
        else:
            xi_Q = xi  # 降维度
            x_KT = x  # 降维度
            x_V = x
        if self.feature_scale != 1:
            xi_Q = self.max_pool(xi_Q)
            x_KT = self.max_pool(x_KT)
        b, c, h, w = xi_Q.size()
        Q_i =  (xi_Q * d_mask).view(m_batchsize, c, -1)
        K_rgb =  x_KT.view(m_batchsize, c, -1)
        ATT = (torch.bmm(Q_i, K_rgb.transpose(1, 2))) * ((self.pool_h * self.pool_w) ** -0.5)
        ATT = self.softmax(ATT)
        V_rgb = x_V.view(m_batchsize, -1, height * width)  # 得到（b,c,w*h）
        x_rgb_new = torch.bmm(ATT, V_rgb).view(m_batchsize, -1, height,width)  # 得到（b,c,w*h） 与红外差异较大的可见光特征  .view(m_batchsize, -1, height, width)

        if self.channel_scale != 1:
            x_i_new = self.xi_conv_up(x_i_new)   # 升维度
            x_rgb_new = self.x_conv_up(x_rgb_new)

        x = x + x_rgb_new
        xi = xi + x_i_new
        return  x, xi



class yolov8_dcn_channel_att_com_dc(BaseModule):  # 11-11    LN
    # def __init__(self,inchannel:int,outchannel:int):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:  # forward
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        scaling = float(height * width) ** -0.5

        xi = xi.contiguous().view(m_batchsize, C, -1)  # 得到（b,c,w*h）
        x = x.contiguous().view(m_batchsize, C, -1)  # 得到（b,c,w*h）

        proj_query_v = xi * scaling  # 得到（b,c,w*h）
        proj_key_v = x.permute(0, 2, 1)  # 得到（b,w*h,c）
        energy_v = torch.bmm(proj_query_v, proj_key_v)  # .to(torch.float32)  # 得到（b,c,c）
        energy_new_v = torch.max(energy_v, -1, keepdim=True)[0].expand_as(energy_v) - energy_v  # 得到（b,c,c）

        attention_v_com = self.softmax(energy_v)
        attention_v_dc = self.softmax(energy_new_v)

        proj_value_v = x  # 得到（b,c,w*h）
        out_v_com = torch.bmm(attention_v_com, proj_value_v).view(m_batchsize, C, height, width)  # 得到（b,c,w*h） 可见光共有特征
        out_v_dc = torch.bmm(attention_v_dc, proj_value_v).view(m_batchsize, C, height, width) # 得到（b,c,w*h）   可见光特有特征


        energy_i = energy_v.permute(0, 2, 1)
        energy_new_i = torch.max(energy_i, -1, keepdim=True)[0].expand_as(energy_i) - energy_i  # 得到（b,c,c）
        attention_i_com = self.softmax(energy_i)
        attention_i_dc = self.softmax(energy_new_i)

        proj_value_i = xi  # 得到（b,c,w*h）
        out_i_com = torch.bmm(attention_i_com, proj_value_i).view(m_batchsize, C, height, width) # 得到（b,c,w*h）
        out_i_dc = torch.bmm(attention_i_dc, proj_value_i).view(m_batchsize, C, height, width)# 得到（b,c,w*h）

        return  out_v_com, out_v_dc, out_i_com, out_i_dc


class pool_channel_attention(BaseModule):  # 11-11    LN
    # def __init__(self,inchannel:int,outchannel:int):
    def __init__(self,
                 channels: int,
                 reduce_ratio: int = 8,
                 # act_cfg: dict = dict(type='ReLU')):
                 act_cfg: dict = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)
        self.fc_max1 = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))

        self.fc_max2 = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))

        self.fc_avg1 = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))

        self.fc_avg2 = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))
    def forward(self, x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:  # forward
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        maxpool_out_x = self.fc_max1(self.max_pool(x))
        maxpool_out_xi = self.fc_max2(self.max_pool(xi))
        maxpool_out_x=maxpool_out_x.squeeze(dim=-1)
        maxpool_out_xi=maxpool_out_xi.squeeze(dim=-1)
        max_attention_x_xi =  torch.bmm(maxpool_out_x,maxpool_out_xi.permute(0, 2, 1))
        max_attention_xi_x = max_attention_x_xi.permute(0, 2, 1)

        max_attention_x_xi = self.softmax(max_attention_x_xi)
        max_attention_xi_x = self.softmax(max_attention_xi_x)

        xi_max=  torch.bmm(max_attention_x_xi,xi.view(m_batchsize, C, height*width)).view(m_batchsize, C, height,width)
        x_max = torch.bmm(max_attention_xi_x, x.view(m_batchsize, C, height*width)).view(m_batchsize, C, height,width)

        avgpool_out_x = self.fc_avg1(self.avg_pool(x)).squeeze(dim=-1)
        avgpool_out_xi = self.fc_avg2(self.avg_pool(xi)).squeeze(dim=-1)
        avg_attention_x_xi = torch.bmm(avgpool_out_x, avgpool_out_xi.permute(0, 2, 1))
        avg_attention_xi_x = avg_attention_x_xi.permute(0, 2, 1)

        avg_attention_x_xi = self.softmax(avg_attention_x_xi)
        avg_attention_xi_x = self.softmax(avg_attention_xi_x)

        xi_avg = torch.bmm(avg_attention_x_xi, xi.view(m_batchsize, C, height*width)).view(m_batchsize, C, height,width)
        x_avg = torch.bmm(avg_attention_xi_x, x.view(m_batchsize, C, height*width)).view(m_batchsize, C, height,width)

        return x_max, xi_max,x_avg,xi_avg



