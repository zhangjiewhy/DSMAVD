from mmengine.model import BaseModule
import torch
from mmyolo.models.plugins.cbam import CBAM, ChannelAttention
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

class Incep_fus_moudle(BaseModule):
    def __init__(self,inchannel: int):
        super().__init__()
        self.norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        self.act_cfg = dict(type='SiLU', inplace=True)
        outchannel=int(inchannel/2)
        self.fusion_1x1 = ConvModule(inchannel,outchannel,kernel_size=1,stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg,bias=True)

        # self.fusion_3x3_1 = ConvModule(inchannel,outchannel,kernel_size=1, stride=1,padding=0,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.fusion_3x3 = ConvModule(inchannel,outchannel,kernel_size=3,stride=1,padding=1,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)

        # self.fusion_5x5_1 = ConvModule(inchannel, outchannel,kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        # self.fusion_5x5_31 = ConvModule(inchannel, outchannel,kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        # self.fusion_5x5_32 = ConvModule(inchannel, outchannel, kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)

        self.fusion_5x5 = ConvModule(inchannel, outchannel, kernel_size=5, stride=1, padding=2, norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.channel_att = ChannelAttention(channels=3*outchannel)  #5*outchanne    3*outchanne

    def forward(self, x: torch.Tensor,x_i:torch.Tensor) -> torch.Tensor:

        x_xi=torch.cat((x, x_i), dim=1)
        fu_1x1 = self.fusion_1x1(x_xi)
        fu_3x3 = self.fusion_3x3(x_xi)
        fu_5x5 = self.fusion_5x5(x_xi)

        x_cat = torch.cat((fu_1x1,fu_3x3,fu_5x5), dim=1)
        # x_cat = torch.cat((x,x_i,fu_1x1, fu_3x3, fu_5x5), dim=1)

        x_chan_att=self.channel_att(x_cat)

        x_chan_att_new=x_chan_att.reshape([x_chan_att.shape[0],-1,int(x_chan_att.shape[1]/3),1,1])
        # # x_chan_att_new = x_chan_att.reshape([x_chan_att.shape[0], -1, int(x_chan_att.shape[1] / 5), 1, 1])
        x_chan_att=torch.softmax(x_chan_att_new,dim=1).reshape(x_chan_att.shape)

        x_1x1, x_3x3, x_5x5 = torch.tensor_split(x_chan_att * x_cat, 3, dim=1)
        x =x_1x1 + x_3x3 + x_5x5
        # x,x_i,x_1x1,x_3x3,x_5x5 =torch.tensor_split(x_chan_att*x_cat,5,dim=1)
        # x = x + x_i + x_1x1 + x_3x3 + x_5x5

        return x