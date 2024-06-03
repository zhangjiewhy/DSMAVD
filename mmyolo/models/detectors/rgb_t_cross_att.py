from mmengine.model import BaseModule
import torch
import torch.nn as nn

class vis_cross_V(BaseModule):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self):
        super(vis_cross_V, self).__init__()
        self.chanel_in = [128, 256, 512]
        ratio = [128,256,512]
        self.RGB_query_conv = nn.ModuleList()
        self.RGB_key_conv = nn.ModuleList()
        self.RGB_value_conv = nn.ModuleList()
        for i in range(len(self.chanel_in)):
            self.RGB_query_conv.append(nn.Conv2d(in_channels=self.chanel_in[i], out_channels=self.chanel_in[i] //ratio[i],kernel_size=1))  # T
            self.RGB_key_conv.append(nn.Conv2d(in_channels=self.chanel_in[i], out_channels=self.chanel_in[i] //ratio[i],kernel_size=1))  # V
            self.RGB_value_conv.append(nn.Conv2d(in_channels=self.chanel_in[i], out_channels=self.chanel_in[i], kernel_size=1))  # V
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_rgb, x_i):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        output_rgb = []
        for i in range(len(self.chanel_in)):
            x = x_rgb[i]
            xi = x_i[i]
            m_batchsize, C, height, width = x.size()
            proj_query = self.RGB_query_conv[i](xi).view(m_batchsize, -1, width * height).permute(0, 2,1)  # （b,c,h,w）-->（b,c,w*h）-->(b,w*h,c)
            proj_key = self.RGB_key_conv[i](x).view(m_batchsize, -1, width * height)  # （b,c,h,w）-->（b,c,w*h）
            attention = torch.bmm(proj_query, proj_key)
            attention = self.softmax(attention)
            proj_value = self.RGB_value_conv[i](x).view(m_batchsize, -1, width * height)

            out_v = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out_v = out_v.view(m_batchsize, C, height, width)
            out_v =out_v + x
            output_rgb.append(out_v)
        return output_rgb
class vis_cross_I(BaseModule):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self):
        super(vis_cross_I, self).__init__()
        self.chanel_in = [128, 256, 512]
        ratio = [128,256,512]
        self.T_query_conv = nn.ModuleList()
        self.T_key_conv = nn.ModuleList()
        self.T_value_conv = nn.ModuleList()
        for i in range(len(self.chanel_in)):
            self.T_query_conv.append(nn.Conv2d(in_channels=self.chanel_in[i], out_channels=self.chanel_in[i] // ratio[i], kernel_size=1))  # V
            self.T_key_conv.append(nn.Conv2d(in_channels=self.chanel_in[i], out_channels=self.chanel_in[i] // ratio[i], kernel_size=1))  # T
            self.T_value_conv.append(nn.Conv2d(in_channels=self.chanel_in[i], out_channels=self.chanel_in[i], kernel_size=1))  # T
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_rgb, x_i):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        output_i = []
        for i in range(len(self.chanel_in)):
            xi = x_i[i]
            x = x_rgb[i]
            m_batchsize, C, height, width = xi.size()

            proj_query = self.T_query_conv[i](x).view(m_batchsize, -1, width * height).permute(0, 2,1)  # （b,c,h,w）-->（b,c,w*h）-->(b,w*h,c)
            proj_key = self.T_key_conv[i](xi).view(m_batchsize, -1, width * height)  # （b,c,h,w）-->（b,c,w*h）
            attention = torch.bmm(proj_query, proj_key)
            attention = self.softmax(attention)
            proj_value = self.T_value_conv[i](xi).view(m_batchsize, -1, width * height)

            out_t = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out_t = out_t.view(m_batchsize, C, height, width)
            out_t = out_t + xi
            output_i.append(out_t)
        return output_i