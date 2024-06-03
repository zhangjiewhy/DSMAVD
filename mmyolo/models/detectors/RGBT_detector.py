# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmyolo.registry import MODELS as MODELSS

from typing import List, Tuple, Union
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn import ConvModule
from mmyolo.models.plugins.cbam import CBAM, ChannelAttention
import mmcv

# from inception_fusion import Incep_fus_moudle

@MODELSS.register_module()
class RGBTDetector(SingleStageDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 backbone_i: ConfigType,
                 neck: ConfigType,
                 neck_i: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 fusion_mode=None,
                 fusion_location=None,
                 add_xuishu=None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.fusion_mode=fusion_mode
        self.fusion_location = fusion_location
        self.add_xuishu = add_xuishu
        self.print_predict=1
        if fusion_location=='before_head':
            self.backbone_i = MODELS.build(backbone_i)
            self.neck_i = MODELS.build(neck_i)
        if fusion_location == 'before_neck':
            self.backbone_i = MODELS.build(backbone_i)

        if self.fusion_mode == 'Concat':
            self.conv_cfg = None
            self.norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  #requires_grad=True
            self.act_cfg = dict(type='SiLU', inplace=True)
            # self.act_cfg = dict(type='LeakyReLU', negative_slope=0.1)   #yolov7_tiny
            self.fu_conv=self.fu_convs()

        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        #模型复杂度分析时执行这个_forward(）   python tools/analysis_tools/get_flops.py
        print('#############RGBT_Detector is _forward()######################')
        x= self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if self.print_predict==1:
            print('#############RGBT_Detector is predict()######################')
            self.print_predict = 0
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x= self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)  #v8进入v5_head的loss
        return losses
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if batch_inputs.shape[1] > 3:
            img = batch_inputs[:, 0:3, :, :]
            drawn_img=img
            img_i = batch_inputs[:, 3:6, :, :]
            # self.show_img_imgi(img,img_i)
            if self.fusion_location == 'before_backbone':
                x = self.feature_fusion(img, img_i, self.fusion_mode)  # 在骨干网络前加conv整合将img与img_i整合成3通道
                x = self.backbone(x)
                if self.with_neck:
                    x = self.neck(x)
            elif self.fusion_location == 'before_neck':
                x= self.backbone(img)
                x_i = self.backbone_i(img_i)
                x = self.feature_fusion(x, x_i, self.fusion_mode)
                if self.with_neck:
                    x = self.neck(x)
            elif self.fusion_location == 'before_head':
                x = self.backbone(img)
                x = self.neck(x)
                x_i = self.backbone_i(img_i)
                x_i = self.neck_i(x_i)
                x = self.feature_fusion(x, x_i, self.fusion_mode)  # 'Concat' ,'add'

            elif self.fusion_location == 'RGBT':
                img_i = batch_inputs[:, 3, :, :]
                x=torch.cat((img, img_i.unsqueeze(1)), dim=1)
                x = self.backbone(x)
                if self.with_neck:
                    x = self.neck(x)
            else:
                raise TypeError('fusion_location is', {self.fusion_location}, 'not true')
        else:
            print("################ ERROR: USE RGB ############")
            x = self.backbone(batch_inputs)
            if self.with_neck:
                x = self.neck(x)
        return x

    def feature_fusion(self, x, x_i, model):
        assert len(x) == len(x_i)
        x_fus =[]
        if model == 'add':
            if self.fusion_location == 'before_backbone':
                x_fus = x + x_i
                return x_fus
            else:
                for i in range(len(x)):
                    x_fus.append(x[i] + x_i[i])
        if model == 'Concat':
            if  self.fusion_location == 'before_backbone':
                return self.fu_conv[0](torch.cat((x, x_i), dim=1))
            else:
                for i in range(len(x)):
                    x_fus.append(self.fu_conv[i](torch.cat((x[i], x_i[i]), dim=1)))
        if model == 'RGBT':
            x = torch.cat((x, x_i[:, 0, :, :].unsqueeze(1)), dim=1)
            return x

        return tuple(x_fus)

    def show_img_imgi(self,img,img_i):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure('show rgb and t')     #img、img_i： (N, C, H ,W)
        for i, el in enumerate(img):
            x_rgb = img[i,:,:,:].detach().cpu().numpy()
            x_rgb[0,:,:] = x_rgb[0,:,:] *255
            x_rgb[1, :, :] = x_rgb[1, :, :] * 255 
            x_rgb[2, :, :] = x_rgb[2, :, :] * 255
            x_rgb=x_rgb.astype(np.uint8).transpose((1, 2, 0))#CHW转HWC
            # x_rgb=x_rgb[:, :, :: -1]#bgr转rgb
    
            x_t = img_i[i,:,:,:].detach().cpu().numpy()
            x_t[0,:,:] = x_t[0,:,:] * 255
            x_t[1, :, :] = x_t[1, :, :] *255
            x_t[2, :, :] = x_t[2, :, :] * 255 
            x_t=x_t.astype(np.uint8).transpose((1, 2, 0))#CHW转HWC
            # x_t = x_t[:, :, :: -1] #bgr转rgb
    
            plt.subplot(1, 2, 1)
            plt.imshow(x_rgb)
            plt.subplot(1, 2, 2)
            plt.imshow(x_t)
            plt.show()

    def fu_convs(self):
        fu_conv=nn.ModuleList()
        if self.fusion_location == 'before_backbone':
            # fu_conv.append(build_conv_layer(self.conv_cfg, 6, 3, kernel_size=3, stride=1, padding=1, bias=False)),
            # fu_conv.append(ConvModule(6, 3, kernel_size=1, stride=1, padding=0, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)),  #1X1
            # fu_conv.append(ConvModule(6, 3, kernel_size=3, stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)),  #3X3
            fu_conv.append(ConvModule(6, 3, kernel_size=5, stride=1, padding=2, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)),  # 5X5

        if self.fusion_location == 'before_neck':
            #yolov5_s  inchannel = [256,512, 1024]   outchannel = [128,256, 512]
            inchannel = [256, 512, 1024]
            outchannel = [128, 256, 512]
            for i in range(len(inchannel)):
                fu_conv.append(ConvModule(
                        inchannel[i],
                        outchannel[i],
                        kernel_size =1,  #5   3   ,1
                        stride=1,
                        padding=0,     #2    1,     0
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
        if self.fusion_location == 'before_head':
            # yolov5_n  inchannel = [128,256,512]   outchannel = [64,128,256]
            # yolov5_s  inchannel = [256,512, 1024]   outchannel = [128,256, 512]
            inchannel = [256, 512, 1024]
            outchannel = [128, 256, 512]
            # inchannel = [128,256,512]
            # outchannel = [64,128,256]
            for i in range(len(inchannel)):
                fu_conv.append(ConvModule(
                inchannel[i],
                outchannel[i],
                kernel_size=3,  #5   3   ,1
                stride=1,
                padding=0,   #2    1,     0
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return fu_conv

    def chan_attion(self):
        chan_attions = nn.ModuleList()
        inchannel = [256, 512, 1024]
        for i in range(len(inchannel)):
            chan_attions.append(ChannelAttention(channels=inchannel[i]))
        return chan_attions

    def cbams(self):
        cbamm=nn.ModuleList()
        inchannel = [256, 512, 1024]
        for i in range(len(inchannel)):
            cbamm.append(CBAM(in_channels=inchannel[i]))
        return cbamm

    def save_featmap(self,flatten_featmaps,drawn_img,channel_reduction,out_file=None,out_file_img=None):
        from mmyolo.utils.misc import auto_arrange_images
        from mmyolo.registry import VISUALIZERS
        import numpy as np
        vis_backends = [dict(type='LocalVisBackend')]
        visualizer = dict(
            type='mmdet.DetLocalVisualizer',
            vis_backends=vis_backends,
            name='visualizer')
        visualizer=VISUALIZERS.build(visualizer)

        drawn_img = drawn_img.detach().cpu().numpy()
        # drawn_img = drawn_img[:, :, ::-1]
        drawn_img[0, :, :] = drawn_img[0, :, :] * 255
        drawn_img[1, :, :] = drawn_img[1, :, :] * 255
        drawn_img[2, :, :] = drawn_img[2, :, :] * 255
        drawn_img = drawn_img.astype(np.uint8).transpose((1, 2, 0))  # CHW转HWC
        shown_imgs = []
        for featmap in flatten_featmaps:
            shown_img = visualizer.draw_featmap(
                featmap[0],
                drawn_img,
                channel_reduction=channel_reduction,
                # channel_reduction=None,
                topk=2,
                arrangement=[1,2])
            shown_imgs.append(shown_img)

        shown_imgs = auto_arrange_images(shown_imgs)

        if out_file:
            aa=shown_imgs[..., ::-1]
            mmcv.imwrite(shown_imgs[..., ::-1], out_file)
        if out_file_img:
            drawn_img = drawn_img[:, :, :: -1]
            mmcv.imwrite(drawn_img, out_file_img)
    # if args.show:
    #     visualizer.show(shown_imgs)