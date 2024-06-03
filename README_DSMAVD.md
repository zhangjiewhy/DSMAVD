#Installation
```shell
conda create -n DSMAVD python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y  
conda activate DSMAVD
pip install openmim                   
mim install "mmengine>=0.6.0"        
mim install "mmcv>=2.0.0rc4,<2.1.0"   
mim install "mmdet>=3.0.0rc6,<3.1.0"  

cd DSMAVD
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```
#Modify files
The load_checkpoint() is located in  ../anaconda3/envs/DSMAVD/lib/python3.8/site packages/mmengine/runner/runner. py,
The modified load_checkpoint() is as follows:
```shell
def load_checkpoint(self,filename: str,map_location: Union[str, Callable] = 'cpu',strict: bool = False,revise_keys: list = [(r'^module.', '')]):
    if isinstance(filename, str):
        checkpoint = _load_checkpoint(filename, map_location=map_location)
    if isinstance(filename, list):
        if  hasattr(self.model, 'module') :       # mlti_gpu
            backbone=self.model.module.backbone
        else:                                        ## single gpu
            backbone = self.model.backbone
        if ((type(backbone).__name__ == 'YOLOv8_backbone_dcn') or (
                type(backbone).__name__ == 'YOLOv8_dcn_channel_att') or (
                type(backbone).__name__ == 'v8_TS_backbone') or (
                type(backbone).__name__ == 'YOLOv8_COM_ATT') or
                type(backbone).__name__ == 'YOLOV8_backbone_CMFIM'):
            checkpoint = _load_checkpoint(filename[0], map_location=map_location)
            checkpoint_i = _load_checkpoint(filename[1], map_location=map_location)
            for key, value in checkpoint_i['state_dict'].items():
                if key.startswith('backbone.stem'):
                    key = key.replace('backbone.stem', 'backbone.stem_i')
                    checkpoint['state_dict'].update({key: value})
                if key.startswith('backbone.stage1'):
                    key = key.replace('backbone.stage1', 'backbone.stage1_i')
                    checkpoint['state_dict'].update({key: value})
                if key.startswith('backbone.stage2'):
                    key = key.replace('backbone.stage2', 'backbone.stage2_i')
                    checkpoint['state_dict'].update({key: value})
                if key.startswith('backbone.stage3'):
                    key = key.replace('backbone.stage3', 'backbone.stage3_i')
                    checkpoint['state_dict'].update({key: value})
                if key.startswith('backbone.stage4'):
                    key = key.replace('backbone.stage4', 'backbone.stage4_i')
                    checkpoint['state_dict'].update({key: value})
                if key.startswith('neck'):
                    key = key.replace('neck', 'neck_i')
                    checkpoint['state_dict'].update({key: value})
        else:
            checkpoint = _load_checkpoint(filename[0], map_location=map_location)
            checkpoint_i = _load_checkpoint(filename[1], map_location=map_location)
            for key, value in checkpoint_i['state_dict'].items():
                if key.startswith('backbone'):
                    key=key.replace('backbone','backbone_i')
                    checkpoint['state_dict'].update({key: value})
                if key.startswith('neck'):
                    key =key.replace('neck', 'neck_i')
                    checkpoint['state_dict'].update({key:value})
    # Add comments to describe the usage of `after_load_ckpt`
    self.call_hook('after_load_checkpoint', checkpoint=checkpoint)
    if is_model_wrapper(self.model):
        model = self.model.module
    else:
        model = self.model
    checkpoint = _load_checkpoint_to_model(
        model, checkpoint, strict, revise_keys=revise_keys)
    self._has_loaded = True
    self.logger.info(f'Load checkpoint from {filename}')
    return checkpoint
```
If load_checkpoint() is modified, the configurations in yolov8_s-FLIR_RGBT.py and yolov8_s_CMFIM-FLIR.py are as follows:
```shell
load_from = ['checkpoint/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth',
            'checkpoint/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth']
```

If load_checkpoint() is not modified, the configurations in yolov8_s-FLIR_RGBT.py and yolov8_s_CMFIM-FLIR.py are as follows:
```shell
load_from = 'checkpoint/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
```

#train and test
   Please see the usage of MMYOLO
