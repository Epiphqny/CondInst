# CondInst
This repository is an unofficial pytorch implementation of [Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664). The model with ResNet-101 backbone achieves 35.1 mAP on COCO val2017 set.

## Install
The code is based on [detectron2](https://github.com/facebookresearch/detectron2). Please check [Install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for installation instructions.

## Training 
Follows the same way as detectron2.

Single GPU:
```
python train_net.py --config-file configs/CondInst/MS_R_101_3x.yaml
```
Multi GPU(for example 8):
```
python train_net.py --num-gpus 8 --config-file configs/CondInst/MS_R_101_3x.yaml
```
Please adjust the IMS_PER_BATCH in the config file according to the GPU memory.


## Inference
First replace the original detectron2 installed postprocessing.py with the [file](https://github.com/Epiphqny/CondInst/blob/master/postprocessing.py) in this repository, as the original file only suit for ROI obatined masks.
The path should be like /miniconda3/envs/py37/lib/python3.7/site-packages/detectron2/modeling/postprocessing.py

Single GPU:
```
python train_net.py --config-file configs/CondInst/MS_R_101_3x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
Multi GPU(for example 8):
```
python train_net.py --num-gpus 8 --config-file configs/CondInst/MS_R_101_3x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Results
After training 36 epochs on the coco dataset using the resnet-101 backbone, the mAP is 0.351 on COCO val2017 dataset:

<img src="AP.jpg">

## Visualization

<img src="condinst.png" width="2000">

