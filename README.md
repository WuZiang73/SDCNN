# Dynamic Snake Convolution Neural Network for Enhanced Image Super-Resolution
## Requirements (Pytorch)  
#### Pytorch 1.13.1

#### Python 3.8

#### torchvision

#### openCv for Python

#### HDF5 for Python

#### Numpy, Scipy

#### Pillow, Scikit-image

#### importlib

## Datasets
### Training dataset

#### The training dataset is downloaded at https://data.vision.ee.ethz.ch/cvl/DIV2K/

### Test datasets

#### The test dataset of Set5 is downloaded at ：https://pan.baidu.com/s/1YqoDHEb-03f-AhPIpEHDPQ (secret code：atwu) (baiduyun) or https://drive.google.com/file/d/1hlwSX0KSbj-V841eESlttoe9Ew7r-Iih/view?usp=sharing (google drive)

#### The test dataset of Set14 is downloaded at ：https://pan.baidu.com/s/1GnGD9elL0pxakS6XJmj4tA (secret code：vsks) (baiduyun) or https://drive.google.com/file/d/1us_0sLBFxFZe92wzIN-r79QZ9LINrxPf/view?usp=sharing (google drive)

#### The test dataset of B100 is downloaded at ：https://pan.baidu.com/s/1GV99jmj2wrEEAQFHSi8jWw （secret code：fhs2) (baiduyun) or https://drive.google.com/file/d/1G8FCPxPEVzaBcZ6B-w-7Mk8re2WwUZKl/view?usp=sharing (google drive)

#### The test dataset of Urban100 is downloaded at ：https://pan.baidu.com/s/15k55SkO6H6A7zHofgHk9fw (secret code：2hny) (baiduyun) or https://drive.google.com/file/d/1yArL2Wh79Hy2i7_YZ8y5mcdAkFTK5HOU/view?usp=sharing (google drive)

#### The test dataset of DIV2K is downloaded at ：https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Commands
### preprocessing

### cd dataset

### python div2h5.py

### Training a model for single scale

#### x2
#### python train.py --patch_size 64 --batch_size 64 --max_steps 800000 --decay 400000 --model sdcnn --ckpt_name SDCNN_x2 --ckpt_dir checkpoint/SDCNN_x2 --scale 2 --num_gpu 1
#### x3
#### python train.py --patch_size 64 --batch_size 64 --max_steps 800000 --decay 400000 --model sdcnn --ckpt_name SDCNN_x3 --ckpt_dir checkpoint/SDCNN_x3 --scale 3 --num_gpu 1

#### x4
#### python train.py --patch_size 64 --batch_size 64 --max_steps 800000 --decay 400000 --model sdcnn --ckpt_name SDCNN_x4 --ckpt_dir checkpoint/SDCNN_x4 --scale 4 --num_gpu 1

### Test with your own parameter setting in the sample.py.

#### python sample.py

### 
