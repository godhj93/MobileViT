# MobileViT Implementation in Tensorflow2

## Introduction
This repository contains tensorflow2 implementation of [MobileViT](https://arxiv.org/abs/2110.02178).

You can train a network for cifar dataset.


![arch]('./../img/arch.png)
![table]('./../img/table1.png)

## Dependencies
```
pip install -r requirements.txt
```
## Usage
```
python train.py --ep 50 --bs 16 --data cifar10 --arch [S, XS, XXS] --size [64, 128, 256, 512, ...]
```

## To do
 - [x]  AdamW Optimizer
 - [x]  Learning rate scheduler(including cosine annealing)
 - [x]  Label Smoothing(0.1)  
 - [ ]  Multi-Scaler Training(Adaptive Batch size)
 - [x]  L2 Weight Decay  
 - [ ]  Change Data augmentation from complex to simple one 
 


