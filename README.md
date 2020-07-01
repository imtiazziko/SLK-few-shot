# SLK-few-shot
Clustering for Few-shot Learning

This repository contains the code for **Clustering for few-shot learning** paper. If you use this code please cite the following paper:
[**Clustering for few-shot learning**]()  
Imtiaz Masud Ziko, Jose Dolz, Eric Granger and Ismail Ben Ayed  

## Introduction
We Adapt clustering algorithm for few-shot learning task. The clustering part works on the initially learned feature extractor on base class data.

## Usage
### 1. Dependencies
- Python 3.6+
- Pytorch 1.0+

### 2. Datasets
#### 2.1 Mini-ImageNet
You can download the dataset from [here](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)

#### 2.2 Tiered-ImageNet
You can download the Tiered-ImageNet from [here](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view).
After downloading and unziping this dataset run the following script to generate split files.
```angular2
python src/utils/tieredImagenet.py --data path-to-tiered --split split/tiered/
```
#### 2.3 CUB
Download and unpack the CUB 200-2011 from [here](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
After downloading and unziping this dataset run the following script to generate split files.
```angular2
python src/utils/cub.py --data path-to-cub --split split/cub/
```

### 3 Train and Test
You can download our pretrained network models on base classes by running:
```angular2
cd ./src
python download_models.py
```
Alternatively to train the network on the base classes from scratch remove the "--evaluate " options in the following script.
The scripts to test SLK-shot:
```angular2
sh run_SLK.sh
```
You can change the commented options accordingly for each dataset.

Some of our results of LaplacianShot with WRN on mini/tiered imageNet/CUB:

| Dataset | Network   | 1-shot | 5-shot |
|---------|-----------|--------|--------|
| Mini    | WRN       | 74.86  | 84.13  |
| Tiered  | WRN       | 80.18  | 87.56  |
| CUB     | Resbet-18 | 80.96  | 88.38  |
