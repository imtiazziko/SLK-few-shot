# SLK-few-shot
Clustering for Transductive Few-shot Learning

This repository contains the code for our paper entitled:

[**Transductive Few-Shot Learning: Clustering is All You Need?**](https://arxiv.org/pdf/2106.09516.pdf)  
Imtiaz Masud Ziko, Malik Boudiaf, Jose Dolz, Eric Granger and Ismail Ben Ayed, ArXiv 2021  

## Introduction
We adapt several clustering methods to transductive inference in few-shot learning tasks. The clustering part works on a feature extractor 
initially trained over the base-class data. Using standard training on the base classes, without resorting to complex meta-learning and episodic-training 
strategies, our regularized-clustering approaches outperform state-of-the-art few-shot methods by significant margins, across various models, settings 
and data sets. Surprisingly, we found that even standard clustering procedures (e.g., K-means), which correspond to particular, non-regularized cases of 
our general model, already achieve competitive performances in comparison to the state-of-the-art in transductive few-shot learning. These surprising 
results point to the limitations of the current few-shot benchmarks, and question the viability of a large body of convoluted few-shot learning techniques 
in the recent literature. 


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
## Results
We get the following results in different few-shot benchmarks:

### On **mini-ImageNet**
 With _WRN_ network:

| Methods  | 1-shot | 5-shot |
|--------- |--------|--------|
| ProtoNet (Snell et al., NeurIPS 2017) | 62.60   | 79.97  |
| CC+rot (Gidaris et al., ICCV 2019)  | 62.93  | 79.87  |
| MatchingNet (Vinyals et al., NeurIPS 2016)     | 64.03  | 76.32  |
| FEAT (Ye et al., CVPR 2020)     | 65.10  | 81.11  |
| Transductive fine-tuning (Dhillon et al., ICLR 2020)     | 65.73 | 78.40 |
| SimpleShot (Wang et al., ArXiv 2019)     | 65.87 | 82.09 |
| SIB (Hu et al., ICLR 2020)     | 70.0 | 79.2 |
| BD-CSPN (Liu et al., ECCV 2020)     | 70.31 | 81.89 |
| LaplacianShot (Ziko et al., ICML 2020)     | 73.44 | 83.93|
| K-means      | 73.80 | **84.62**|
| K-modes    | 74.78 | 84.45|
| SLK-Means     | 74.75 | 84.61|
| SLK-MS      | **75.17** | 84.28|

### On **tiered-ImageNet**

With _WRN_ network:

| Methods  | 1-shot | 5-shot |
|--------- |--------|--------|
| CC+rot (Gidaris et al., ICCV 2019)  | 70.53  | 84.98  |
| FEAT (Ye et al., CVPR 2020)     | 70.41  | 84.38  |
| Transductive fine-tuning (Dhillon et al., ICLR 2020)     | 73.34 | 85.50 |
| SimpleShot (Wang et al., ArXiv 2019)     | 70.90 | 85.76 |
| BD-CSPN (Liu et al., ECCV 2020)     | 78.74 | 86.92 |
| LaplacianShot (Ziko et al., ICML 2020)     | 78.80 | **87.72** |
| K-means      | 79.78 | 87.23|
| K-modes    | 80.67 | 87.23|
| SLK-Means     | 80.55 | 87.57|
| SLK-MS      | **81.13** | 87.69|

### On **CUB**

With _ResNet-18_ network

| Methods  | 1-shot | 5-shot |
|--------- |--------|--------|
| MatchingNet (Vinyals et al., NeurIPS 2016)     | 73.49  | 84.45  |
| MAML (Finn et al., ICML 2017)     | 68.42 | 83.47 |
| ProtoNet (Snell et al., NeurIPS 2017)     | 72.99 | 86.64 |
| RelationNet (Sung et al., CVPR 2018)     | 68.58 | 84.05 |
| Chen (Chen et al., ICLR 2019)    | 67.02 | 83.58  |
| SimpleShot (Wang et al., ArXiv 2019)    | 70.28  | 86.37  |
| LaplacianShot (Ziko et al., ICML 2020)     | 79.93 | 88.59 |
| K-means      | 80.30 | 88.51|
| K-modes    | 81.73 | 88.58|
| SLK-Means     | 81.40 | **88.61**|
| SLK-MS      | **81.88** | 88.55|
