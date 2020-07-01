#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

lmd=0.7 # Overridden when tune=True
#datapath=../few-shot/data/images  # for mini
datapath=../few-shot/data/tiered-imagenet/data  # for tiered
#datapath=../few-shot/data/CUB/CUB_200_2011/images # for CUB
testit=600
## ## Resnet 18
config=./configs/tiered/softmax/resnet18.config # tiered
#config=./configs/mini/softmax/resnet18.config # mini
#config=./configs/cub/softmax/resnet18.config # cub

#### WRN
#config=./configs/tiered/softmax/wideres.config # tiered
#config=./configs/mini/softmax/wideres.config # mini
#config=./configs/cub/softmax/wideres.config # cub
#
bound=False
tune=False
k=3
mode=kmeans
log=kmeans.log
##
python ./src/train_SLK.py -c $config  --lmd $lmd --tune-lmd $tune  --data $datapath --slk --mode $mode --lap-bound $bound  --log-file $log --meta-test-iter $testit --evaluate
##
mode=MS
log=MS.log
python ./src/train_SLK.py -c $config  --lmd $lmd --tune-lmd $tune  --data $datapath --slk --mode $mode --lap-bound $bound  --log-file $log --meta-test-iter $testit --evaluate

mode=ncut
log=Ncut.log
python ./src/train_lshot_SLK.py -c $config --knn $k  --lmd $lmd --tune-lmd $tune  --data $datapath --slk --mode $mode --lap-bound $bound  --log-file $log --meta-test-iter $testit --evaluate

bound=True
tune=True
#
mode=kmeans
log=SLK-kmeans.log
python ./src/train_SLK.py -c $config --knn $k  --lmd $lmd --tune-lmd $tune  --data $datapath --slk --mode $mode --lap-bound $bound  --log-file $log --meta-test-iter $testit --evaluate
#
mode=MS
log=SLK-MS.log
python ./src/train_SLK.py -c $config --knn $k  --lmd $lmd --tune-lmd $tune  --data $datapath --slk --mode $mode --lap-bound $bound  --log-file $log --meta-test-iter $testit --evaluate