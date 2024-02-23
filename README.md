# DSCA-Net
This repository provides the code for the methods and experiments presented in our paper '**Dual-stream Class-adaptive Network for
Semi-supervised Hyperspectral Image Classification**'. 

**If you have any questions, you can send me an email. My mail address is fangyx@hnu.edu.cn.**



## Directory structure

```
path to dataset:
                ├─Data
                  ├─PaviaU
                  	├─PaviaU.mat
                  	├─PaviaU_gt.mat
                  	├─PaviaU_10_label_train_1.mat
                  	├─PaviaU_10_unlabeled_train_1.mat
                  	├─PaviaU_10_label_test_1.mat
                    ...

                  ├─Houston 2013
                  	├─Houston.mat
                  	├─Houston_gt.mat
                  	├─Houston_10_label_train_1.mat
                  	├─Houston_10_unlabeled_train_1.mat
                  	├─Houston_10_label_test_1.mat
                    ...
```

## Generate experimental samples

```
SGLP.py
```

## Train

```
main.py
```

## Citation

If you find this paper useful, please cite:

```
Ting Lu, Yuxin Fang, Wei Fu, Kexin Ding and Xudong Kang, "Dual-stream Class-adaptive Network for Semi-supervised Hyperspectral Image Classificatio," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-11, 2024, Art no. 5507511, doi: 10.1109/TGRS.2024.3357455.
```
