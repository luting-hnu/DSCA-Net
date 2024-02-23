
import random
import torch
import HyperX
import numpy as np
from scipy import io

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def get_dataloaders(batchsize=30, n=0, dataset='Houston'):
    # setting parameters
    DataPath = '/Data/{}/{}.mat'.format(dataset, dataset)
    TRPath = '/Data/{}/10label/{}_10label_train{}.mat'.format(dataset, dataset, n)
    TRPath_u = '/Data/{}/10label/{}_10label_unlabeled_train{}.mat'.format(dataset, dataset, n)
    TSPath = '/Data/{}/10label/{}_10label_test{}.mat'.format(dataset, dataset, n)
    TrLabel_s = io.loadmat(TRPath)
    TrLabel_u = io.loadmat(TRPath_u)
    TsLabel = io.loadmat(TSPath)
    TrLabel_s = TrLabel_s['data']
    TrLabel_u = TrLabel_u['data']
    TsLabel = TsLabel['data']
    Data = io.loadmat(DataPath)

    if dataset == 'IndianPines10':
       Data = Data['indian_pines_corrected']
    elif dataset == 'PaviaU':
        Data = Data['paviaU']
    elif dataset == 'KSC':
        Data = Data['KSC']
    elif dataset == 'Houston':
        Data = Data['img']  # Houston
    elif dataset == 'IndianPines':
        Data = Data['indian_pines_corrected']
    elif dataset == 'Washington DC':
        Data = Data['washington_dc']
    elif dataset == 'Botswana':
        Data = Data['Botswana']
    Data = Data.astype(np.float32)
    [m, n, l] = Data.shape
    # normalization method 1: map to [0, 1]
    # 数据归一化
    for i in range(l):
        Data[:, :, i] = (Data[:, :, i] - Data[:, :, i].min()) / (Data[:, :, i].max() - Data[:, :, i].min())
    x = Data

    # 数据边界填充，准备分割数据块
    temp = x[:, :, 0]
    patchsize = 24   # input spatial size for model
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')

    for i in range(l):
        temp = x[:, :, i]
        pad_width = np.floor(patchsize / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2
    pad_width_data = x2
    # 构建测试数据集
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, l, patchsize, patchsize), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch = np.reshape(patch, (patchsize * patchsize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, patchsize, patchsize))
        TestPatch[i, :, :, :] = patch
        patchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    # primary test data and transform_w test data
    TestPatch1 = torch.from_numpy(TestPatch)
    TestLabel1 = torch.from_numpy(TestLabel)-1
    TestLabel1 = TestLabel1.long()

    train_dataset_s = HyperX.dataLoad_x(x2, TrLabel_s, patch_size=patchsize, center_pixel=True, flip_augmentation=True)
    train_loader_s = DataLoader(train_dataset_s, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=8)

    train_dataset_u = HyperX.dataLoad_u(x2, TrLabel_u, patch_size=patchsize, center_pixel=True, flip_augmentation=True,
                                        mixture_augmentation=True)
    train_loader_u = DataLoader(train_dataset_u, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=8)

    test_data = TensorDataset(TestPatch1, TestLabel1)
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_loader_s, test_loader, train_loader_u, TestLabel1, TestPatch1, pad_width_data, train_dataset_u


if __name__ == '__main__':

    train_loader_s, test_loader, train_loader_u, TestLabel1, TestPatch1, pad_width_data, train_dataset_u = get_dataloaders()
    print(len(train_loader_s), len(train_loader_u), len(test_loader))



