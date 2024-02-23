import copy
import itertools
import numpy as np
import torch
from scipy import io
import HyperX
import metric

def split_dataset(gt_2D, numTrain=0.01):
    """
    功能：按照比例分割训练集与测试集
    输入：（二维原始标签y，每类训练数量(int)或训练集比例(float)）
    输出：训练集一维坐标,测试集一维坐标
    备注：当某类别数量过少时,就训练集测试集复用
    """
    gt_1D = np.reshape(gt_2D, (-1, 1))
    train_gt_1D = np.zeros_like(gt_1D)
    test_gt_1D = np.zeros_like(gt_1D)
    train_idx, test_idx, numList = [], [], []
    numClass = np.max(gt_1D)  # 获取最大类别数
    for i in range(1, numClass + 1):  # 忽略背景元素
        idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
        numList.append(len(idx))  # 得到该类别的数量
        np.random.shuffle(idx)  # 对坐标乱序
        Train = numTrain if numTrain > 1 else int(len(idx) * numTrain)
        train_idx.append(idx[:Train])  # 收集每一类的训练坐标
        if len(idx) > Train * 2:
            test_idx.append(idx[Train:])  # 收集每一类的测试坐标
        else:  # 如果该类别数目过少，则训练集验证集重合使用(考虑到indianPines)
            test_idx.append(idx[-Train:])
    for i in range(len(train_idx)):
        for j in range(len(train_idx[i])):
            train_gt_1D[train_idx[i][j]] = i + 1
        for k in range(len(test_idx[i])):
            test_gt_1D[test_idx[i][k]] = i + 1
    train_gt_2D = np.reshape(train_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    test_gt_2D = np.reshape(test_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    return train_gt_2D, test_gt_2D


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
    res = num/denom
    res[np.isneginf(res)] = 0
    return res


def split_dataset_equal(gt_2D, numTrain=20):
    """
    功能：按照每类数量分割训练集与测试集
    输入：（二维原始标签y，每类训练数量）
    输出：训练集一维坐标,测试集一维坐标
    """
    gt_1D = np.reshape(gt_2D, (-1, 1))
    train_gt_1D = np.zeros_like(gt_1D)
    test_gt_1D = np.zeros_like(gt_1D)
    train_idx, test_idx, numList = [], [], []
    numClass = np.max(gt_1D)  # 获取最大类别数
    for i in range(1, numClass + 1):  # 忽略背景元素
        idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
        numList.append(len(idx))  # 得到该类别的数量
        np.random.shuffle(idx)  # 对坐标乱序
        if len(idx) < numTrain:
            train_idx.append(idx[:(numTrain // 2)])  # 样本不足，收集每一类的前n/2个作为训练样本
            test_idx.append(idx[numTrain // 2:])  # 收集每一类剩余的作为测试样本
        else:
            train_idx.append(idx[:numTrain])  # 收集每一类的前n个作为训练样本
            test_idx.append(idx[numTrain:])  # 收集每一类剩余的作为测试样本
    for i in range(len(train_idx)):
        for j in range(len(train_idx[i])):
            train_gt_1D[train_idx[i][j]] = i + 1
        for k in range(len(test_idx[i])):
            test_gt_1D[test_idx[i][k]] = i + 1
    train_gt_2D = np.reshape(train_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    test_gt_2D = np.reshape(test_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    return train_gt_2D, test_gt_2D


def split_dataset_SGLP(data_path, gt_path, train_2d, test_2d, unlabeled_num, num_superpixel=500):
    """
    """
    data, sp_graph, sp_labels, gt_2D = HyperX.superpixel_dataset(data_path, gt_path, num_superpixel=num_superpixel)

    sp_patch = np.zeros(len(sp_graph[0]), dtype=np.int64)
    numClass = np.max(gt_2D)  # 获取最大类别数
    train_list = []
    new_sp_labels = copy.deepcopy(sp_labels)
    # get labeled pixel average
    for i in range(1, numClass + 1):
        corrdinate = np.argwhere(train_2d == i)
        for idx, cor in enumerate(corrdinate):
            train_list.append(data[cor[0], cor[1], :])
            index = sp_labels[cor[0], cor[1]]
            cor0 = np.argwhere(sp_labels == index)
            sp_patch[(index-1)] = i
            for _, cor1 in enumerate(cor0):
                new_sp_labels[cor1[0], cor1[1]] = i
    counts = np.bincount(sp_patch)
    train_array = np.array(train_list)
    train_array = np.array_split(train_array, numClass, axis=0)
    train_avg = {}

    for i in range(0, numClass):
        train_avg[i] = np.mean(train_array[i], axis=0)

    # labeled superpixel patch
    unlabeled_sp_patch_idx = np.where(sp_patch == 0)[0]
    unlabeled_list = []
    num_list = []
    current, num = 0, 0
    for i in zip(unlabeled_sp_patch_idx):
        cor2 = np.argwhere((sp_labels-1) == i)
        for num, cor2 in enumerate(cor2):
            unlabeled_list.append(data[cor2[0], cor2[1], :])
        num_list.append(num+1)
    unlabeled_avg = {}
    for i in range(len(num_list)):
        unlabeled_array = np.array(unlabeled_list[current:current+num_list[i]])
        current += num_list[i]
        unlabeled_avg[i] = np.mean(unlabeled_array, axis=0)
    train_avg = np.array(list(train_avg.values()))
    unlabeled_avg = np.array(list(unlabeled_avg.values()))
    # cos_similar
    martix = get_cos_similar_matrix(train_avg, unlabeled_avg)

    label_cos = np.argmax(martix, axis=0) + 1

    value_array = np.zeros_like(martix)
    for i in range(numClass):
        for idx, label in enumerate(label_cos):
            if label-1 == i:
                value_array[i][idx] = martix[i][idx]
    sp_patch_cos = copy.deepcopy(sp_patch)

    for i, index in enumerate(unlabeled_sp_patch_idx):
        sp_patch_cos[index] = label_cos[i]

    counts_cos = np.bincount(sp_patch_cos)

    sp_labels_cos = copy.deepcopy(new_sp_labels)

    for i, index in enumerate(unlabeled_sp_patch_idx):
        cor3 = np.argwhere(sp_labels == (index+1))
        for idx, cor in enumerate(cor3):
            sp_labels_cos[cor[0], cor[1]] = sp_patch_cos[index]

    # calculate OA
    test_1d = np.reshape(test_2d, (-1, 1))
    plabels_cos_1d = np.reshape(sp_labels_cos, (-1, 1))

    prey_cos, test_label = [], []
    for i in range(1, numClass + 1):
        idx1 = np.where(test_1d == i)[0]  # 记录下该类别的坐标值
        for index in zip(idx1):
            test_label.append(test_1d[index][0])
            prey_cos.append(plabels_cos_1d[index][0])
    # split unlabeled dataset
    train_1d = np.reshape(train_2d, (-1, 1))
    unlabeled_train_gt_1D = np.zeros_like(train_1d)
    unlabeled_train_idx, numList1, numList2, numList3 = [], [], [], []
    train_idx, prey_idx = [], []
    numClass = np.max(train_1d)  # 获取最大类别数
    for i in range(1, numClass + 1):
        idx1 = np.where(train_1d == i)[0]
        train_idx.append(idx1)
        numList1.append(len(idx1))
        # np.random.shuffle(idx1)

        idx2 = np.where(plabels_cos_1d == i)[0]  # 记录下该类别的坐标值
        for index in zip(idx2):
            if (np.array(train_idx) != index).all():
                prey_idx.append(index[0])
        numList2.append(len(prey_idx))
    # cut the same label out
        idx3 = np.array(list(set(idx2).difference(set(idx1))))
        numList3.append(len(idx3))
    num_array = np.array(numList2)
    prey_num_array = copy.deepcopy(num_array)
    for i in range(1, len(num_array)):
        prey_num_array[i] = num_array[i] - num_array[i-1]
    prey_array = {}
    cur = 0
    for i in range(len(prey_num_array)):
        prey_array[i] = prey_idx[cur:cur + prey_num_array[i]]
        cur += prey_num_array[i]
        np.random.shuffle(prey_array[i])
    # get the correct unlabeled train set
    for i in range(0, numClass):
        numList4 = prey_array[i]
        if len(numList4) < unlabeled_num:
            unlabeled_train_idx.append(numList4[:])  # 样本不足，收集每一类的样本数作为训练样本
        else:
            unlabeled_train_idx.append(numList4[:unlabeled_num])  # 收集每一类的前n个作为训练样本
    for i in range(len(unlabeled_train_idx)):
        for j in range(len(unlabeled_train_idx[i])):
            unlabeled_train_gt_1D[unlabeled_train_idx[i][j]] = i + 1
    unlabeled_train_gt_2D = np.reshape(unlabeled_train_gt_1D, (gt_2D.shape[0], gt_2D.shape[1]))
    # results = all_np(unlabeled_train_gt_2D)  # check the number of unlabeled samples correct or not
    return unlabeled_train_gt_2D


def all_np(arr):
    List = list(itertools.chain.from_iterable(arr))
    arr = np.array(List)
    key = np.unique(arr)
    results = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        results[k] = v
    return results


# DataPath = '/home/server03/桌面/Yuxin/Data/IndianPines10/IndianPines_corrected.mat'
# GTPath = '/home/server03/桌面/Yuxin/Data/IndianPines10/Indian_pines_gt10.mat'

DataPath = '/home/server03/桌面/Yuxin/Data/PaviaU/PaviaU.mat'
GTPath = '/home/server03/桌面/Yuxin/Data/PaviaU/PaviaU_gt.mat'

# DataPath = '/home/server03/桌面/Yuxin/Data/KSC/KSC.mat'
# GTPath = '/home/server03/桌面/Yuxin/Data/KSC/KSC_gt.mat'


# DataPath = '/home/server03/桌面/Yuxin/Data/Houston/Houston.mat'
# GTPath = '/home/server03/桌面/Yuxin/Data/Houston/Houston_gt.mat'


GT = io.loadmat(GTPath)
# GT=GT['XiongAn_gt']
# GT = GT['indian_pines_gt10']
# GT = GT['Houston_gt']
# GT = GT['salinas_gt']
GT = GT['paviaU_gt']
# GT = GT['Botswana_gt']
# GT = GT['KSC_gt']
# GT = GT['WHU_Hi_LongKou_gt']
# GT = GT['pavia_gt']  # paviaC
##test val train
# train_gt_2D, gt_2D = split_dataset_equal(GT, numTrain=50)
# val_gt_2D, test_gt_2D = split_dataset_equal(gt_2D, numTrain=2000)
# train_gt_2D, gt_2D = split_dataset(GT, numTrain=0.1)
# val_gt_2D, test_gt_2D = split_dataset(gt_2D, numTrain=0.3)00.

# io.savemat('./Data/YCD/YCD16_eachClass_10%_train.mat', {'data': train_gt_2D})  #保存mat文件
# io.savemat('./Data/YCD/YCD16_eachClass_10%_val.mat', {'data': val_gt_2D}
# io.savemat('./Data/YCD/YCD16_eachClass_10%_test.mat', {'data': test_gt_2D})
##train test
# labeled_train_gt_2D, unlabeled_train_gt_2D, test_gt_2D = split_dataset_equal(GT, numTrain=10, unlabeled_num=6000)
# train_gt_2D, test_gt_2D = split_dataset(GT, numTrain=0.1)
labeled_train_gt_2D, test_gt_2D = split_dataset_equal(GT, numTrain=10)
unlabeled_train_gt_2D = split_dataset_SGLP(DataPath, GTPath, labeled_train_gt_2D, test_gt_2D, unlabeled_num=500, num_superpixel=6287)
io.savemat('/home/server03/桌面/Yuxin/Data/KSC/10label/KSC_10label_train9.mat', {'data': labeled_train_gt_2D})  #保存mat文件
io.savemat('/home/server03/桌面/Yuxin/Data/KSC/10label/KSC_10label_unlabeled_train9.mat', {'data': unlabeled_train_gt_2D})  #保存mat文件
io.savemat('/home/server03/桌面/Yuxin/Data/KSC/10label/KSC_10label_test9.mat', {'data': test_gt_2D})
##TSNE
# train_gt_2D, test_gt_2D = split_dataset_equal(GT, numTrain=500)
#
# io.savemat('./KSC_eachClass_500_tsne.mat', {'data': train_gt_2D})  #保存mat文件
