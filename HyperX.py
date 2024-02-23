import math

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from Preprocessing import Processor
from utils import HSI_to_superpixels, create_association_mat, create_spixel_graph


class dataLoad_x(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad_x, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2, y1:y2]
        data1 = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            # data = self.radiation_noise(data)
            data1 = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        data1 = np.asarray(np.copy(data1).transpose((2, 0, 1)), dtype='float32')
        # data1 = data[:, self.patch_size // 2, self.patch_size // 2]  #
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data1 = torch.from_numpy(data1)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label


class dataLoad_u(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad_u, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        self.ignored_labels = []
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        self.transform_s1 = transforms.Compose([
                         transforms.ToPILImage(),
                         # transforms.RandomVerticalFlip(),
                         # transforms.RandomGrayscale(p=0.2),
                         # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
                         transforms.ToTensor(),
                         # transforms.Normalize(mean=[0.5], std=[0.225])
                         ])

        self.transform_s2 = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.RandomHorizontalFlip(),
                         # transforms.RandomGrayscale(p=0.3),
                         # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                         # transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.6),
                         ])
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.7, 1.3), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.2, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
                # data2[idx] = data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2, y1:y2]
        data1 = self.data[x1:x2, y1:y2]
        data2 = self.data[x1:x2, y1:y2]
        [_, _, l] = data1.shape
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data1 = self.radiation_noise(data)
        if self.mixture_augmentation:
            data2 = self.mixture_noise(data1, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        # data = data.astype(np.uint8)
        # for k in range(l):
        #     data2[:, :, k] = self.transform_s1(data[:, :, k])
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        data1 = np.asarray(np.copy(data1).transpose((2, 0, 1)), dtype='float32')
        # data2 = np.asarray(np.copy(data2).transpose((2, 0, 1)), dtype='float32')
        # data1 = data[:, self.patch_size // 2, self.patch_size // 2]  #
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data1 = torch.from_numpy(data1)
        label = torch.from_numpy(label)

        label = label - 1
        return data1, data1, label


class dataLoad1(torch.utils.data.Dataset):
    "图像块是奇数时使用"
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad1, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2 + 1, y1:y2 + 1]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label


class dataLoad2(torch.utils.data.Dataset):
    "图像块是偶数时使用"
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad2, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label


class dataLoad3(torch.utils.data.Dataset):
    "3D卷积"
    """ Generic class for a hyperspectral scene"""
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad3, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2 + 1, y1:y2 + 1]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data).unsqueeze(0)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label


class dataLoad4(torch.utils.data.Dataset):
    "图像块是奇数时使用,output: (b, 1, d, w, h)"
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, patch_size=16, center_pixel=True, flip_augmentation=True, radiation_augmentation=True, mixture_augmentation=True):
        """
        Args:
            data: 3D hyperspectral image---->(m, n, c)
            gt: 2D array of labels---->(m,n)
            patch_size: 图像块大小
            center_pixel: bool, 中心像素确定label
            flip_augmentation: bool, 随机左右、上下翻转
            radiation_augmentation:bool, 随机辐射增强
            mixture_augmentation：bool,
        """
        super(dataLoad4, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        mask[gt == 0] = 0
        x_pos, y_pos = np.nonzero(mask)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(arrays):
        horizontal = np.random.random() > 0.5   #范围（0-1）
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = np.fliplr(arrays)  #左右翻转
        if vertical:
            arrays = np.flipud(arrays)  #上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]  #原坐标
        x_center = x + self.patch_size // 2  #边界扩充后的坐标
        y_center = y + self.patch_size // 2
        x1, y1 = x_center - self.patch_size // 2, y_center - self.patch_size // 2  #左上角和右下角的坐标
        x2, y2 = x_center + self.patch_size // 2, y_center + self.patch_size // 2

        data = self.data[x1:x2 + 1, y1:y2 + 1]
        label = self.label[x, y]

        if self.flip_augmentation and self.patch_size > 1:
            data = self.flip(data)
        if self.radiation_augmentation:
            data = self.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data).unsqueeze(0)
        label = torch.from_numpy(label)

        label = label - 1
        return data, label


class dataload_superpixel(torch.utils.data.Dataset):
    def __init__(self, path_to_data, path_to_gt, path_to_sp=None, patch_size=(7, 7), num_superpixel=500, transform=None, pca=True,
                 pca_dim=8, is_superpixel=True, is_labeled=True):
        self.transform = transform
        self.num_superpixel = num_superpixel
        self.path_to_sp = path_to_sp
        p = Processor()
        img, gt = p.prepare_data(path_to_data, path_to_gt)
        self.img = img
        self.gt = gt
        n_row, n_column, n_band = img.shape
        if pca:
            img = scale(img.reshape(n_row * n_column, -1))  # .reshape((n_row, n_column, -1))
            pca = PCA(n_components=pca_dim)
            img = pca.fit_transform(img).reshape((n_row, n_column, -1))
        if is_superpixel:
            if path_to_sp is not None:
                self.sp_labels = loadmat(self.path_to_sp)['labels']
                # show_superpixel(self.sp_labels, img[:, :, :3])
            else:
                self.sp_labels = HSI_to_superpixels(img, num_superpixel=self.num_superpixel, is_pca=False,
                                                    is_show_superpixel=False)
                # show_superpixel(self.sp_labels, img[:, :, :3])
            self.association_mat = create_association_mat(self.sp_labels)
            self.sp_graph, self.sp_centroid = create_spixel_graph(img, self.sp_labels)
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
        y = p.standardize_label(y_)  # [21025]
        for i in np.unique(y):
            print(np.nonzero(y == i)[0].shape[0])

        if not is_labeled:
            self.n_classes = np.unique(y).shape[0] - 1
        else:
            self.n_classes = np.unique(y).shape[0]
        n_samples, n_row, n_col, n_channel = x_patches.shape
        self.data_size = n_samples
        x_patches = scale(x_patches.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))
        x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
        self.x_tensor, self.y_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor), \
                                       torch.from_numpy(y).type(torch.LongTensor)

    def __getitem__(self, idx):
        x, y = self.x_tensor[idx], self.y_tensor[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.data_size


def superpixel_dataset(path_to_data, path_to_gt, num_superpixel=500):
    p = Processor()
    Data, gt_2d = p.prepare_data(path_to_data, path_to_gt)
    [m, n, l] = Data.shape
    # normalization method 1: map to [0, 1]
    num_superpixel = math.floor(m*n // 50)  # can define by urself
    # 数据归一化
    for i in range(l):
        Data[:, :, i] = (Data[:, :, i] - Data[:, :, i].min()) / (Data[:, :, i].max() - Data[:, :, i].min())
    data = Data
    sp_labels = HSI_to_superpixels(data, num_superpixel=num_superpixel, is_pca=True, is_show_superpixel=False)
    # association_mat = create_association_mat(sp_labels)
    sp_graph, sp_centroid = create_spixel_graph(Data, sp_labels)
    return Data, sp_graph, sp_labels, gt_2d
