"""
Created on  16:19:51 2022.10.22
@author: 24
"""
from tqdm import tqdm
import numpy as np
from datetime import datetime
from model import Network, Supervisednetwork
from sklearn import manifold
import itertools
import seaborn as sns
import visdom
import spectral
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries, find_boundaries, quickshift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, normalize
from matplotlib import cm
import matplotlib as mpl
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math



def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


def set_model(args):
    if args.model == 'Supervisednetwork':
        model0 = Supervisednetwork(n_classes=args.num_classes, bands=args.bands, low_dim=args.feature_dim)
        return model0

    if args.model == 'Network':
        model1 = Network(args.bands, args.num_classes, args.feature_dim)
        return model1


def DrawCluster(label, cluster, dataset, oa):
    palette = np.array([])
    label = np.array(label)
    num_class = int(max(label))
    cluster = np.array(cluster)
    if dataset == 'PU':
        palette = np.array([[0, 0, 255],
                            [76, 230, 0],
                            [255, 190, 232],
                            [255, 0, 0],
                            [156, 156, 156],
                            [255, 255, 115],
                            [0, 255, 197],
                            [132, 0, 168],
                            [0, 0, 0]])
        palette = palette * 1.0 / 255
    elif dataset == 'InP':
        palette = np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 170, 0]])
        palette = palette * 1.0 / 255
    elif dataset == 'SA':
        palette = np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 170, 0]])
        palette = palette * 1.0 / 255
    elif dataset == 'PD':
        palette = np.array([[237, 227, 81],
                            [167, 237, 81],
                            [0, 0, 0],
                            [181, 117, 14],
                            [77, 122, 15],
                            [186, 186, 186]])
        palette = palette * 1.0 / 255
    elif dataset == 'Houston':
        palette = np.array([[0, 139, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [255, 127, 80],
                            [255, 0, 255],
                            [139, 139, 0],
                            [0, 139, 139],
                            [0, 255, 0],
                            [0, 255, 255],
                            [0, 30, 190],
                            [127, 255, 0],
                            [218, 112, 214],
                            [46, 139, 87],
                            [0, 0, 139],
                            [255, 165, 0], ])
    palette = palette * 1.0 / 255
    tsne = manifold.TSNE(n_components=2, init='pca')
    X_tsne = tsne.fit_transform(cluster)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne-x_min)/(x_max-x_min)
    plt.figure()

    for i in range(num_class):
        xx1 = X_norm[np.where(label==i), 0]
        yy1 = X_norm[np.where(label==i), 1]

        plt.scatter(xx1, yy1, color=palette[i].reshape(1, -1), s=20, linewidths=2)
    plt.xlim(np.min(X_norm)-0.0001, np.max(X_norm)+0.0001)
    plt.ylim(np.min(X_norm)-0.0001, np.max(X_norm)+0.0001)
    # plt.legend(['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
    #                     'Fallow_smooth',
    #                     'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
    #                     'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
    #                     'Vinyard_untrained', 'Vinyard_vertical_trellis'],
    #                     bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    if dataset == 'PU':
        plt.legend(['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                    'Self-Blocking Bricks', 'Shadows'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    elif dataset == 'InP':
        plt.legend(['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
                    'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    elif dataset == 'SA':
        plt.legend(['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                    'Fallow_smooth',
                    'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                    'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                    'Vinyard_untrained', 'Vinyard_vertical_trellis'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    elif dataset == 'KSC':
        plt.legend(['Scrub', 'Willow swamp', 'Cabbage palm hammock', 'Cabbage palm/oak hammock', 'Slash pine'
                    'Oak/broakleaf hammock', 'Hardwood swamp', 'Graminoid', 'Spaitina marsh', 'Cattail marsh'
                    'Salt marsh', 'Mud flats', 'Water'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    elif dataset == 'Houston':
        plt.legend(['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
                    , 'Soil', 'Water', 'Residential',
                    'Commercial', 'Road', 'Highway', 'Railway',
                    'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig('results/cluster/' + str('.%6f' % oa)+'.svg', dpi=600, bbox_inches='tight')
    plt.show()
    # plt.savefig("./cluster.png")
    # Y = tsne


def result_display(net, img, patchsize, batchsize, Classes):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = patchsize
    center_pixel = True
    batch_size = batchsize
    n_classes = Classes
    pad_width = np.floor(patchsize / 2)
    pad_width = np.int(pad_width)
    kwargs = {'step': 1, 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    time = 0
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.cuda()
            data = data.squeeze(1)
            tim = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output,  _ = net(data)
            # _, _, _, _, output, _, _, _ = net(data, data, data, data)
            output = torch.softmax(output, dim=1)
            end.record()
            torch.cuda.synchronize()
            tim = start.elapsed_time(end)
            time += tim
            if isinstance(output, tuple):
                output = output[1]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2, :] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs, time


def DrawResult(y_pred, dataset, num_class, OA):
    row, col = 0, 0
    palette = np.array([])
    num_class = num_class
    if dataset == 'PaviaU':
        row = 610
        col = 340
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 25]])
        palette = palette * 1.0 / 255
    elif dataset == 'IndianPines10':
        row = 145
        col = 145
        palette = np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],])
        palette = palette * 1.0 / 255
    elif dataset == 'SA':
        row = 512
        col = 217
        palette = np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 170, 0]])
        palette = palette * 1.0 / 255
    elif dataset == 'PD':
        row = 377
        col = 512
        palette = np.array([[237, 227, 81],
                            [167, 237, 81],
                            [0, 0, 0],
                            [181, 117, 14],
                            [77, 122, 15],
                            [186, 186, 186]])
        palette = palette * 1.0 / 255
    elif dataset == 'Houston':
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [128, 255, 0],
                            [49, 137, 87],
                            [1, 139, 0],
                            [160, 82, 46],
                            [1, 255, 255],
                            [255, 255, 255],
                            [215, 191, 215],
                            [254, 0, 0],
                            [138, 0, 2],
                            [0, 0, 0],
                            [255, 255, 1],
                            [239, 154, 1],
                            [85, 26, 142],
                            [255, 127, 80], ])
        palette = palette * 1.0 / 255
    X_result = np.zeros((y_pred.shape[0], y_pred.shape[1], 3))
    for i in range(0, num_class):
        X_result[np.where(y_pred == i)[0], np.where(y_pred == i)[1], 0] = palette[i, 0]
        X_result[np.where(y_pred == i)[0], np.where(y_pred == i)[1], 1] = palette[i, 1]
        X_result[np.where(y_pred == i)[0], np.where(y_pred == i)[1], 2] = palette[i, 2]

    X_result = np.reshape(X_result, (row, col, 3))

    # X_mask[1:-1,1:-1,:] = X_result
    plt.axis("off")
    plt.imsave('results/cls_map/' + dataset + '_default_net' + str(OA) + '.svg', X_result)
    return X_result


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    if w % 2 == 0:
        end_H = H - h + offset_h
        end_W = W - w + offset_w
    else:
        end_H = H - h + offset_h + 1
        end_W = W - w + offset_w + 1
    for x in range(0, end_W, step):
        if x + w > W:
            x = W - w
        for y in range(0, end_H, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_to_color(x,palette):
    return convert_to_color_(x, palette=palette)


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)), np.transpose(gt, (2, 0, 1))], nrow=2, opts={'caption': caption})


def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})


def visualization(prediction, img, gt, dataset, TrLabel, TsLabel):
    prediction = prediction + 1
    train_gt = TrLabel
    test_gt = TsLabel
    RGB_BANDS = (0, 0, 0)
    LABEL_VALUES = []
    if dataset == 'PU':
        RGB_BANDS = (55, 41, 12)
        LABEL_VALUES = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    if dataset == 'IP':
        RGB_BANDS = (43, 21, 11)  # AVIRIS sensor
        LABEL_VALUES = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
    if dataset == 'Houston':
        RGB_BANDS = (55, 41, 12)
        LABEL_VALUES = ['Undefined', 'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tress', 'Soil', 'Water',
                        'Residential', 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot1', 'Parking Lot2',
                        'Tennis Court', 'Running Track']
    palette = None
    ignored_labels = [0]
    viz = visdom.Visdom(env='Houston')
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
    display_predictions(convert_to_color(train_gt, palette), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt, palette), viz, caption="Test ground truth")
    mask = np.zeros(gt.shape, dtype='bool')
    IGNORED_LABELS = ignored_labels
    # for a in IGNORED_LABELS:
    #     mask[gt == a] = True
    prediction[mask] = 0
    color_prediction = convert_to_color(prediction, palette)
    return color_prediction, viz, palette


def HSI_to_superpixels(img, num_superpixel, is_pca=False, is_show_superpixel=False):
    n_row, n_col, n_band = img.shape
    pca_dim = 3
    if is_pca:
        img = scale(img.reshape(n_row * n_col, -1))  # .reshape((n_row, n_column, -1))
        pca = PCA(n_components=pca_dim)
        img = pca.fit_transform(img).reshape((n_row, n_col, -1))
        # pca = PCA(n_components=0.95)
        # img = pca.fit_transform(scale(img.reshape(-1, pca_dim))).reshape(n_row, n_col, -1)

    # superpixel_label = slic(img, n_segments=num_superpixel, compactness=20, max_num_iter=10, convert2lab=False,
    #                         enforce_connectivity=True, min_size_factor=0.3, max_size_factor=2, slic_zero=False)
    superpixel_label = slic(img, n_segments=num_superpixel, compactness=20, max_num_iter=10)
    # superpixel_label = quickshift(img, ratio=1.0, kernel_size=7, max_dist=125)
    if is_show_superpixel:
        x = minmax_scale(img[:, :, :3].reshape(-1, 3)).reshape(n_row, n_col, -1)
        # color = (162/255, 169/255, 175/25)
        color = (132/255, 133/255, 135/255)
        mask = mark_boundaries(x, superpixel_label, color=color, mode='subpixel')
        mask_boundary = find_boundaries(superpixel_label, mode='subpixel')
        # mask_ = np.ones((mask_boundary.shape[0], mask_boundary.shape[1], 3))
        mask[mask_boundary] = color
        plt.figure()
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
    return superpixel_label

def show_superpixel(label, x=None):
    color = (132 / 255, 133 / 255, 135 / 255)
    if x is not None:
        color = (162/255, 169/255, 175/25)
        x = minmax_scale(x.reshape(label.shape[0] * label.shape[1], -1))
        x = x.reshape(label.shape[0], label.shape[1], -1)
        mask = mark_boundaries(x[:, :, :3], label, color=(1, 1, 0), mode='outer')
    else:
        mask_boundary = find_boundaries(label, mode='subpixel')
        mask = np.ones((mask_boundary.shape[0], mask_boundary.shape[1], 3))
        mask[mask_boundary] = color
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig('superpixel.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


def create_association_mat(superpixel_labels):
    labels = np.unique(superpixel_labels)
    # print(labels)
    n_labels = labels.shape[0]
    print('num superpixel: ', n_labels)
    n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
    association_mat = np.zeros((n_pixels, n_labels))
    superpixel_labels_ = superpixel_labels.reshape(-1)
    for i, label in enumerate(labels):
        association_mat[np.where(label == superpixel_labels_), i] = 1
    return association_mat


def create_spixel_graph(source_img, superpixel_labels):
    s = source_img.reshape((-1, source_img.shape[-1]))  # [21025, 200]
    a = create_association_mat(superpixel_labels)  # [21025, 196]
    # t = superpixel_labels.reshape(-1)
    mean_fea = np.matmul(a.T, s)  # [196, 200]
    regions = regionprops(superpixel_labels + 1)  # 196
    n_labels = np.unique(superpixel_labels).shape[0]  # 196
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid  # centroid coordinates
    ss_fea = np.concatenate((mean_fea, center_indx), axis=1)  # [196, 202]
    ss_fea = minmax_scale(ss_fea)
    # [196, 196]
    adj = kneighbors_graph(ss_fea, n_neighbors=50, mode='distance', include_self=False).toarray()  # [196, 196]

    # # # show initial graph
    # import matplotlib.pyplot as plt
    # adj_ = np.copy(adj)
    # adj_[np.where(adj != 0)] = 1
    # plt.imshow(adj_, cmap='hot')
    # plt.show()

    # # auto calculate gamma in Gaussian kernel
    X_var = ss_fea.var()
    gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    adj[np.where(adj != 0)] = np.exp(-np.power(adj[np.where(adj != 0)], 2) * gamma)

    # adj = euclidean_dist(ss_fea, ss_fea).numpy()
    # adj = np.exp(-np.power(adj, 2) * gamma)
    np.fill_diagonal(adj, 0)

    # show_graph(adj, center_indx)
    return adj, center_indx


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    import torch
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
    return dist

# def cosine_sim_with_temperature(x, temperature=0.5):
#     x = normalize(x)
#     sim = np.matmul(x, x.T) / temperature  # Dot similarity


def show_graph(adj, node_pos):
    plt.style.use('seaborn-white')
    D = np.diag(np.reshape(1./np.sum(adj, axis=1), -1))
    adj = np.dot(D, adj)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    G = nx.from_numpy_array(adj)
    for i in range(node_pos.shape[0]):
        G.nodes[i]['X'] = node_pos[i]
    # edge_weights = [(u, v) for u, v in G.edges()]
    pos = nx.get_node_attributes(G, 'X')
    # nx.draw(G, pos=pos, node_size=40, node_color='b', edge_color='black')  #  #fabebe # white
    nx.draw(G, pos=pos, node_size=40, node_color='#CD3700')  # #fabebe # white
    norm_v = mpl.colors.Normalize(vmin=0, vmax=adj.max())
    cmap = cm.get_cmap(name='PuBu')
    m = cm.ScalarMappable(norm=norm_v, cmap=cmap)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=d['weight'] * 5, alpha=0.5, edge_color=m.to_rgba(d['weight']))
    # draw graph
    # nx.draw(G, pos=pos, node_size=40)
    # nx.draw(G, pos=pos, node_size=node_size, node_color=color)
    plt.show()


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with log_softmax, which is more numerically stable."""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y, labels, reduction='mean'):
        log_p = torch.log_softmax(y, dim=1)
        return F.nll_loss(log_p, labels, reduction=reduction)


class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """Self-training loss with confidence threshold."""

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.criterion = CrossEntropyLoss()

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (self.criterion(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels


class DynamicThresholdingModule(object):
    r"""
    Dynamic thresholding module from `FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling
    <https://arxiv.org/abs/2110.08263>`_. At time :math:`t`, for each category :math:`c`,
    the learning status :math:`\sigma_t(c)` is estimated by the number of samples whose predictions fall into this class
    and above a threshold (e.g. 0.95). Then, FlexMatch normalizes :math:`\sigma_t(c)` to make its range between 0 and 1
    .. math::
        \beta_t(c) = \frac{\sigma_t(c)}{\underset{c'}{\text{max}}~\sigma_t(c')}.
    The dynamic threshold is formulated as
    .. math::
        \mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau,
    where \tau denotes the pre-defined threshold (e.g. 0.95), :math:`\mathcal{M}` denotes a (possibly non-linear)
    mapping function.
    Args:
        threshold (float): The pre-defined confidence threshold
        warmup (bool): Whether perform threshold warm-up. If True, the number of unlabeled data that have not been
            used will be considered when normalizing :math:`\sigma_t(c)`
        mapping_func (callable): An increasing mapping function. For example, this function can be (1) concave
            :math:`\mathcal{M}(x)=\text{ln}(x+1)/\text{ln}2`, (2) linear :math:`\mathcal{M}(x)=x`,
            and (3) convex :math:`\mathcal{M}(x)=2/2-x`
        num_classes (int): Number of classes
        n_unlabeled_samples (int): Size of the unlabeled dataset
        device (torch.device): Device
    """

    def __init__(self, threshold, mapping_func, num_classes, n_unlabeled_samples):
        self.threshold = threshold
        self.warmup = False
        self.mapping_func = mapping_func
        self.num_classes = num_classes
        self.n_unlabeled_samples = n_unlabeled_samples
        self.net_outputs = torch.zeros(n_unlabeled_samples, dtype=torch.long).cuda()
        self.net_outputs.fill_(-1)

    def get_threshold(self, pseudo_labels):
        """Calculate and return dynamic threshold"""
        pseudo_counter = Counter(self.net_outputs.tolist())
        if max(pseudo_counter.values()) == self.n_unlabeled_samples:
            # In the early stage of training, the network does not output pseudo labels with high confidence.
            # In this case, the learning status of all categories is simply zero.
            status = torch.zeros(self.num_classes).cuda()
        else:
            if not self.warmup and -1 in pseudo_counter.keys():
                pseudo_counter.pop(-1)
            max_num = max(pseudo_counter.values())
            # estimate learning status
            status = [
                pseudo_counter[c] / max_num for c in range(self.num_classes)
            ]
            status = torch.FloatTensor(status).cuda()
        # calculate dynamic threshold
        dynamic_threshold = self.threshold * self.mapping_func(status[pseudo_labels])
        return dynamic_threshold

    def update(self, idxes, selected_mask, pseudo_labels):
        """Update the learning status
        Args:
            idxes (tensor): Indexes of corresponding samples
            selected_mask (tensor): A binary mask, a value of 1 indicates the prediction for this sample will be updated
            pseudo_labels (tensor): Network predictions
        """
        if idxes[selected_mask == 1].nelement() != 0:
            self.net_outputs[idxes[selected_mask == 1]] = pseudo_labels[selected_mask == 1]


def self_training_loss(logits_u_s, logits_u_w, class_acc, ts_hold):

    logits_u_w = logits_u_w.detach()
    pseudo_label = torch.softmax(logits_u_w, dim=1)

    max_probs, max_idx = torch.max(pseudo_label, dim=1)
    # mask = max_probs.ge(ts_hold * class_acc[max_idx]).float()  # linear
    # mask = max_probs.ge(ts_hold * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
    # mask = max_probs.ge(ts_hold * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
    # mask = max_probs.ge(ts_hold * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
    mask = max_probs.ge(ts_hold * (class_acc[max_idx] + 0.5) / (class_acc[max_idx] + 0.5 + torch.exp(-5 * class_acc[max_idx]))).float()
    select = max_probs.ge(ts_hold).long()
    masked_loss = F.cross_entropy(logits_u_s, max_idx, reduction='none') * mask
    return masked_loss.mean(), mask, select, max_idx.long()

