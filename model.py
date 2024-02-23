import torch
import torch.nn as nn



class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Encoder1(nn.Module):
    def __init__(self, bands, feature_dim):
        super(Encoder1, self).__init__()
        self.dim1 = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, self.dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim1 * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim1 * 2, self.dim1 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim1 * 4, self.dim1 * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.conv1(x)  # [90, 64, 12, 12]
        x2 = self.conv2(x1)  # [90, 128, 6, 6]
        x3 = self.conv3(x2)  # [90, 256, 3, 3]
        return x3


class Projector(nn.Module):
    def __init__(self, low_dim):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(low_dim*2, 16)
        self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        # self.relu_mlp = nn.ReLU()
        self.fc2 = nn.Linear(16, low_dim)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu_mlp(x)
        x = self.fc2(x)
        x = self.l2norm(x)

        return x


class Classifier0(nn.Module):
    def __init__(self, n_classes):
        super(Classifier0, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.conv(x)
        # x1 = x.view(x.size(0), -1)
        # x = self.avg(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        # x = torch.softmax(x, dim=1)
        return x


class Supervisednetwork(nn.Module):
    def __init__(self, bands, n_classes, low_dim):
        super(Supervisednetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(low_dim*2, low_dim*2, 1),
            nn.BatchNorm2d(low_dim*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.encoder = Encoder1(bands, low_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(low_dim*2, n_classes)

    def forward(self, x):
        feature = self.encoder(x)
        x = self.conv(feature)
        # x = self.avgpool(feature)
        x = torch.flatten(x, start_dim=1)
        y = self.head(x)
        return y


class Classifier(nn.Module):

    def __init__(self, num_classes=10, feature_dim=256):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.features_dim = feature_dim
        # pseudo head and worst-case estimation head
        mlp_dim = 2 * self.features_dim
        self.head = nn.Linear(self.features_dim, num_classes)
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            # nn.Dropout(0.5),
            nn.Linear(mlp_dim, num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        y = self.head(x1)
        y_pseudo = self.pseudo_head(x1)
        # y_pseudo = self.head(x1)
        return y, y_pseudo
        # return y




class Network2(nn.Module):
    def __init__(self, bands, n_classes, low_dim):
        super(Network2, self).__init__()

        self.encoder = Encoder1(bands, low_dim)
        self.projector = Projector(low_dim)
        self.classifier = Classifier0(n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, u_w=None, u_s=None):
        if u_w==None and u_s==None :
            feature = self.encoder(x)
            cx = self.classifier(feature)
            feature = torch.mean(feature, dim=(2, 3))
            feature = self.projector(feature)
            return feature, cx

        feature1_3 = self.encoder(x)
        cx = self.classifier(feature1_3)
        feature1_3 = self.avgpool(feature1_3)
        feature1_3 = torch.mean(feature1_3, dim=(2, 3))
        feature1_3 = self.projector(feature1_3)

        feature2_3 = self.encoder(u_w)
        cuw = self.classifier(feature2_3)
        feature2_3 = self.avgpool(feature2_3)
        feature2_3 = torch.mean(feature2_3, dim=(2, 3))
        feature2_3 = self.projector(feature2_3)

        feature3_3 = self.encoder(u_s)
        cus = self.classifier(feature3_3)
        feature3_3 = self.avgpool(feature3_3)
        feature3_3 = torch.mean(feature3_3, dim=(2, 3))
        feature3_3 = self.projector(feature3_3)

        return feature1_3, feature2_3, feature3_3, cx, cuw, cus


class Network(nn.Module):
    def __init__(self, bands, n_classes, feature_dim):
        super(Network, self).__init__()
        self.encoder = Encoder1(bands, feature_dim)
        self.classifier = Classifier(n_classes, feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.encoder(x)
        y, y_pseudo = self.classifier(f)

        return y, y_pseudo
        # return y


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance
