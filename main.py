import math
from copy import deepcopy
from itertools import cycle
import argparse
import numpy as np
import torch
import torch.optim as optim
from collections import Counter
from loader import get_dataloaders
import metric
from utils import AverageMeter, set_model, DrawCluster, visualization, result_display, display_predictions, convert_to_color, DrawResult
import utils
import torch.nn.functional as F


def train(model,
          optim,
          lr_schdlr,
          args,
          selected_label,
          classwise_acc,
          labeled_train_loader,
          unlabeled_train_loader,
          ):
    cls_losses = AverageMeter()
    self_training_losses = AverageMeter()
    # define loss function
    cls_criterion = utils.CrossEntropyLoss()
    selected_label = selected_label.cuda()

    model.train()
    for batch_idx, data in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        x_l, labels_l = data[0][0], data[0][1]
        x_u, x_u_strong, labels_u = data[1][0], data[1][1], data[1][2]
        x_l = x_l.cuda()
        x_u = x_u.cuda()
        x_u_strong = x_u_strong.cuda()
        labels_l = labels_l.cuda()
        batch_size = x_l.size(0)
        t = list(range(batch_size*batch_idx, batch_size*(batch_idx+1), 1))
        t = (torch.from_numpy(np.array(t))).cuda()

        # --------------------------------------
        x = torch.cat((x_l, x_u, x_u_strong), dim=0)
        y, y_pseudo = model(x)
        # cls loss on labeled data
        y_l = y[:args.batch_size]
        cls_loss = cls_criterion(y_l, labels_l)
        # self training loss on unlabeled data
        y_u, _ = y[args.batch_size:].chunk(2, dim=0)
        _, y_u_strong = y_pseudo[args.batch_size:].chunk(2, dim=0)

        #
        confidence, pseudo_labels = torch.softmax(y_u.detach(), dim=1).max(dim=1)
        mask = confidence.ge(0.95 * ((-0.3) * (torch.pow((classwise_acc[pseudo_labels] - 1), 2)) + 1)).float()
        #     self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # else:
        self_training_loss = (F.cross_entropy(y_u_strong, pseudo_labels, reduction='none') * mask).mean()
        # if batch_idx == 100:
        #     print(t_p)
        #     print(confidence.mean())
        if t[mask == 1].nelement() != 0:
            selected_label[t[mask == 1]] = pseudo_labels[mask == 1]

        loss = cls_loss + self_training_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        cls_losses.update(cls_loss.item())
        self_training_losses.update(self_training_loss.item())
    return cls_losses.avg, self_training_losses.avg, selected_label


# test for one epoch
def test(model, test_loader):
    model.eval()
    total_accuracy,  total_num = 0.0, 0.0
    prediction = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            logits, _ = model(data)
            out = torch.softmax(logits, dim=1)
            pred_labels = out.argsort(dim=-1, descending=True)
            total_num += data.size(0)
            total_accuracy += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            for num in range(len(logits)):
                prediction.append(np.array(logits[num].cpu().detach().numpy()))

    return total_accuracy / total_num * 100, prediction


def main():
    dataset_names = ['KSC' 'PU', 'Houston']
    parser = argparse.ArgumentParser(description='Pseudo label for HSIC')
    parser.add_argument('--dataset', type=str, default='Houston', choices=dataset_names)
    parser.add_argument('--model', default='SAT', type=str, choices='Network')
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for last conv')
    parser.add_argument('--batch_size', default=30, type=int, help='Number of data in each mini-batch')
    parser.add_argument('--epoches', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--runs', type=int, default=10, help='number of training times')
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epoches
    # data prepare
    for n in range(0, args.runs):
        print(f"----Now begin the {format(n)} run----")
        labeled_train_loader, _, unlabeled_train_loader, _, _, _, unlabeled_dataset = get_dataloaders(batchsize=batch_size, n=n)
        _, test_loader, _, TestLabel, TestPatch, pad_width_data, _ = get_dataloaders(batchsize=batch_size, n=n)

        args.bands = int(TestPatch.shape[1])
        args.num_classes = len(np.unique(TestLabel))
        model = set_model(args)
        model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], 0.2, last_epoch=-1)
        args.patchsize = 24
        args.threshold = 0.95
        # training loop
        best_acc, best_epoch = 0.0, 0
        selected_label = torch.ones((len(unlabeled_dataset),), dtype=torch.long, ) * -1
        classwise_acc = torch.zeros((args.num_classes, )).cuda()
        for epoch in range(1, epochs+1):
            pseudo_counter = Counter(selected_label.tolist())
            # print('pseudo_counter:', pseudo_counter)
            if max(pseudo_counter.values()) < len(unlabeled_dataset):
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
            loss_x, loss_u, selected_label = \
                train(model=model, optim=optimizer, lr_schdlr=lr_scheduler, classwise_acc=classwise_acc,
                      selected_label=selected_label, labeled_train_loader=labeled_train_loader,
                      unlabeled_train_loader=unlabeled_train_loader,  args=args)
            test_acc, predictions = test(model, test_loader)
            print('Epoch: [{}/{}] | classify loss_x:{:.4f} | classify loss_u:{:.4f} | Test Acc:{:.2f}'
                  .format(epoch, epochs, loss_x, loss_u, test_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(model, 'results/best_acc_result1.pth')
        print('Best test_acc_1: {:.2f} at epoch {}'.format(best_acc, best_epoch))

        model = torch.load('results/best_acc_result1.pth')
        model.eval()

        pred_y = np.empty((len(TestLabel)), dtype='float32')
        number = len(TestLabel) // 100

        for i in range(number):
            temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
            temp = temp.cuda()
            temp2, _ = model(temp)
            #  _, temp2, _, _, _, = model(temp, temp, temp, temp)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
            del temp, temp2, temp3, _
        # 不足100个的情况
        if (i + 1) * 100 < len(TestLabel):
            temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
            temp = temp.cuda()
            temp2, _ = model(temp)
            # _, _, _,  _, temp2, _, _, _ = model(temp, temp, temp, temp)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
            del temp, temp2, temp3, _

        # 评价指标
        pred_y = torch.from_numpy(pred_y).long()
        Classes = np.unique(TestLabel)
        EachAcc = np.empty(len(Classes))
        AA = 0.0
        for i in range(len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0
            for j in range(len(TestLabel)):
                if TestLabel[j] == cla:
                    sum += 1
                if TestLabel[j] == cla and pred_y[j] == cla:
                    right += 1
            EachAcc[i] = right.__float__() / sum.__float__()
            AA += EachAcc[i]

        print('-------------------')
        for i in range(len(EachAcc)):
            print('|第%d类精度：' % (i + 1), '%.2f|' % (EachAcc[i] * 100))
            print('-------------------')
        AA *= 100 / len(Classes)

        results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
        print('test accuracy（OA）: %.2f ' % results["Accuracy"], 'AA : %.2f ' % AA, 'Kappa : %.2f ' % results["Kappa"])
        print('confusion matrix :')
        print(results["Confusion matrix"])


if __name__ == '__main__':
    main()



