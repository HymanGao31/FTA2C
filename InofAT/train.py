import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from models.resnet_fta2c import ResNet18_FTA2C
# from models.resnet_fta2c_ok import ResNet18_FTA2C
from models.vgg_fta2c import vgg16_FTA2C
from models.wideresnet34_fta2c import WideResNet34_FTA2C

from attacks.pgd import PGD

from tqdm.auto import tqdm

import argparse
import os
from utils import create_logger
from time import time

import matplotlib.pyplot as plt

from utils import count_entropy, count_distance

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='FSR Training')
parser.add_argument('--lam_sep', type=float, default=1.0, help='weight for separation loss')
parser.add_argument('--lam_rec', type=float, default=1.0, help='weight for recalibration loss')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for classifier')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
parser.add_argument('--dataset', type=str, default='cifar10', help='target dataset')
parser.add_argument('--model', type=str, default='resnet18', help='model name')
parser.add_argument('--eps', type=float, default=8., help='perturbation constraint epsilon')
parser.add_argument('--alpha', type=float, default=0.25, help='step size alpha')
parser.add_argument('--tau', type=float, default=0.1, help='tau for Gumbel softmax')
parser.add_argument('--device', type=int, help='device id')
parser.add_argument('--beta', '-beta', type=float, default=1,
                    help='channel regularization')
parser.add_argument('--log_root', default='log',
                    help='the directory to save the logs or other imformations (e.g. images)')
args = parser.parse_args()

print(args)
model_dir = os.path.join(args.model, args.dataset)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
print(device)
start_epoch = 1
best_adv_acc = 0.

if args.dataset == 'cifar10':
    num_classes = 10
    image_size = (32, 32)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='constant', value=0).squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),   #随机裁剪
        transforms.RandomHorizontalFlip(),  #随机水平反转
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)

elif args.dataset == 'svhn':
    num_classes = 10
    image_size = (32, 32)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True)

    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False)

elif args.dataset == "cifar100":
    print("use cifar100")
    num_classes = 100
    image_size = (32, 32)
    # Prepare data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False, num_workers=4)

elif args.dataset == "TinyImageNet":
    #print("use TinyImageNet")
    image_size = (64, 64)
    # root = 'D:\code\FTA2C_12-26\data\\tiny-imagenet-200'
    root = '../data/tiny-imagenet-200'
    id_dic = {}
    # for i, line in enumerate(open(root + '\\wnids.txt', 'r')):
    for i, line in enumerate(open(root + '/wnids.txt', 'r')):
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    data_transform = {
        "train": transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                saturation=0.4, hue=0.1)],
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=3,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ]),
        "val": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])}
    trainset = TrainTinyImageNet(root, id=id_dic, transform=data_transform["train"])
    testset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])

    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=args.bs,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=1)
    testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=1)

    #print("TinyImageNet Loading SUCCESS" + "\nlen of train dataset: " + str(len(trainset)) + "\nlen of val dataset: " + str(len(valset)))

elif args.dataset == "ImageNet":
    from torch.utils.data import DataLoader
    from traffic_imagenet import Traffic
    root = "../data/1111"
    image_size = (224, 224)
    num_classes = 30
    # 引用数据集
    trainset = Traffic(root, 244, mode="train")
    testset = Traffic(root, 244, mode="val")

    # 加载数据集
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.bs)
    # 数据集长度
    data_train_len = len(trainset)
    data_val_len = len(testset)

    print("data_train_len={}, data_val_len={}".format(data_train_len, data_val_len))

    print("num_classes:,image_size", num_classes, image_size)


models = {
    'resnet18': ResNet18_FTA2C(tau=args.tau, num_classes=num_classes, image_size=image_size),
    'vgg16': vgg16_FTA2C(tau=args.tau, num_classes=num_classes, image_size=image_size),
    'wideresnet34': WideResNet34_FTA2C(tau=args.tau, num_classes=num_classes, image_size=image_size),
}

model_name = args.model
net = models[model_name]
net = net.to(device)
#print(net)
cudnn.benchmark = True


criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

save_folder = '%s_%s' % (args.dataset, args.model)

log_folder = os.path.join(args.log_root, save_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

setattr(args, 'log_folder', log_folder)

logger = create_logger(log_folder, 'info')

def get_pred(out, labels):
    out1 = out.sort(dim=-1, descending=True)
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)  #分类等于标签，返回second_pred，否则返回pred

    return adv_label


attack = PGD(net, args.eps/255.0, args.alpha * (args.eps/255.0), min_val=0, max_val=1, max_iters=10, _type='linf')


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

loss_bei_epoch = []
clr_accuracy_bei_epoch = []
adv_accuracy_bei_epoch = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    adv_cls_losses = 0
    sep_losses = 0
    rec_losses = 0
    adv_correct = 0
    sup_losses = 0
    rf_losses = 0
    total = 0

    # 记录训练过程中的损失值
    global train_loss_history
    global loss_bei_epoch
    global clr_accuracy_bei_epoch
    global adv_accuracy_bei_epoch

    adjust_learning_rate(optimizer, epoch)

    begin_time = time()

    with tqdm(total=(len(trainset) - len(trainset) % args.bs)) as _tqdm:
        _tqdm.set_description('(Train)Epoch:{}/{}'.format(epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print('batch_idx is {}'.format(batch_idx))
            net.eval() # net.train(False) 评估模型，关闭drop BN等
            adv_inputs = attack.perturb(inputs, targets, True, args.beta)
            net.train() #保证BN层用每一批次数据的均值和方差，和eval()保证BN用全部训练数据的均值和方差

            #adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = net(inputs)
            # out为鲁棒卷积后+非鲁棒压缩后激活、r_outputs为分离的鲁棒、nr_outputs是非鲁棒激活、class_wise_output为压缩损失
            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_sup_outputs, adv_rec_outputs, featrue = net(adv_inputs, targets, is_train=False)
            adv_labels = get_pred(adv_outputs, targets)  #为啥用第二大的结果作为非鲁棒的预测
            # adv_labels = targets

            y_pre, _, _, _, _, feature4 = net(inputs, targets, is_train=False)

            adv_out_f, adv_r_f, adv_nf_f = featrue
            cle_out_f, cle_r_f, cle_nf_f = feature4

            normed_clean = F.normalize(cle_r_f, dim=-1)
            matrix_clean = torch.mm(normed_clean, normed_clean.t())

            normed_feature = F.normalize(adv_r_f, dim=-1)
            matrix_adv = torch.mm(normed_feature, normed_feature.t())

            diff = torch.exp(torch.abs(matrix_adv - matrix_clean))
            loss_rf = torch.mean(diff)

            ########InfoAT#########
            clean_entropy_loss = count_entropy(y_pre)

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == targets).float())
            clr_accuracy_bei_epoch.append(accuracy.item())

            #y_pre_hat = net(adv_inputs)
            adver_entropy_loss = count_entropy(adv_outputs)
            # print("adver_entropy_loss: ", adver_entropy_loss)
            CE_loss = criterion(adv_outputs, targets)
            # print("CE_loss: ", CE_loss)

            distance = count_distance(y_pre, adv_outputs)
            # print("distance: ", distance)
            y_prediction = torch.max(adv_outputs, dim=1)[1]
            accuracy = torch.mean((y_prediction == targets).float())
            adv_accuracy_bei_epoch.append(accuracy.item())

            adv_cls_loss = CE_loss - 0.5 * adver_entropy_loss + 1 * clean_entropy_loss * distance


            # adv_cls_loss = criterion(adv_outputs, targets)
            # print('\n adv_cls_loss  is {}'.format(adv_cls_loss))

            r_loss = torch.tensor(0.).to(device)
            if not len(adv_r_outputs) == 0:
                for r_out in adv_r_outputs:
                    r_loss += args.lam_sep * criterion(r_out, targets)
                r_loss /= len(adv_r_outputs)

            nr_loss = torch.tensor(0.).to(device)
            if not len(adv_nr_outputs) == 0:
                for nr_out in adv_nr_outputs:
                    nr_loss += args.lam_sep * criterion(nr_out, adv_labels)
                nr_loss /= len(adv_nr_outputs)

            sep_loss = r_loss + nr_loss

            rec_loss = torch.tensor(0.).to(device)
            if not len(adv_rec_outputs) == 0:
                for rec_out in adv_rec_outputs:
                    rec_loss += args.lam_rec * criterion(rec_out, targets)
                rec_loss /= len(adv_rec_outputs)

            channel_reg_loss = 0.
            for extra_output in adv_sup_outputs:
                channel_reg_loss += F.cross_entropy(extra_output, targets)
            channel_reg_loss = channel_reg_loss / len(adv_sup_outputs)


            loss = adv_cls_loss + sep_loss + args.beta * channel_reg_loss + rec_loss + loss_rf

            loss_bei_epoch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            adv_cls_losses += adv_cls_loss.item()
            # print('adv_cls_loss is {},adv_cls_losses  is {},batch_idx is {}'.format(adv_cls_loss, adv_cls_losses, batch_idx))
            sep_losses += sep_loss.item()
            rec_losses += rec_loss.item()
            sup_losses += channel_reg_loss.item()
            rf_losses += loss_rf.item()

            totalloss = adv_cls_losses + sep_losses + rec_losses + sup_losses + rf_losses
            totalloss = totalloss / (batch_idx + 1)
            # 记录损失值
            train_loss_history.append(totalloss)

            _, adv_predicted = adv_outputs.max(1)
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()

            _tqdm.set_postfix(
                Adv_Loss='{:.3f}'.format(adv_cls_losses / (batch_idx + 1)),
                Sep_Loss='{:.3f}'.format(sep_losses / (batch_idx + 1)),
                Rec_Loss='{:.3f}'.format(rec_losses / (batch_idx + 1)),
                Sup_Loss='{:.3f}'.format(sup_losses / (batch_idx + 1)),
                rf_Loss='{:.3f}'.format(rf_losses / (batch_idx + 1)),
                Adv_Acc='{:.3f}%'.format(100. * adv_correct / total),
            )
            _tqdm.update(inputs.shape[0])

    logger.info('epoch: %d, spent %.2f s, adv_loss: %.3f, Sep_Loss: %.3f,Rec_Loss: %.3f,sup_losses:%.3f,rf_losses:%.3f,'
                ' Adv_Acc: %.3f %%' % (epoch, time() - begin_time, adv_cls_losses / (batch_idx + 1),
            sep_losses / (batch_idx + 1), rec_losses / (batch_idx + 1),
            rf_losses / (batch_idx + 1), sup_losses / (batch_idx + 1), 100. * adv_correct / total))

    return train_loss_history


def test(epoch):
    net.eval()
    ori_test_loss = 0
    adv_test_loss = 0
    ori_correct = 0
    adv_correct = 0
    total = 0

    global best_adv_acc

    begin_time = time()

    with tqdm(total=(len(testset) - len(testset) % args.bs), dynamic_ncols=True) as _tqdm:
        _tqdm.set_description('(Test)Epoch:{}/{}'.format(epoch, args.epoch))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = attack.perturb(inputs, targets, False)
            net.eval()

            ori_outputs, ori_r_outputs, ori_nr_outputs, ori_sup_outputs, ori_rec_outputs,_ = net(inputs, is_eval=True)
            adv_outputs, adv_r_outputs, adv_nr_outputs, adv_sup_outputs, adv_rec_outputs,_ = net(adv_inputs, is_eval=True)

            ori_loss = criterion(ori_outputs, targets)
            ori_test_loss += ori_loss.item()
            _, ori_predicted = ori_outputs.max(1)
            ori_correct += ori_predicted.eq(targets).sum().item()

            adv_loss = criterion(adv_outputs, targets)
            adv_test_loss += adv_loss.item()
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()

            total += targets.size(0)

            _tqdm.set_postfix(
                Ori_Loss='{:.3f}'.format(ori_test_loss/(batch_idx+1)),
                Ori_Acc='{:.3f}%'.format(100.*ori_correct/total),
                Adv_Loss='{:.3f}'.format(adv_test_loss/(batch_idx+1)),
                Adv_Acc='{:.3f}%'.format(100.*adv_correct/total),
            )
            _tqdm.update(inputs.shape[0])

    va_adv_acc = 100. * adv_correct / total


    if va_adv_acc > best_adv_acc:
        # torch.save(net.state_dict(), './weights/{}/{}/{}_best.pth'.format(args.dataset, args.model, args.save_name))
        torch.save(net.state_dict(),
                   os.path.join(model_dir, 'model_inofat_best.pt'))
        best_adv_acc = va_adv_acc

    logger.info('epoch: %d, spent %.2f s, Ori_Loss: %.3f, Ori_Acc: %.3f %%,Adv_Loss: %.3f, Adv_Acc: %.3f %%, best_adv_Acc: %.3f %%'
        % ( epoch, time() - begin_time, ori_test_loss / (batch_idx + 1), 100. * ori_correct / total, adv_test_loss / (batch_idx + 1), va_adv_acc, best_adv_acc))

    torch.save(net.state_dict(), os.path.join(model_dir, 'model_inofat.pt'))


# 记录训练过程中的损失值
train_loss_history = []

def main():
    for epoch in range(start_epoch, args.epoch + 1):
        train(epoch)
        test(epoch)
    # 绘制损失曲线
    plt.plot(train_loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

