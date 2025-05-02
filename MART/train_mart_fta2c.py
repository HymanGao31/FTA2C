from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.resnet_fta2c import ResNet18_FTA2C as ResNet18_FTA2C
from models.vgg_fta2c import vgg16_FTA2C as Vgg16_FTA2C
from models.wideresnet34_fta2c import WideResNet34_FTA2C
from mart_reg import mart_loss
import numpy as np
import time
from tqdm.auto import tqdm
from TinyImagedata import TrainTinyImageNet, ValTinyImageNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num_steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step_size', type=float,default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', type=float,default=1.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--save_freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--device', type=int, default=0, help='device id')
parser.add_argument('--cr_beta', default=2.0,
                    help='weight before channel regularization')
parser.add_argument('--dataset', type=str, default='cifar10')

args = parser.parse_args()
print(args)
# settings
model_dir = os.path.join(args.model, args.dataset)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logfile = os.path.join(model_dir, 'log.txt')
log_writer = open(logfile, 'a')

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

if args.dataset == 'cifar10':
    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

elif args.dataset == 'svhn':
    # setup data loader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

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
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
elif args.dataset == "TinyImageNet":
    #print("use TinyImageNet")
    image_size = (64, 64)
    root = '../../data/tiny-imagenet-200'
    id_dic = {}
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

    train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=4)

    #print("TinyImageNet Loading SUCCESS" + "\nlen of train dataset: " + str(len(trainset)) + "\nlen of val dataset: " + str(len(valset)))

elif args.dataset == "ImageNet":
    from torch.utils.data import DataLoader
    from traffic_imagenet import Traffic
    root = "./data/1111"
    image_size = (224, 224)
    num_classes = 30
    # 引用数据集
    trainset = Traffic(root, 244, mode="train")
    testset = Traffic(root, 244, mode="val")

    # 加载数据集
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size)
    # 数据集长度
    data_train_len = len(trainset)
    data_val_len = len(testset)

    print("data_train_len={}, data_val_len={}".format(data_train_len, data_val_len))

    print("num_classes:", num_classes)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    with tqdm(total=(len(trainset) - len(trainset) % args.batch_size)) as _tqdm:
        _tqdm.set_description('(Train)Epoch:{}/{}'.format(epoch, args.epochs))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # calculate robust loss
            loss = mart_loss(model=model,
                             epoch=epoch,
                             device=device,
                             x_natural=data,
                             y=target,
                             optimizer=optimizer,
                             step_size=args.step_size,
                             epsilon=args.epsilon,
                             perturb_steps=args.num_steps,
                             beta=args.beta,
                             cr_beta=args.cr_beta)
            loss.backward()
            optimizer.step()

            # print progress
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                log_writer.write('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    out,_,_,_,_,_ = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output,_,_,extra_ouput,_,_ = model(X_pgd, y)
            loss = nn.CrossEntropyLoss()(output, y)
            for output in extra_ouput:
                loss += nn.CrossEntropyLoss()(output, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    adv_output,_,_,_,_,_ = model(X_pgd)
    err_pgd = (adv_output.data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    log_writer.write('natural_acc: %.4f \n'%(1 - natural_err_total / len(test_loader.dataset)))
    print('robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))
    log_writer.write('robust_acc: %.4f \n' % (1 - robust_err_total / len(test_loader.dataset)))
    return 1 - natural_err_total / len(test_loader.dataset), 1 - robust_err_total / len(test_loader.dataset)


def main():
    if args.dataset == "cifar100":
        num_classes = 100
        image_size = (32, 32)
    elif args.dataset == "TinyImageNet":
        num_classes = 200
        image_size = (64, 64)
    elif args.dataset == "ImageNet":
        num_classes = 30
        image_size = (224, 224)
    else:
        num_classes = 10
        image_size = (32, 32)

    if args.model == "resnet18":
        model = ResNet18_FTA2C(num_classes=num_classes, image_size=image_size).to(device)
    elif args.model == "vgg":
        model = Vgg16_FTA2C(num_classes=num_classes, image_size=image_size).to(device)
    elif args.model == "wideresnet":
        model = WideResNet34_FTA2C(num_classes=num_classes, image_size=image_size).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    natural_acc = []
    robust_acc = []

    best_adv = 0.

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        start_time = time.time()

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        print('================================================================')
        log_writer.write('================================================================\n')

        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader)


        print('using time:', time.time() - start_time)

        natural_acc.append(natural_err_total)
        robust_acc.append(robust_err_total)
        print('================================================================')
        log_writer.write('================================================================\n')

        if robust_err_total > best_adv:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model_mart_best.pt'))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'model_mart_checkpoint_best.tar'))
            best_adv = robust_err_total

        #file_name = os.path.join(log_dir, 'train_stats.npy')
        #np.save(file_name, np.stack((np.array(natural_acc), np.array(robust_acc))))

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model_mart_last.pt'))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'model_mart_checkpoint_last.tar'))


if __name__ == '__main__':
    main()
