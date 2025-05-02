from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from trades_reg import trades_loss
from models.resnet_fta2c import ResNet18_FTA2C as ResNet18_FTA2C
from models.vgg_fta2c import vgg16_FTA2C as Vgg16_FTA2C
from models.wideresnet34_fta2c import WideResNet34_FTA2C
from torch.autograd import Variable

from tqdm.auto import tqdm
from TinyImagedata import TrainTinyImageNet, ValTinyImageNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
parser.add_argument('--step_size', type=float, default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', type=float, default=4.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--device', type=int, default=0, help='device id')
parser.add_argument('--cr_beta', default=2.0,
                    help='regularization, i.e., beta in channel regularization')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', type=str, default='cifar10')

args = parser.parse_args()

print(args)
# settings
model_dir = os.path.join(args.model, args.dataset)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logfile = os.path.join(model_dir, 'log.txt')
log_writer = open(logfile, 'a')

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            log_writer.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output,_,_,extra_output,_,_ = model(data)
            train_loss += F.cross_entropy(output, target).item()
            # extra_loss = 0.
            # for output in extra_output:
            #     extra_loss += F.cross_entropy(output, target, size_average=False).item()
            # extra_loss /= len(extra_output)
            # train_loss += args.cr_beta * extra_loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    log_writer.write('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_,_,extra_output,_,_ = model(data)
            test_loss += F.cross_entropy(output, target).item()
            # for output in extra_output:
            #     test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test clean: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    log_writer.write('Test clean: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output,_,_,extra_ouput,_,_ = model(X_pgd, y)
            loss = nn.CrossEntropyLoss()(output, y)
            extra_loss = 0.
            for output in extra_ouput:
                extra_loss += nn.CrossEntropyLoss()(output, y)
            extra_loss /= len(extra_ouput)
            loss += args.cr_beta * extra_loss
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            adv_data = _pgd_whitebox(model, data, target, epsilon=0.031, num_steps=20, step_size=0.003)
            output,_,_,extra_output,_,_ = model(adv_data)
            test_loss += F.cross_entropy(output, target).item()
            # extra_loss = 0.
            # for output in extra_output:
            #     extra_loss += F.cross_entropy(output, target, size_average=False).item()
            # extra_loss /= len(extra_output)
            # test_loss += args.cr_beta * extra_loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test Adv: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    log_writer.write('Test Adv: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    # init model, ResNet18() can be also used here for training
    if args.model == "resnet18":
        model = ResNet18_FTA2C(num_classes=num_classes, image_size=image_size).to(device)
    elif args.model == "vgg":
        model = Vgg16_FTA2C(num_classes=num_classes, image_size=image_size).to(device)
    elif args.model == "wideresnet":
        model = WideResNet34_FTA2C(num_classes=num_classes, image_size=image_size).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_adv_acc = 0.

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        log_writer.write('\n================================================================\n')
        eval_train(model, device, train_loader)
        _, nat_acc = eval_test(model, device, test_loader)
        _, adv_acc = eval_adv_test(model, device, test_loader)
        print('================================================================')
        log_writer.write('\n================================================================\n')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model_trades_last.pt'))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'model_trades_checkpoint_last.tar'))

        if adv_acc > best_adv_acc:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model_trades_best_adv.pt'))
            best_adv_acc = adv_acc

    log_writer.close()


if __name__ == '__main__':
    main()
