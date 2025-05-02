from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.resnet_fta2c import ResNet18_FTA2C as ResNet18_FTA2C
from models.vgg_fta2c import vgg16_FTA2C as Vgg16_FTA2C
from models.wideresnet34_fta2c import WideResNet34_FTA2C
from models.resnet import ResNet50
from TinyImagedata import ValTinyImageNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--dataset', default='cifar10', help='use what dataset')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num_steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step_size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--model_path',
                    default='model_trades_best_adv.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source_model_path',
                    default='model_black.pth',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target_model_path',
                    default='model_trades_best_adv.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white_box_attack', default=False,
                    help='whether perform white-box attack')
parser.add_argument('--device', type=int, default=0, help='device id')
parser.add_argument('--beta', default=2.0,
                    help='regularization')

args = parser.parse_args()
print(args)
# settings
use_cuda = torch.cuda.is_available()
device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

model_path = './{}/{}/{}'.format(args.model, args.dataset, args.model_path)
source_model_path = './{}/{}/{}'.format(args.model, args.dataset, args.source_model_path)
target_model_path = './{}/{}/{}'.format(args.model, args.dataset, args.target_model_path)

# set up data loader
if args.dataset == 'cifar10':
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif args.dataset == 'svhn':
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN('./data', split='test', download=True,
                                  transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False)
elif args.dataset == "cifar100":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
elif args.dataset == "TinyImageNet":
    #print("use TinyImageNet")
    image_size = (64, 64)
    root = '../../data/tiny-imagenet-200'
    id_dic = {}
    for i, line in enumerate(open(root + '/wnids.txt', 'r')):
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    data_transform = {
        "val": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])}
    testset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])

    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.test_batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=4)

    #print("TinyImageNet Loading SUCCESS" + "\nlen of train dataset: " + str(len(trainset)) + "\nlen of val dataset: " + str(len(valset)))
elif args.dataset == "ImageNet":
    from torch.utils.data import DataLoader
    from traffic_imagenet import Traffic
    root = "../../data/1111"
    image_size = (224, 224)
    num_classes = 30
    # 引用数据集
    testset = Traffic(root, 244, mode="val")

    # 加载数据集
    test_loader = DataLoader(testset, batch_size=args.test_batch_size)
    # 数据集长度
    data_val_len = len(testset)

    print("data_val_len={}".format(data_val_len))

    print("num_classes:", num_classes)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  random=True,
                  beta=1.):
    # out, _ = model(X, _eval=True)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output,_,_,extra_output,_,_ = model(X_pgd, y)
            loss = nn.CrossEntropyLoss()(output, y)
            extra_loss = 0.
            for output in extra_output:
                extra_loss += nn.CrossEntropyLoss()(output, y)
            extra_loss /= len(extra_output)
            loss += beta * extra_loss
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    adv_output,_,_,_,_,_= model(X_pgd)
    err_pgd = (adv_output.data.max(1)[1] != y.data).float().sum()
    return err_pgd

def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  beta=args.beta):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output,_,_,extra_output,_,_ = model(X_pgd, y)
            correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
            tmp1 = torch.argsort(output, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
            loss = - F.relu(correct_logit-wrong_logit)
            extra_loss = 0.
            for output in extra_output:
                correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
                tmp1 = torch.argsort(output, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
                extra_loss += - F.relu(correct_logit - wrong_logit)
            extra_loss /= len(extra_output)
            loss += beta * extra_loss

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    output,_,_,_,_,_ = model(X_pgd)
    err_pgd = (output.data.max(1)[1] != y.data).float().sum()
    return err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out,_,_,_,_,_ = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            out_source = model_source(X_pgd)
            if isinstance(out_source, tuple):
                out_source = out_source[0]
            loss = nn.CrossEntropyLoss()(out_source, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd)[0].data.max(1)[1] != y.data).float().sum()
    #print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def clean_test(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out,_,_,_,_,_ = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        err_total += err
    print('Clean Acc: ', 1 - err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_pgd20(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, random=True, beta=args.beta)
        robust_err_total += err_robust
    print('PGD20 robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_various_epsilon(model, device, test_loader):
    model.eval()
    test_epsilon = [2.0/255, 4.0/255, 8.0/255, 16.0/255, 32.0/255]
    for epsilon in test_epsilon:
        robust_err_total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_robust = _pgd_whitebox(model, X, y, random=True, beta=args.beta, epsilon=epsilon, step_size=epsilon/10.0)
            robust_err_total += err_robust
        print('PGD20 epsilon %.4f robust_acc: '% (epsilon), 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_pgd100(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, epsilon=args.epsilon,
                  num_steps=100, step_size=args.step_size, random=True, beta=args.beta)
        robust_err_total += err_robust
    print('PGD100 robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_fgsm(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, epsilon=8/255.0, num_steps=1,
                                step_size=8/255.0, random=True, beta=args.beta)
        robust_err_total += err_robust
    print('FGSM robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_cw(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _cw_whitebox(model, X, y, beta=args.beta)
        robust_err_total += err_robust
    print('cw robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))



def eval_adv_test_blackbox_fsgm(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y, epsilon=8/255.0, num_steps=1,
                                step_size=8/255.0)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', 1 - natural_err_total / len(test_loader.dataset))
    print('fsgm robust_err_total: ', 1 - robust_err_total / len(test_loader.dataset))


def eval_adv_test_blackbox_cw(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _cw_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
    print('cw_robust_err_total: ', 1 - robust_err_total / len(test_loader.dataset))


def _cw_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=30,
                  step_size=args.step_size):

    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output = model_source(X_pgd)
            if isinstance(output, tuple):
                output = output[0]
            correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
            tmp1 = torch.argsort(output, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
            loss = - F.relu(correct_logit-wrong_logit)

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    output = model_target(X_pgd)
    if isinstance(output, tuple):
        output = output[0]
    err_pgd = (output.data.max(1)[1] != y.data).float().sum()
    return err_pgd



def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_err_total: ', 1 - robust_err_total / len(test_loader.dataset))


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

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        if args.model == "resnet18":
            model = ResNet18_FTA2C(num_classes=num_classes, image_size=image_size).to(device)
        elif args.model == "vgg":
            model = Vgg16_FTA2C(num_classes=num_classes, image_size=image_size).to(device)
        elif args.model == "wideresnet":
            model = WideResNet34_FTA2C(num_classes=num_classes, image_size=image_size).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        clean_test(model, device, test_loader)
        #eval_adv_various_epsilon(model, device, test_loader)
        eval_adv_test_whitebox_pgd20(model, device, test_loader)
        eval_adv_test_whitebox_fgsm(model, device, test_loader)
        eval_adv_test_whitebox_cw(model, device, test_loader)
        eval_adv_test_whitebox_pgd100(model, device, test_loader)

    else:
        # black-box attack
        print('pgd black-box attack')
        if args.model == "resnet18":
            model_target = ResNet18_FTA2C().to(device)
            model_target.load_state_dict(torch.load(target_model_path, map_location=device))
            model_source = ResNet50().to(device)
            model_source.load_state_dict(torch.load(source_model_path, map_location=device))
        elif args.model == "vgg":
            model_target = Vgg16_FTA2C().to(device)
            model_target.load_state_dict(torch.load(target_model_path))
            model_source = Vgg16_FTA2C().to(device)
            model_source.load_state_dict(torch.load(source_model_path))

        eval_adv_test_blackbox_fsgm(model_target, model_source, device, test_loader)
        eval_adv_test_blackbox(model_target, model_source, device, test_loader)
        eval_adv_test_blackbox_cw(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
