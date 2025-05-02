'''
Some parts of the code are modified from:
CAS : https://github.com/bymavis/CAS_ICLR2021
CIFS : https://github.com/HanshuYAN/CIFS
'''


import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from models.BaseModel import BaseModelDNN
from TinyImagedata import ValTinyImageNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--load_name', type=str, default='cifar10_resnet18_best',help='specify checkpoint load name')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'


if args.model == 'resnet18':
    from models.resnet_fta2c import ResNet18_FTA2C
    net = ResNet18_FTA2C

elif args.model == 'vgg16':
    from models.vgg_fta2c import vgg16_FTA2C
    net = vgg16_FTA2C

elif args.model == 'wideresnet34':
    from models.wideresnet34_fta2c import WideResNet34_FTA2C
    net = WideResNet34_FTA2C

if args.dataset == 'cifar10':
    image_size = (32, 32)
    num_classes = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)

elif args.dataset == 'svhn':
    image_size = (32, 32)
    num_classes = 10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)

elif args.dataset == "cifar100":
    image_size = (32, 32)
    num_classes = 100
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.bs, shuffle=False, num_workers=4)
elif args.dataset == "TinyImageNet":
    #print("use TinyImageNet")
    image_size = (64, 64)
    root = '../data/tiny-imagenet-200'
    id_dic = {}
    for i, line in enumerate(open(root + '/wnids.txt', 'r')):
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    print('num_classes=', num_classes)
    data_transform = {
        "val": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])}
    testset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])

    testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=args.bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=4)

elif args.dataset == "ImageNet":
    from torch.utils.data import DataLoader
    from traffic_imagenet import Traffic
    root = "../data/1111"
    image_size = (224, 224)
    num_classes = 30
    # 引用数据集
    testset = Traffic(root, 244, mode="val")

    # 加载数据集
    testloader = DataLoader(testset, batch_size=args.bs)
    # 数据集长度
    data_val_len = len(testset)

    print("data_val_len={}".format(data_val_len))

    print("num_classes:", num_classes)


def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label


class CE_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits_final, target):
        loss = F.cross_entropy(logits_final, target)

        return loss


class CW_loss(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits_final, target):
        loss = self._cw_loss(logits_final, target, num_classes=self.num_classes)

        return loss

    def _cw_loss(self, output, target, confidence=50, num_classes=10):
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss


class Classifier(BaseModelDNN):
    def __init__(self) -> None:
        super(BaseModelDNN).__init__()
        self.net = net(tau=args.tau, num_classes=num_classes, image_size=image_size).to(device)
        self.set_requires_grad([self.net], False)

    def predict(self, x, is_eval=True):
        return self.net(x, is_eval=is_eval)


def main():
    model = Classifier()
    checkpoint = torch.load('./weights/{}/{}/{}.pth'.format(args.dataset, args.model, args.load_name), map_location=device)
    model.net.load_state_dict(checkpoint)
    model.net.eval()

    from advertorch_fta2c.attacks import FGSM, LinfPGDAttack

    lst_attack = [
        (FGSM, dict(
            loss_fn=CE_loss(),
            eps=8 / 255,
            clip_min=0.0, clip_max=1.0, targeted=False), 'FGSM'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=10, eps_iter=0.25 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-10'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=20, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-20'),
        (LinfPGDAttack, dict(
            loss_fn=CE_loss(),
            eps=8 / 255, nb_iter=100, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'PGD-100'),
        (LinfPGDAttack, dict(
            loss_fn=CW_loss(num_classes=num_classes),
            eps=8 / 255, nb_iter=30, eps_iter=0.1 * (8 / 255), rand_init=False,
            clip_min=0.0, clip_max=1.0, targeted=False), 'C&W'),
    ]
    attack_results = []
    for attack_class, attack_kwargs, name in lst_attack:
        from metric.classification import defense_success_rate

        message, defense_success, natural_success = defense_success_rate(model.predict,
                                                                         testloader, attack_class,
                                                                         attack_kwargs, device=device)

        message = name + ': ' + message
        print(message)
        attack_results.append(defense_success)
    attack_results.append(natural_success)
    attack_results = torch.cat(attack_results, 1)
    attack_results = attack_results.sum(1)
    attack_results[attack_results < len(lst_attack) + 1] = 0.
    if args.dataset == 'cifar10':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 10000.))
    elif args.dataset == 'svhn':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 26032.))
    elif args.dataset == 'cifar100':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 10000.))
    elif args.dataset == 'TinyImageNet':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 10000.))
    elif args.dataset == 'ImageNet':
        print('Ensemble : {:.2f}%'.format(100. * attack_results.count_nonzero() / 7752.))

if __name__ == '__main__':
    main()
