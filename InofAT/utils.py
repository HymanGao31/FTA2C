import os
import json
import logging

import numpy as np

import torch

from scipy.stats import entropy

class LabelDict():
    def __init__(self, dataset='cifar-10'):
        self.dataset = dataset
        if dataset == 'cifar-10':
            self.label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 
                         4: 'deer',     5: 'dog',        6: 'frog', 7: 'horse',
                         8: 'ship',     9: 'truck'}

        self.class_dict = {v: k for k, v in self.label_dict.items()}

    def label2class(self, label):
        assert label in self.label_dict, 'the label %d is not in %s' % (label, self.dataset)
        return self.label_dict[label]

    def class2label(self, _class):
        assert isinstance(_class, str)
        assert _class in self.class_dict, 'the class %s is not in %s' % (_class, self.dataset)
        return self.class_dict[_class]


def count_entropy_old(input):
    num = len(input)
    input = input.cpu().detach().numpy()
    # count_array = Counter(pixels).values()
    total_loss = 0
    for i in input:
        total_loss += entropy(i)

    return total_loss / num

def count_entropy(y_pre_hat):
    # 将预测结果转换为概率分布
    y_pre_hat = torch.softmax(y_pre_hat, dim=1)
    # 计算每个类别的概率的负对数
    entropy = -torch.sum(y_pre_hat * torch.log(y_pre_hat), dim=1)
    # 返回平均熵损失
    return torch.mean(entropy)

def count_distance(clean, adver):
    num = len(clean)
    return torch.sum((clean-adver) ** 2)

def create_logger(save_path='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, 'my_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


def count_parameters(model):
    # copy from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    # baldassarre.fe's reply
    return sum(p.numel() for p in model.parameters() if p.requires_grad)