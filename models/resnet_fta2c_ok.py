import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gumbel_sigmoid import GumbelSigmoid


class Separation(torch.nn.Module):
    def __init__(self, size, num_channel=64, tau=0.1):
        super(Separation, self).__init__()
        C, H, W = size
        #print('Separation size {}'.format(size))
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )
        #print('Separation')
        #print(self.sep_net)

    def forward(self, feat, is_eval=False):
        #print('feat shape {}'.format(feat.shape))
        rob_map = self.sep_net(feat)
        #print(rob_map.shape)
        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        #print(mask.shape)
        mask = torch.nn.Sigmoid()(mask)
        #print(mask.shape)
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        #print(mask.shape)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)

        #print(mask.shape)
        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask


#NoRobustTranformer
class NoRobustTranformer(nn.Module):
    def __init__(self, Channel, num_channel=64):
        super(NoRobustTranformer, self).__init__()
        C = Channel
        self.rec_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, nr_feat, mask):
        rec_units = self.rec_net(nr_feat)
        rec_units = rec_units * (1 - mask)

        return rec_units



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        #print(self)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#NoRobustCompress = CASBlock
class NoRobustCompress(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_classes=10):
        super(NoRobustCompress, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.fc = nn.Linear(planes, num_classes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, label=None, is_train=False):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)  #残差连接

        out = F.relu(out)
        # print(out.shape)
        fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1)
        fc_out = self.fc(fc_in.view(out.shape[0], out.shape[1]))
        #print('is_train {}'.format(is_train))
        if is_train:
            N, C, H, W = out.shape
            mask = self.fc.weight[label, :]
            out = out * mask.view(N, C, 1, 1)
        else:
            N, C, H, W = out.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            out = out * mask.view(N, C, 1, 1)
        return out, fc_out


#norfeaturecom = layer5,norfeaturetran = recalibration
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, tau=0.1, image_size=(32, 32)):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.tau = tau
        self.image_size = image_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.separation = Separation(size=(512, int(self.image_size[0] / 8), int(self.image_size[1] / 8)), tau=self.tau) #分离网络
        self.norfeaturetran = NoRobustTranformer(Channel = 512) #非鲁棒调整
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.norfeaturecom = self._make_layer_(NoRobustCompress, 512, num_blocks[3], stride=1, num_classes=num_classes)  #非鲁棒压缩网络

        self.aux = nn.Sequential(nn.Linear(512, num_classes))

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def _make_layer_(self, block, planes, num_blocks, stride, num_classes):
        strides = [stride] + [1] * (num_blocks - 1)
        #print('_make_layer_ strides {}'.format(strides))
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_classes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=None, is_eval=False, is_train=False):
        r_outputs = []
        nr_outputs = []
        class_wise_output = []
        rec_outputs = []

        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)

        out1 = out
        feature4 = F.avg_pool2d(out1, 4).view(out1.size(0), -1)

        r_feat, nr_feat, mask = self.separation(out, is_eval=is_eval)

        feature4_rf = F.avg_pool2d(r_feat, 4).view(r_feat.size(0), -1)
        feature4_nrf = F.avg_pool2d(nr_feat, 4).view(nr_feat.size(0), -1)

        sepnr_feat = nr_feat
        for layer in self.norfeaturecom:
            sepnr_feat, layer5_out = layer(sepnr_feat, y, is_train=is_train)
            class_wise_output.append(layer5_out)

        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)

        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        rec_feat = self.norfeaturetran(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        out = r_feat + rec_feat + sepnr_feat

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        # out为鲁棒卷积后+非鲁棒压缩后激活、r_outputs为分离的鲁棒、nr_outputs是非鲁棒激活、class_wise_output为非鲁棒的压缩损失,rec_outputs为非鲁棒调整的激活
        return out, r_outputs, nr_outputs, class_wise_output, rec_outputs, (feature4, feature4_rf, feature4_nrf)

def ResNet18_FTA2C(num_classes=10, tau=0.1, image_size=(32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, tau=tau, image_size=image_size)
