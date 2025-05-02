import torch
import torch.nn as nn
from models.gumbel_sigmoid import GumbelSigmoid
import torch.nn.functional as F

__all__ = [
    'vgg16_FTA2C',
]


class Separation(torch.nn.Module):
    def __init__(self, size, num_channel=64, tau=0.1):
        super(Separation, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, feat, is_eval=False):
        mask = self.sep_net(feat)

        mask = mask.reshape(mask.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)

        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask


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
        # rec_feat = nr_feat + rec_units

        # return rec_feat
        return rec_units


class Conv_FC_Block(nn.Module):

    def __init__(self, chann_in, chann_out, k_size, p_size):
        super(Conv_FC_Block, self).__init__()
        self.conv = nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size)
        self.bn = nn.BatchNorm2d(chann_out)
        self.relu = nn.ReLU()
        self.channel_out = chann_out
        self.fc = nn.Linear(self.channel_out, 10)

    def forward(self, x, label=None, is_train=False):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1)
        fc_out = self.fc(fc_in.view(out.shape[0], out.shape[1]))
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


def vgg_conv_block_channel_reg(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [Conv_FC_Block(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    # layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.ModuleList(layers)

#norfeaturecom = layer5,norfeaturetran = recalibration
class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True, tau=0.1, image_size=(32, 32)):
        super(VGG, self).__init__()
        self.image_size = image_size

        self.block0, self.pool0, self.block1, self.pool1, self.block2, self.pool2, self.block3, self.pool3, self.block4, self.pool4 = features
        self.tau = tau

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        self.separation = Separation(size=(512, int(self.image_size[0] / 8), int(self.image_size[1] / 8)), tau=self.tau)
        self.norfeaturetran = NoRobustTranformer(Channel = 512)
        self.aux = nn.Sequential(nn.Linear(512, num_classes))

        self.norfeaturecom = vgg_conv_block_channel_reg([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, y=None, is_eval=False, is_train=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []
        extra_output = []

        x = self.block0(x)
        x = self.pool0(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)

        out1 = x
        feature4 = F.avg_pool2d(out1, 4).view(out1.size(0), -1)

        r_feat, nr_feat, mask = self.separation(x, is_eval=is_eval)

        feature4_rf = F.avg_pool2d(r_feat, 4).view(r_feat.size(0), -1)
        feature4_nrf = F.avg_pool2d(nr_feat, 4).view(nr_feat.size(0), -1)

        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)
        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        rec_feat = self.norfeaturetran(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        x = r_feat + rec_feat

        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)

        sepnr_feat = nr_feat
        for layer in self.norfeaturecom:
            sepnr_feat, layer5_out = layer(sepnr_feat, y, is_train=is_train)
            extra_output.append(layer5_out)
        vgg16_features = self.max_pooling(sepnr_feat)

        x = x + vgg16_features

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, r_outputs, nr_outputs, extra_output, rec_outputs, (feature4, feature4_rf, feature4_nrf)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    features = []
    in_channels = 3
    for v in cfg:
        if v == 'fsr':
            continue
        elif v == 'M':
            features.append(nn.Sequential(*layers))
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            features.append(nn.Sequential(*layers))
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return features


cfgs = {
    'vgg16_FSR': ['fsr', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, tau=0.1, num_classes=10, image_size=(32, 32), **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes, image_size=image_size, init_weights=True, tau=tau, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16_FTA2C(pretrained=False, progress=True, tau=0.1, num_classes=10, image_size=(32, 32), **kwargs):
    return _vgg('vgg16_FSR', 'vgg16_FSR', True, pretrained, progress, tau=tau, num_classes=num_classes, image_size=image_size, **kwargs)

