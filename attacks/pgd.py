'''
Codes modified from original codes of:
CAS : https://github.com/bymavis/CAS_ICLR2021
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)  # 将x张量中的每个元素限制在最大值 max_x 和最小值 min_x 之间

    else:
        raise NotImplementedError

    return x


def adv_get_pred(out, labels):
    out1 = out.sort(dim=-1, descending=True)
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)  #分类等于标签，返回second_pred，否则返回pred

    return adv_label
class PGD():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model

        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type

    def perturb(self, original_images, labels, random_start=False, beta=2):
        device = original_images.get_device()
        print(device)
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.to(device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs, outputs_r, outputs_nr, outputs_sep, outputs_rec,_ = self.model(x, labels, is_eval=True, is_train=True)

                adv_labels = adv_get_pred(outputs, labels)

                cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                # print('cls_loss  is {}'.format(cls_loss))
                loss = cls_loss

                channel_reg_loss = 0.
                for extra_output in outputs_sep:
                    channel_reg_loss += nn.CrossEntropyLoss()(extra_output, labels)

                if len(outputs_sep) > 0:
                    channel_reg_loss /= len(outputs_sep)

                loss += beta * channel_reg_loss

                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                                            only_inputs=True)[0]
                x.data += self.alpha * torch.sign(grads.data)
                x = project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)

        return x
