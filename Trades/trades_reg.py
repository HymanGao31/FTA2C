import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()



def get_pred(out, labels):
    out1 = out.sort(dim=-1, descending=True)
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)  #分类等于标签，返回second_pred，否则返回pred

    return adv_label

def get_other_loss(device, outputs, nat_err_labels, y):

    nat_r_outputs, nat_nr_outputs, nat_rec_outputs = outputs
    r_loss = torch.tensor(0.).to(device)
    if not len(nat_r_outputs) == 0:
        for r_out in nat_r_outputs:
            r_loss += F.cross_entropy(r_out, y)
        r_loss /= len(nat_r_outputs)

    nr_loss = torch.tensor(0.).to(device)
    if not len(nat_nr_outputs) == 0:
        for nr_out in nat_nr_outputs:
            nr_loss += F.cross_entropy(nr_out, nat_err_labels)
        nr_loss /= len(nat_nr_outputs)

    sep_loss = r_loss + nr_loss

    rec_loss = torch.tensor(0.).to(device)
    if not len(nat_rec_outputs) == 0:
        for rec_out in nat_rec_outputs:
            rec_loss += F.cross_entropy(rec_out, y)
        rec_loss /= len(nat_rec_outputs)

    loss_other = sep_loss + rec_loss

    return loss_other

def get_other_loss_adv(device, outputs_adv,out_nat,batch_size):

    nat_r_outputs, nat_nr_outputs, nat_rec_outputs = out_nat
    adv_r_outputs, adv_nr_outputs, adv_rec_outputs = outputs_adv

    criterion_kl = nn.KLDivLoss(reduction='sum')


    extra_loss_robust1 = torch.tensor(0.).to(device)
    if not len(adv_r_outputs) == 0:
        for i in range(len(adv_r_outputs)):
            extra_loss_robust1 += (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_r_outputs[i], dim=1),
                                F.softmax(nat_r_outputs[i], dim=1))
    extra_loss_robust1 /= len(adv_r_outputs)

    extra_loss_robust2 = torch.tensor(0.).to(device)
    if not len(adv_nr_outputs) == 0:
        for i in range(len(adv_nr_outputs)):
            extra_loss_robust2 += (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_nr_outputs[i], dim=1),
                                F.softmax(nat_nr_outputs[i], dim=1))
    extra_loss_robust2 /= len(adv_nr_outputs)

    extra_loss_robust3 = torch.tensor(0.).to(device)
    if not len(adv_rec_outputs) == 0:
        for i in range(len(adv_rec_outputs)):
            extra_loss_robust3 += (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_rec_outputs[i], dim=1),
                                F.softmax(nat_rec_outputs[i], dim=1))
    extra_loss_robust3 /= len(adv_rec_outputs)


    loss_other = extra_loss_robust1 + extra_loss_robust2 + extra_loss_robust3

    return loss_other
def trades_loss(model,
                epoch,
                device,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                cr_beta=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                #output_adv, extraoutput_adv = model(x_adv, y, _eval=True)
                adv_outputs, adv_r_outputs, adv_nr_outputs, adv_sup_outputs,\
                    adv_rec_outputs, adv_featrue = model(x_adv, y, is_train=False)
                #output_nat, extraoutput_nat = model(x_natural, y, _eval=True)
                nat_outputs, nat_r_outputs, nat_nr_outputs, nat_sup_outputs,\
                    nat_rec_outputs, nat_featrue = model(x_natural, y, is_train=False)

                loss_kl = criterion_kl(F.log_softmax(adv_outputs, dim=1),
                                       F.softmax(nat_outputs, dim=1))
                channel_reg_loss = 0.
                for i in range(len(adv_sup_outputs)):
                    channel_reg_loss += criterion_kl(F.log_softmax(adv_sup_outputs[i], dim=1),
                                       F.softmax(nat_sup_outputs[i], dim=1))
                channel_reg_loss /= len(adv_sup_outputs)
                loss_kl += cr_beta * channel_reg_loss
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    ###########################nature logits loss begin ###################################

    #logits, extra_output_nat = model(x_natural, y, _eval=False)
    logits, nat_r_outputs, nat_nr_outputs, nat_sup_outputs, \
        nat_rec_outputs, nat_featrue = model(x_natural, y, is_train=False)
    cle_out_f, cle_r_f, cle_nf_f = nat_featrue
    loss_natural = F.cross_entropy(logits, y)

    nat_err_labels = get_pred(logits, y)  # 为啥用第二大的结果作为非鲁棒的预测

    nat_out_list = (nat_r_outputs, nat_nr_outputs, nat_rec_outputs)

    loss_natural += get_other_loss(device,nat_out_list,nat_err_labels,y)

    extra_loss_natural = 0.
    for output in nat_sup_outputs:
        extra_loss_natural += F.cross_entropy(output, y)
    extra_loss_natural /= len(nat_sup_outputs)
    loss_natural += cr_beta * extra_loss_natural

    # print(' loss_natural: %.3f' % loss_natural)

    ###########################nature logits loss end###################################

    ###########################adv logits loss begin###################################

    #output_adv, extra_output_adv = model(x_adv, y, _eval=False)
    adv_outputs, adv_r_outputs, adv_nr_outputs, adv_sup_outputs, \
        adv_rec_outputs, adv_featrue = model(x_adv, y, is_train=False)
    adv_out_f, adv_r_f, adv_nf_f = adv_featrue

    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_outputs, dim=1),
                                                    F.softmax(logits, dim=1))

    adv_err_labels = get_pred(adv_outputs, y)  # 为啥用第二大的结果作为非鲁棒的预测
    adv_out_list = (adv_r_outputs, adv_nr_outputs, adv_rec_outputs)
    loss_robust += get_other_loss_adv(device, adv_out_list, nat_out_list,batch_size)
    # print(' loss_robust1: %.3f' % loss_robust)
    extra_loss_robust = 0.
    for i in range(len(adv_sup_outputs)):
        extra_loss_robust += (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_sup_outputs[i], dim=1),
                                                        F.softmax(nat_sup_outputs[i], dim=1))
    extra_loss_robust /= len(adv_sup_outputs)
    loss_robust += cr_beta * extra_loss_robust
    # print(' loss_robust2: %.3f' % loss_robust)
    loss = loss_natural + beta * loss_robust

    if epoch > 10:
        ####################
        normed_clean = F.normalize(cle_r_f, dim=-1)
        matrix_clean = torch.mm(normed_clean, normed_clean.t())

        normed_feature = F.normalize(adv_r_f, dim = -1)
        matrix_adv = torch.mm(normed_feature, normed_feature.t())
        diff = torch.exp(torch.abs(matrix_adv - matrix_clean))
        loss_lfrc = torch.mean(diff)

        loss += loss_lfrc
        # print(' loss: %.3f\n' % loss)
        ################

    return loss
