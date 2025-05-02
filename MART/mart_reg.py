import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_pred(out, labels):
    out1 = out.sort(dim=-1, descending=True)
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)  #分类等于标签，返回second_pred，否则返回pred

    return adv_label


def mart_loss(model,
              epoch,
              device,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf',
              cr_beta=1.0):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_output, adv_r_outputs, adv_nr_outputs, adv_sup_outputs,\
                    adv_rec_outputs, adv_featrue = model(x_adv, y, is_train=False)
                loss_ce = F.cross_entropy(adv_output, y)
                extra_ce = 0.
                for output in adv_sup_outputs:
                    extra_ce += F.cross_entropy(output, y)
                extra_ce /= len(adv_sup_outputs)
                loss_ce += cr_beta * extra_ce
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # logits, extra_outputs_nat = model(x_natural, y, _eval=False)
    logits, nat_r_outputs, nat_nr_outputs, nat_sup_outputs, \
        nat_rec_outputs, nat_featrue = model(x_natural, y, is_train=False)
    cle_out_f, cle_r_f, cle_nf_f = nat_featrue
    # logits_adv, extra_outputs_robust = model(x_adv, y, _eval=False)
    adv_outputs, adv_r_outputs, adv_nr_outputs, adv_sup_outputs, \
        adv_rec_outputs, adv_featrue = model(x_adv, y, is_train=False)
    adv_out_f, adv_r_f, adv_nf_f = adv_featrue
    #########################adv loss begin###########################
    adv_probs = F.softmax(adv_outputs, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(adv_outputs, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    adv_err_labels = get_pred(adv_outputs, y)  # 为啥用第二大的结果作为非鲁棒的预测

    extra_adv_probs_r = []
    extra_loss_adv = torch.tensor(0.).to(device)
    for output in adv_r_outputs:
        adv_probs_extra_r = F.softmax(output, dim=1)
        # print(adv_probs_extra)
        extra_adv_probs_r.append(adv_probs_extra_r)
        tmp1 = torch.argsort(adv_probs_extra_r, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        extra_loss_adv += (F.cross_entropy(output, y) + F.nll_loss(torch.log(1.0001 - adv_probs_extra_r + 1e-12), new_y))
    extra_loss_adv /= len(nat_r_outputs)
    loss_adv += extra_loss_adv

    extra_adv_probs_nr = []
    extra_loss_adv = torch.tensor(0.).to(device)
    for output in adv_nr_outputs:
        adv_probs_extra_nr = F.softmax(output, dim=1)
        # print(adv_probs_extra)
        extra_adv_probs_nr.append(adv_probs_extra_nr)
        tmp1 = torch.argsort(adv_probs_extra_nr, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        extra_loss_adv += (F.cross_entropy(output, adv_err_labels) + F.nll_loss(torch.log(1.0001 - adv_probs_extra_nr + 1e-12), new_y))
    extra_loss_adv /= len(nat_nr_outputs)
    loss_adv += extra_loss_adv

    extra_adv_probs_rec = []
    extra_loss_adv = torch.tensor(0.).to(device)
    for output in adv_rec_outputs:
        adv_probs_extra_rec = F.softmax(output, dim=1)
        # print(adv_probs_extra)
        extra_adv_probs_rec.append(adv_probs_extra_rec)
        tmp1 = torch.argsort(adv_probs_extra_rec, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        extra_loss_adv += (F.cross_entropy(output, y) + F.nll_loss(torch.log(1.0001 - adv_probs_extra_rec + 1e-12), new_y))
    extra_loss_adv /= len(nat_nr_outputs)
    loss_adv += extra_loss_adv

    extra_adv_probs = []
    extra_loss_adv = torch.tensor(0.).to(device)
    for output in adv_sup_outputs:
        adv_probs_extra = F.softmax(output, dim=1)
        # print(adv_probs_extra)
        extra_adv_probs.append(adv_probs_extra)
        tmp1 = torch.argsort(adv_probs_extra, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        extra_loss_adv += (F.cross_entropy(output, y) + F.nll_loss(torch.log(1.0001 - adv_probs_extra + 1e-12), new_y))
    extra_loss_adv /= len(adv_sup_outputs)
    loss_adv += extra_loss_adv
    #########################adv loss end###########################

    #########################nature loss begin###########################
    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    extra_loss_robust = torch.tensor(0.).to(device)
    for idx, output in enumerate(nat_r_outputs):
        nat_probs_extra_r = F.softmax(output, dim=1)
        # true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        extra_loss_robust += (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(extra_adv_probs_r[idx] + 1e-12), nat_probs_extra_r), dim=1) * (1.0000001 - true_probs))
    extra_loss_robust /= len(nat_r_outputs)
    loss_robust += extra_loss_robust

    extra_loss_robust = torch.tensor(0.).to(device)
    for idx, output in enumerate(nat_nr_outputs):
        nat_probs_extra_nr = F.softmax(output, dim=1)
        # true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        extra_loss_robust += (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(extra_adv_probs_nr[idx] + 1e-12), nat_probs_extra_nr), dim=1) * (1.0000001 - true_probs))
    extra_loss_robust /= len(nat_nr_outputs)
    loss_robust += extra_loss_robust

    extra_loss_robust = torch.tensor(0.).to(device)
    for idx, output in enumerate(nat_rec_outputs):
        nat_probs_extra_rec = F.softmax(output, dim=1)
        # true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        extra_loss_robust += (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(extra_adv_probs_rec[idx] + 1e-12), nat_probs_extra_rec), dim=1) * (1.0000001 - true_probs))
    extra_loss_robust /= len(nat_rec_outputs)
    loss_robust += extra_loss_robust

    extra_loss_robust = torch.tensor(0.).to(device)
    for idx, output in enumerate(nat_sup_outputs):
        nat_probs_extra = F.softmax(output, dim=1)
        # true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        extra_loss_robust += (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(extra_adv_probs[idx] + 1e-12), nat_probs_extra), dim=1) * (1.0000001 - true_probs))
    extra_loss_robust /= len(nat_sup_outputs)
    loss_robust += extra_loss_robust
    #########################nature loss end###########################

    loss = loss_adv + beta * loss_robust

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
