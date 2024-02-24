import torch
import numpy as np


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=x.size(0))
        lam = np.stack((lam, 1 - lam), 0).max(0)
        lam = torch.tensor(lam).float().cuda()
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = (
        lam.view(lam.size(0), 1, 1, 1) * x
        + (1 - lam).view(lam.size(0), 1, 1, 1) * x[index, :]
    )
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam.view(lam.size(0), 1) * criterion(pred, y_a, reduction="none")
    loss_b = (1 - lam).view(lam.size(0), 1) * criterion(pred, y_b, reduction="none")
    loss = loss_a + loss_b
    return torch.mean(loss)
