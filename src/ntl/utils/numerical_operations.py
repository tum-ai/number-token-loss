import torch


def signed_log(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_signed_log(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)