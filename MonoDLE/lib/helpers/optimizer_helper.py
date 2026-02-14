import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from experiments.config import cfg

def build_optimizer(model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{'params': biases, 'weight_decay': 0},
                  {'params': weights, 'weight_decay': cfg['optimizer']['weight_decay']}]


    if cfg['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg['optimizer']['lr'], momentum=0.9)
    elif cfg['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg['optimizer']['lr'])
    elif cfg['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(parameters, lr=cfg['optimizer']['lr'], betas=(cfg['optimizer']['beta_1'], cfg['optimizer']['beta_2']))
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg['optimizer']['type'])

    return optimizer


