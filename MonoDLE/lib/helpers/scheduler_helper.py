import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from torch.optim.optimizer import Optimizer
import math
from experiments.config import cfg
from typing import Tuple, List



def build_lr_scheduler(**kwargs):
    _type = cfg['lr_scheduler'].get('type', 'base')
    assert _type in ['base', 'cyclic']
    if _type == 'base':
        return build_lr_scheduler_base(**kwargs)
    elif _type == 'cyclic':
        return build_lr_scheduler_cyclic(**kwargs)


def build_lr_scheduler_cyclic(optimizer, last_epoch=None):
    lr_scheduler = CyclicScheduler(
            optimizer,
            total_epochs=cfg['trainer']['max_epoch'],
            target_lr_ratio=(10, 1e-4),
            target_momentum_ratio=(0.85 / 0.95, 1.0),
            period_up=0.4
        )
    warmup_lr_scheduler = None

    return lr_scheduler, warmup_lr_scheduler



def build_lr_scheduler_base(optimizer, last_epoch=-1):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['lr_scheduler']['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['lr_scheduler']['decay_rate']
        return cur_decay

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    warmup_lr_scheduler = None
    if cfg['lr_scheduler']['warmup']:
        warmup_lr_scheduler = CosineWarmupLR(optimizer, num_epoch=5, init_lr=0.00001)
    return lr_scheduler, warmup_lr_scheduler



def build_bnm_scheduler(cfg, model, last_epoch):
    if not cfg['lr_scheduler']['enabled']:
        return None

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['lr_scheduler']['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['lr_scheduler']['decay_rate']
        return max(cfg['lr_scheduler']['momentum']*cur_decay, cfg['lr_scheduler']['clip'])

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) *
                (1 - math.cos(math.pi * self.last_epoch / self.num_epoch)) / 2
                for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.init_lr + (base_lr - self.init_lr) * self.last_epoch / self.num_epoch
                for base_lr in self.base_lrs]



class CyclicScheduler(lr_sched._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 total_epochs: int,      
                 target_lr_ratio: Tuple[int, int] = (10, 1e-4),     
                 target_momentum_ratio: Tuple[float, float] = (0.85 / 0.95, 1.),    
                 period_up: float = 0.4):   
        
        assert (optimizer.__class__.__name__ == 'AdamW'), \
            "Currently, this scheduler only supports 'AdamW' optimizer."
        
        self.total_epochs = total_epochs
        
        self.target_lr_ratio = target_lr_ratio
        self.target_momentum_ratio = target_momentum_ratio
        
        self.period_up = period_up
        self.epochs_up = int(self.total_epochs * self.period_up)
        
        for group in optimizer.param_groups:
            group.setdefault('initial_momentum', group['betas'][0])
        self.base_momentum = [
            group['initial_momentum']
            for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=-1)
        
    
    def get_lr(self) -> List[float]:
        
        self.set_momentum()
        
        # Phase 1: LR Step-Up
        if (self.last_epoch < self.epochs_up):
            return [self._annealing_func(base_lr * 1.0,
                                         base_lr * self.target_lr_ratio[0],
                                         (self.last_epoch+1 - 0) / (self.epochs_up - 0))    
                    for base_lr in self.base_lrs]
        
        # Phase 2: LR Step-Down
        else:
            return [self._annealing_func(base_lr * self.target_lr_ratio[0],
                                         base_lr * self.target_lr_ratio[1],
                                         (self.last_epoch+1 - self.epochs_up) / (self.total_epochs - self.epochs_up))
                    for base_lr in self.base_lrs]
        

    def set_momentum(self):
        # Phase 1: Beta Step-Down
        if (self.last_epoch < self.epochs_up):
            regular_momentums = [self._annealing_func(base_momentum * 1.0,
                                                      base_momentum * self.target_momentum_ratio[0],
                                                      (self.last_epoch+1 - 0) / (self.epochs_up - 0))
                                 for base_momentum in self.base_momentum]
        
        # Phase 2: Beta Step-Up
        else:
            regular_momentums = [self._annealing_func(base_momentum * self.target_momentum_ratio[0],
                                                      base_momentum * self.target_momentum_ratio[1],
                                                      (self.last_epoch+1 - self.epochs_up) / (self.total_epochs - self.epochs_up))
                                 for base_momentum in self.base_momentum]
        
        for param_group, mom in zip(self.optimizer.param_groups, regular_momentums):
            param_group['betas'] = (mom, param_group['betas'][1])
            

    def _annealing_func(self, start: float, end: float, factor: float, weight: float = 1.) -> float:
        cos_out = math.cos(math.pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out
