import torch
import numpy as np
from torch.utils.data import DataLoader
from experiments.config import cfg

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset, KITTI_PostProcesser


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(workers=4, prefetch=4, ddp=False):
    train_set = KITTI_Dataset(split='train')
    val_set = KITTI_Dataset(split='val')

    # prepare dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg['dataset']['batch_size'],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=False,            
        pin_memory=True,
        drop_last=True,           # dont use last step
        prefetch_factor=prefetch,
        )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg['dataset']['batch_size'],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=prefetch,
        )

    if cfg['trainer'].get('tv4t_mode', False):
        test_set = KITTI_Dataset(split='test')
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=cfg['dataset']['batch_size'],
            num_workers=workers,
            worker_init_fn=my_worker_init_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=prefetch,
        )
    else:
        test_loader = None


    kitti_post_processer = KITTI_PostProcesser()

    return train_loader, val_loader, test_loader, kitti_post_processer

