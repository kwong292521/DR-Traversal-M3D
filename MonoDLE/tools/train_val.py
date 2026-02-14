import warnings
warnings.simplefilter("error", FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
parser = argparse.ArgumentParser(description='Object-Scene-Camera Decomposition and Recomposition for Data-Efficient Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('--gpu', dest='gpu', default='0', help='gpu id in nvidia-smi format')
args = parser.parse_args()

import os
# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# set CPU
set_cpu = True
import psutil
pid = os.getpid()
if set_cpu:
    cpu2use = 8
    cpu_scan_time = 5
    cpu_list = [
        
        ]
    if len(cpu_list) == 0:
        cpu_usage = psutil.cpu_percent(interval=cpu_scan_time, percpu=True)
        sorted_usage = sorted(enumerate(cpu_usage), key=lambda x: x[1])
        cpu_list = [index for index, value in sorted_usage[:cpu2use]]
    os.sched_setaffinity(pid, cpu_list)
affinity = psutil.Process(pid).cpu_affinity()
cpu_info = f'Process {pid} is running on CPUs: {affinity}'

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib/datasets/kitti'))   # for instance databse

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.utils_helper import create_logger_dist
from lib.helpers.utils_helper import set_random_seed
from experiments.config import cfg
import yaml

import torch


n_workers = 4       
n_prefetch = 4

def main(rank, world_size):
    set_random_seed(cfg.get('random_seed', 444))

    train_loader, val_loader, test_loader, kitti_post_processer = build_dataloader(workers=n_workers, prefetch=n_prefetch)
    model = build_model().to(rank)

    logger = create_logger_dist(os.path.join(cfg['trainer']['log_dir'], 'train', cfg['trainer']['log_fname']), rank)
    logger.info(cpu_info)
    logger.info('Configs: \n' + yaml.dump(cfg, default_flow_style=False))

    #  build optimizer
    optimizer = build_optimizer(model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(optimizer=optimizer, last_epoch=-1)

    logger.info('###################  Training  ##################')
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      kitti_post_processer=kitti_post_processer,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      rank=rank
                    )
    trainer.train()

    #========= NOTE: DEBUG
    # trainer.eval_and_get_AP(1)    # NOTE: for val set debug
    # trainer.eval_one_epoch(epoch=0, _data_loader=test_loader, _split='test')    # NOTE: for test set debug


if __name__ == '__main__':
    main(rank=0, world_size=1)
