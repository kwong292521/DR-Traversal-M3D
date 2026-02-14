import torch
import numpy as np
import logging
import random

class NumbaInfoFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == "numba" and record.levelno == logging.INFO)

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    logger = logging.getLogger(__name__)
    
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    
    return logger


def create_logger_dist(log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)

    logger = logging.getLogger(__name__)

    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
