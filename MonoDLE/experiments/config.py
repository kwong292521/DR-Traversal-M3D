import yaml


cfg_file = './experiments/fully_supervised.yaml'

# cfg_file = './experiments/sparse_supervised_pretrain.yaml'

# cfg_file = './experiments/sparse_supervised_finetune@10%.yaml'


def get_cfg():
    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

    scenes_count = {
        'n_train_scenes': 3712,
        'n_trainval_scenes': 7481,
        'n_val_scenes': 3769,
        'n_test_scenes': 7518,
    }
    cfg['dataset']['augment']['scenes_count'] = scenes_count

    return cfg

cfg = get_cfg()