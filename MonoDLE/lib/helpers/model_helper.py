from experiments.config import cfg

from lib.models.centernet3d import CenterNet3D

def build_model():
    assert cfg['model']['type'] in [
        'centernet3d@rgb',                  # rgb baseline
    ]
    if cfg['model']['type'] in ['centernet3d@rgb', 'centernet3d@depth']:
        return build_baseline()

def build_baseline():
    modality = cfg['model']['type'].split('@')[1]
    IN_pretrained = cfg['model'][f'{modality}_IN_pretrained']

    return CenterNet3D(backbone=cfg['model']['backbone'], neck=cfg['model']['neck'], num_class=cfg['model']['num_class'], IN_pretrained=IN_pretrained)
