import os
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(
        model=None, 
        optimizer=None, 
        epoch=None, 
        best_AP=None, 
        best_epoch=None, 
        AP_stats=None,
        ):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 
            'model_state': model_state, 
            'optimizer_state': optim_state,
            'best_AP': best_AP,
            'best_epoch': best_epoch,
            'AP_stats': AP_stats,
            }

def save_checkpoint(state, filename, logger):
    logger.info("==> Saving to checkpoint '{}'".format(filename))
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)

        best_AP = checkpoint.get('best_AP', 0.)
        best_epoch = checkpoint.get('best_epoch', 1)
        AP_stats = checkpoint.get('AP_stats', {})
 
        if (model is not None) and (checkpoint['model_state'] is not None):
            filtered_state = {}
            for key, value in checkpoint['model_state'].items():
                if key.startswith('base_encoder.baseline'):
                    new_key = key[len('base_encoder.baseline.'):]
                    filtered_state[new_key] = value
            if len(filtered_state) == 0: filtered_state = checkpoint['model_state']

            # model.load_state_dict(checkpoint['model_state'], strict=False)      
            model.load_state_dict(filtered_state, strict=False)      
        if (optimizer is not None) and ('optimizer_state' in checkpoint):
            if checkpoint['optimizer_state'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_AP, best_epoch, AP_stats