import os
import tqdm
import json
import copy
import time
import cv2

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, extract_dets_from_outputs_oracle
from lib.helpers.decode_helper import decode_detections
from lib.losses.centernet_loss import compute_loss

from experiments.config import cfg

from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import kitti_eval
from lib.datasets.kitti.kitti_utils import Calibration

class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 val_loader,
                 test_loader,
                 kitti_post_processer,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 rank):
        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.kitti_post_processer = kitti_post_processer
        
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        if rank is not None:
            # for ddp
            self.rank = rank
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.rank = 0
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if train_loader is not None:
            self.class_name = train_loader.dataset.class_name
        elif val_loader is not None:
            self.class_name = val_loader.dataset.class_name
        elif test_loader is not None:
            self.class_name = test_loader.dataset.class_name


        if (self.test_loader is not None) and cfg['trainer'].get('tv4t_mode', False):
            self.test_epochs_list = [i for i in range(cfg['lr_scheduler']['decay_list'][-1], cfg['trainer']['max_epoch']+1, cfg['trainer']['save_frequency'])]
            
            self.val_epochs_list = [i for i in range(cfg['lr_scheduler']['decay_list'][0], cfg['trainer']['max_epoch']+1, 5)]
            self.val_epochs_list += [i for i in range(10, cfg['lr_scheduler']['decay_list'][0]-9, 10)]
            if cfg['trainer']['max_epoch'] not in self.val_epochs_list: self.val_epochs_list.append(cfg['trainer']['max_epoch'])
            
            self.val_epochs_list += [1, self.epoch+1]    # for debug
            self.test_epochs_list += [1, self.epoch+1]    # for debug
            
            self.val_epochs_list = sorted(list(np.unique(self.val_epochs_list)))
            self.test_epochs_list = sorted(list(np.unique(self.test_epochs_list)))

            self.logger.info(f'VAL EPOCHS: {self.val_epochs_list}')
            self.logger.info(f'TEST EPOCHS: {self.test_epochs_list}')

        else:
            self.test_epochs_list = None

            self.val_epochs_list = [i for i in range(cfg['lr_scheduler']['decay_list'][0], cfg['trainer']['max_epoch']+1, cfg['trainer']['save_frequency'])]
            self.val_epochs_list += [i for i in range(10, cfg['lr_scheduler']['decay_list'][0]-9, 10)]
            if cfg['trainer']['max_epoch'] not in self.val_epochs_list: self.val_epochs_list.append(cfg['trainer']['max_epoch'])

            self.val_epochs_list += [1, self.epoch+1]    # for debug

            self.val_epochs_list = sorted(list(np.unique(self.val_epochs_list)))

            self.logger.info(f'VAL EPOCHS: {self.val_epochs_list}')


        # for AP stats glances and resume
        self.AP_stats = self._AP_stats_init()
        self.best_AP = 0.
        self.best_epoch = 1
        os.makedirs('./outputs/val/{}'.format(cfg['trainer']['train_idx']), exist_ok=True)

        self._load_ckpts()


    def _load_ckpts(self):
        if cfg['trainer'].get('pretrain_model'):
            assert os.path.exists(cfg['trainer']['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['trainer']['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg['trainer'].get('resume_model'):
            assert os.path.exists(cfg['trainer']['resume_model'])
            resume_info = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['trainer']['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.epoch, self.best_AP, self.best_epoch, self.AP_stats  = resume_info
            self.epoch = cfg['trainer'].get('resume_start_epoch', self.epoch)
            if self.epoch == 0:
                self.best_AP, self.best_epoch = 0., 1

            _lrs_type = cfg['lr_scheduler'].get('type', 'base')
            if _lrs_type == 'base':
                self.lr_scheduler.last_epoch = self.epoch - 1 if self.epoch < 5 else self.epoch - 5
            elif _lrs_type == 'cyclic':
                self.lr_scheduler.last_epoch = self.epoch
            
            if self.train_loader is not None: self.train_loader.dataset.cur_epoch = self.epoch

            if not self.AP_stats:
                self.AP_stats = self._AP_stats_init()

    def _AP_stats_init(self):
        AP_stats = {}
        # for epoch in self.val_epochs_list:
        for epoch in range(cfg['trainer']['max_epoch']+1):
            AP_stats['epoch_%d'%epoch] = {}
            for obj_cls in ['Car', 'Pedestrian', 'Cyclist', 'Overall']:
                AP_stats['epoch_%d'%epoch][obj_cls] = {}
                for target in ['3D', 'BEV', '2D', 'AOS']:
                    AP_stats['epoch_%d'%epoch][obj_cls][target] = {}
        return AP_stats
    
    def _AP_stats_update(self, AP_stats, ret_dict):
        for obj_cls in ['Car', 'Pedestrian', 'Cyclist', 'Overall']:
            for target in ['3D', 'BEV', '2D', 'AOS']:
                for level in ['easy', 'moderate', 'hard']:
                    if obj_cls != 'Overall':
                        ret_key = 'KITTI/{}_{}_AP40_{}_strict'.format(obj_cls, target, level)
                    elif obj_cls == 'Overall':
                        ret_key = 'KITTI/{}_{}_AP40_{}'.format(obj_cls, target, level)
                    if type(ret_dict[ret_key]) == np.float64:
                        AP_stats['epoch_%d'%self.epoch][obj_cls][target][level] = ret_dict[ret_key]
                    elif type(ret_dict[ret_key]) == np.ndarray:
                        AP_stats['epoch_%d'%self.epoch][obj_cls][target][level] = float(ret_dict[ret_key][0])
        
        return AP_stats


    def train(self):
        train_stats = {}
        start_epoch = self.epoch

        # GT label init
        gt_path = self.train_loader.dataset.aug_scene_loader.data_dirs['trainval']['label_dir']
        gt_split_file = os.path.join(self.train_loader.dataset.aug_scene_loader.data_dirs['trainval']['split_dir'], 'val.txt')

        with open(gt_split_file, 'r') as f:
            lines = f.readlines()
        val_image_ids = [int(line) for line in lines]
        gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

        os.makedirs(os.path.join(cfg['trainer']['log_dir']+'/checkpoints', cfg['trainer']['train_idx']), exist_ok=True)

        for epoch in range(start_epoch, cfg['trainer']['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:      
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            ei_loss = self.train_one_epoch()
            self.epoch += 1
            self.train_loader.dataset.cur_epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            log_str = '====EPOCH[%04d/%04d] TRAIN====' % (self.epoch, cfg['trainer']['max_epoch'])
            log_str += ' %s:%.4f,' %('total_loss', ei_loss['total_loss'])
            for key in sorted(ei_loss.keys()):
                if key != 'total_loss':
                    log_str += ' %s:%.4f,' %(key, ei_loss[key])
            self.logger.info(log_str)

            ei_loss_tmp = {}
            for key in ei_loss.keys():
                if key != 'total_loss':
                    ei_loss_tmp[key] = float(ei_loss[key])
            ei_loss_tmp['total_loss'] = float(ei_loss['total_loss'])
            train_stats['epoch_'+str(self.epoch)] = ei_loss_tmp
            with open(os.path.join(cfg['trainer']['log_dir'], 'stats', cfg['trainer']['train_states_fname']), 'w') as f:
                json.dump(train_stats, f)

            # eval one epoch @ val set
            if self.epoch in self.val_epochs_list:
                self.logger.info('------ EVAL EPOCH %03d @ VAL SET ------' % (self.epoch))
                self.eval_one_epoch(self.epoch, _data_loader=self.val_loader, _split='val')

                det_path = './outputs/val/{}/epoch_{}'.format(cfg['trainer']['train_idx'], self.epoch)

                dt_annos = kitti.get_label_annos(det_path)
                result, ret_dict = kitti_eval(
                                        gt_annos, 
                                        dt_annos, 
                                        ['Car', 'Pedestrian', 'Cyclist'], 
                                        eval_types=['bbox', 'bev', '3d', 'aos'])
                
                self.AP_stats = self._AP_stats_update(self.AP_stats, ret_dict)

                # car_ap = self.AP_stats['epoch_%d'%self.epoch]['Car']['3D']['moderate']
                car_ap = self.AP_stats['epoch_%d'%self.epoch]['Car']['BEV']['moderate']
                if car_ap > self.best_AP:
                    self.best_AP = car_ap
                    self.best_epoch = self.epoch

                    ckpt_name = os.path.join(cfg['trainer']['log_dir']+'/checkpoints', cfg['trainer']['train_idx'], 'checkpoint_best')
                    save_checkpoint(
                        get_checkpoint_state(self.model, self.optimizer, self.epoch, 
                                                self.best_AP, self.best_epoch, self.AP_stats), 
                        ckpt_name, 
                        self.logger)
                log_str = '====== Current AP @ Epoch {} is {:.4f}'.format(self.epoch, car_ap)
                self.logger.info(log_str)
                log_str = '====== Best AP @ Epoch {} is {:.4f}'.format(self.best_epoch, self.best_AP)
                self.logger.info(log_str)

            ckpt_name = os.path.join(cfg['trainer']['log_dir']+'/checkpoints', cfg['trainer']['train_idx'], 'checkpoint_latest')
            save_checkpoint(
                get_checkpoint_state(self.model, self.optimizer, self.epoch, 
                                        self.best_AP, self.best_epoch, self.AP_stats), 
                ckpt_name, 
                self.logger)

            # eval one epoch @ test set 
            if (self.test_epochs_list is not None) and (self.epoch in self.test_epochs_list):
                self.logger.info('------ EVAL EPOCH %03d @ TEST SET ------' % (self.epoch))
                self.eval_one_epoch(self.epoch, _data_loader=self.test_loader, _split='test')

                # save pure model
                ckpt_name = os.path.join(cfg['trainer']['log_dir']+'/checkpoints', cfg['trainer']['train_idx'], f'checkpoint_epoch_{self.epoch}')
                save_checkpoint(
                    get_checkpoint_state(model=self.model, epoch=self.epoch),
                    ckpt_name,
                    self.logger)

            with open('./outputs/val/{}/stats.json'.format(cfg['trainer']['train_idx']), 'w') as f:
                json.dump(self.AP_stats, f)

        return None


    def train_one_epoch(self):
        self.model.train()
        disp_dict = {}
        stat_dict = {}
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # NOTE: data post-process
            inputs, targets, info = self.kitti_post_processer(batch_data, data_aug=True)

            outputs = self.model(inputs=inputs, targets=targets, mode='train')                 

            total_loss, loss_terms = compute_loss(outputs, targets, self.epoch)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            loss_terms['total_loss'] = total_loss.item()
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key]
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key]

            # display statistics in terminal
            if trained_batch % cfg['trainer']['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / cfg['trainer']['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
        
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch

        return stat_dict


    def eval_one_epoch(self, epoch, _data_loader, _split='val'):
        assert _split in ['val', 'test']

        _std_method = 'uni_K' if cfg['dataset']['augment'].get('use_uni_intrinsic') else 'shift'

        self.model.eval()

        results = {}

        progress_bar = tqdm.tqdm(total=len(_data_loader), leave=True, desc=f'Evaluation Progress @ {_split}')
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(_data_loader):
                # NOTE: post-processing
                inputs, targets, info = self.kitti_post_processer(batch_data, data_aug=False)

                # get corresponding calibs & stdlize
                calibs = [] 
                affine_inv_Ts = []
                for idx, offset, img_size in zip(info['img_id'], info['to_std_offset'], info['img_size']):
                    
                    if _split == 'test':
                        _calib_file = os.path.join(_data_loader.dataset.aug_scene_loader.data_dirs['test']['calib_dir'], '%06d.txt' % idx)
                        assert os.path.exists(_calib_file)
                        calib = Calibration(_calib_file)
                    else:
                        calib = _data_loader.dataset.aug_scene_loader.get_calibration(idx)

                    if _std_method == 'uni_K':
                        w, h = img_size
                        calib, affine_T = _data_loader.dataset.aug_scene_loader.uni_intrinsic(calib, h=h, w=w, ret_affine=True)
                        # get inv affine (dst to src trans)
                        affine_T = np.concatenate([affine_T, np.array([[0, 0, 1]])], axis=0)
                        _inv_T = np.linalg.inv(affine_T)
                        affine_inv_Ts.append(_inv_T[:2])
                    elif _std_method == 'shift':
                        calib.P2[0, 2] += offset[0]
                        calib.P2[1, 2] += offset[1]
                        calib.cu += offset[0]
                        calib.cv += offset[1]
                    calibs.append(calib)
    
                # inference forward
                outputs = self.model(inputs=inputs, targets=targets, mode=_split)

                cls_mean_size = _data_loader.dataset.cls_mean_size
                # get corresponding transform tensor to numpy
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                

                dets = extract_dets_from_outputs(outputs, K=50)
                # dets = extract_dets_from_outputs_oracle(outputs, targets)

                dets = dets.detach().cpu().numpy()
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        cls_mean_size = cls_mean_size,
                                        threshold = cfg['tester']['threshold'],
                                        to_std_offset = info['to_std_offset'] if _std_method == 'shift' else None,
                                        affine_T = affine_inv_Ts if _std_method == 'uni_K' else None
                                        )                 
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
    
        self.save_results(results, epoch, output_dir=f'./outputs/{_split}')
                

    def save_results(self, results, epoch, output_dir='./outputs/val'):
        output_dir = os.path.join(output_dir, cfg['trainer']['train_idx'], 'epoch_' + str(epoch))
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()


    def eval_and_get_AP(self, epoch):
        # GT label init
        gt_path = self.train_loader.dataset.aug_scene_loader.data_dirs['trainval']['label_dir']
        gt_split_file = os.path.join(self.train_loader.dataset.aug_scene_loader.data_dirs['trainval']['split_dir'], 'val.txt')


        with open(gt_split_file, 'r') as f:
            lines = f.readlines()
        val_image_ids = [int(line) for line in lines]
        gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
        
        self.eval_one_epoch(epoch, _data_loader=self.val_loader, _split='val')
        det_path = './outputs/val/{}/epoch_{}'.format(cfg['trainer']['train_idx'], epoch)

        dt_annos = kitti.get_label_annos(det_path)
        result, ret_dict = kitti_eval(
                            gt_annos, 
                            dt_annos, 
                            ['Car', 'Pedestrian', 'Cyclist'], 
                            eval_types=['bbox', 'bev', '3d', 'aos'])
        self.logger.info(f'========== results ==========')
        self.logger.info(result)

