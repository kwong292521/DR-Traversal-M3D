import os
import numpy as np
import cv2
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

from kitti_utils import angle2class, gaussian_radius, draw_umich_gaussian, Object3d, Calibration, UniIntrinsic
from kitti_augmenter import KITTI_Augmenter

from experiments.config import cfg

class KITTI_Dataset(data.Dataset):
    def __init__(self, split, data_aug=None):
        # basic configuration
        self.split = split
        assert self.split in ['train', 'val', 'trainval', 'test']

        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['dataset'].get('use_3d_center', True)
        self.writelist = cfg['dataset'].get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg['dataset'].get('bbox2d_type', 'proj')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg['dataset'].get('meanshape', False)
        self.class_merging = cfg['dataset'].get('class_merging', False)
        self.use_dontcare = cfg['dataset'].get('use_dontcare', False)

        if self.class_merging:  # False 
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:  # False
            self.writelist.extend(['DontCare'])

        # augment scene loader
        self.aug_scene_loader = KITTI_Augmenter(self.split)

        # data augmentation configuration
        if data_aug is not None:
            self.data_augmentation = data_aug
        else:
            self.data_augmentation = True if split in ['train', 'trainval'] else False
        
        # statistics
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191, 1.62856739, 3.52588311],
                                       [1.73698127, 0.59706367, 1.76282397]], dtype=np.float32)  # H*W*L

        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 4
        self.cur_epoch = 0

        # scenes idx for val / test
        train_split_file = os.path.join(self.aug_scene_loader.data_dirs['trainval']['split_dir'], 'train.txt')
        trainval_split_file = os.path.join(self.aug_scene_loader.data_dirs['trainval']['split_dir'], 'trainval.txt')
        val_split_file = os.path.join(self.aug_scene_loader.data_dirs['trainval']['split_dir'], 'val.txt')
        test_split_file = os.path.join(self.aug_scene_loader.data_dirs['trainval']['split_dir'], 'test.txt')
        
        self.train_idx_list = np.array([
            int(item) for item in [x.strip() for x in open(train_split_file).readlines()]
        ])
        self.trainval_idx_list = np.array([
            int(item) for item in [x.strip() for x in open(trainval_split_file).readlines()]
        ])
        self.val_idx_list = np.array([
            int(item) for item in [x.strip() for x in open(val_split_file).readlines()]
        ])
        self.test_idx_list = np.array([
            int(item) for item in [x.strip() for x in open(test_split_file).readlines()]
        ])
        
        
        if cfg['dataset']['augment'].get('dont_cp_in_ped_scene', False):
            with open(self.aug_scene_loader.data_dirs['trainval']['ped_scenes_file'], 'r') as f:
                ped_scenes = f.read().strip().split('\n')
                ped_scenes = np.array([int(item) for item in ped_scenes])
            _tm = ~np.isin(self.train_idx_list, ped_scenes)
            self.train_idx_list = self.train_idx_list[_tm]
            _tvm = ~np.isin(self.trainval_idx_list, ped_scenes)
            self.trainval_idx_list = self.trainval_idx_list[_tvm]
        
        
        # generate scene list for each epoch
        bs = cfg['dataset']['batch_size']
        self.bs = bs
        n_raw_bs = int(bs * cfg['dataset']['augment']['raw_scene_ratio'])
        
        if self.split == 'train':
            step = len(self.train_idx_list) // bs
            
            self.scene_iter_list = np.zeros((cfg['trainer']['max_epoch'], step, bs), dtype=np.int64)
            self.scene_type_list = np.zeros((cfg['trainer']['max_epoch'], step, bs), dtype=np.bool_)
            n_scene = step * bs
            for i in range(cfg['trainer']['max_epoch']):
                tmp = self.train_idx_list.copy()
                np.random.shuffle(tmp)
                self.scene_iter_list[i] = tmp[:n_scene].reshape(step, bs)
            self.scene_type_list[:, :, n_raw_bs:] = True
                
        elif self.split == 'trainval':
            step = len(self.trainval_idx_list) // bs
            
            self.scene_iter_list = np.zeros((cfg['trainer']['max_epoch'], step, bs), dtype=np.int64)
            self.scene_type_list = np.zeros((cfg['trainer']['max_epoch'], step, bs), dtype=np.bool_)

            n_scene = step * bs
            for i in range(cfg['trainer']['max_epoch']):
                tmp = self.trainval_idx_list.copy()
                np.shuffle(tmp)
                self.scene_iter_list[i] = tmp[:n_scene].reshape(step, bs)            
            self.scene_type_list[:, :, n_raw_bs:] = True


        # ! Sparse Supervised Settings
        if cfg['dataset'].get('sparse_supervised_pt', False) and self.split in ['train']:
            with open(self.aug_scene_loader.data_dirs['trainval']['sparse_objs_ID_file'], 'r') as f:
                sparse_objs_ID = f.read().strip().split('\n')
                sparse_objs_ID = [int(item) for item in sparse_objs_ID]
                sparse_objs_ID = np.array(sparse_objs_ID)   
                self.sparse_objs_ID = sparse_objs_ID
                
            self.aug_scene_loader._get_sparse_objs(self.sparse_objs_ID, np.array([0]))
            self.scene_type_list.fill(True)

        if cfg['dataset'].get('sparse_supervised', False) and self.split in ['train']:
            train_objs_ratio = cfg['dataset']['train_objs_ratio']
            assert train_objs_ratio in [10, 20, 25, 30, 40, 50]
            self.train_objs_ratio = train_objs_ratio
            with open(self.aug_scene_loader.data_dirs['trainval']['sparse_objs_ID_file'], 'r') as f:
                sparse_objs_ID = f.read().strip().split('\n')
                sparse_objs_ID = [int(item) for item in sparse_objs_ID]
                sparse_objs_ID = np.array(sparse_objs_ID)   
                self.sparse_objs_ID = sparse_objs_ID
            with open(self.aug_scene_loader.data_dirs['trainval']['nobjs_ratio_train_idx_file'], 'rb') as f:
                nobjs_ratio_train_idx = pickle.load(f)
                
            if train_objs_ratio == 20:
                self._used_scene_idx_list = nobjs_ratio_train_idx[17].squeeze()
            elif train_objs_ratio == 10:
                self._used_scene_idx_list = nobjs_ratio_train_idx[8].squeeze()
            else:
                self._used_scene_idx_list = nobjs_ratio_train_idx[train_objs_ratio].squeeze()
            _used_scene_idx_list = self._used_scene_idx_list
            
            # process objs database
            self.aug_scene_loader._get_sparse_objs(self.sparse_objs_ID, self._used_scene_idx_list)
            
            # process scene_iter_list
            step = int(len(_used_scene_idx_list) // n_raw_bs)
            self.scene_iter_list = np.zeros((cfg['trainer']['max_epoch'], step, bs), dtype=np.int64)
            n_scene = step * bs
            n_raw_scene = step * n_raw_bs
            
            for i in range(cfg['trainer']['max_epoch']):
                tmp = _used_scene_idx_list.copy()
                np.random.shuffle(tmp)
                
                self.scene_iter_list[i, :, :n_raw_bs] = \
                    tmp[:n_raw_scene].reshape(1, step, n_raw_bs)

            nolabel_idx_list = self.train_idx_list.copy()
            
            np.random.shuffle(nolabel_idx_list)
            n_nolabel = len(nolabel_idx_list)
            _start = 0
            for i in range(cfg['trainer']['max_epoch']):
                if _start + step*(bs-n_raw_bs) > n_nolabel:
                    _start = 0
                    np.random.shuffle(nolabel_idx_list)
                self.scene_iter_list[i, :, n_raw_bs:] = nolabel_idx_list[_start:_start+step*(bs-n_raw_bs)].reshape(step, bs-n_raw_bs)
                _start += step*(bs-n_raw_bs)
            # print('debug')
            self.scene_type_list = np.zeros_like(self.scene_iter_list, dtype=np.bool_)
            self.scene_type_list[:, :, n_raw_bs:] = True



    def __len__(self):
        scenes_count = cfg['dataset']['augment']['scenes_count']
        if self.split in ['train', 'trainval']:
            x = self.scene_iter_list[0].size
        elif self.split == 'val':
            x = scenes_count['n_val_scenes']
        elif self.split == 'test':
            x = scenes_count['n_test_scenes']

        return x
    
    
    def __getitem__(self, iter_idx):
        #  ============================   get inputs   ===========================
        is_empty = False
        if self.split in ['train', 'trainval']:
            b_idx, s_idx = iter_idx // self.bs, iter_idx % self.bs
            scene_idx = self.scene_iter_list[self.cur_epoch, b_idx, s_idx]
            is_empty = self.scene_type_list[self.cur_epoch, b_idx, s_idx]
        elif self.split == 'val':
            scene_idx = int(self.val_idx_list[iter_idx])
        elif self.split == 'test':
            scene_idx = int(self.test_idx_list[iter_idx])
        objs, img, depth_map, calib, img_size, random_flip_flag, to_std_offset, pts_rect2img_T, pose_T = \
            self.aug_scene_loader.load(data_aug=self.data_augmentation, scene_idx=scene_idx, empty=is_empty)
    
        #  ============================   get labels   ==============================
        features_size = self.resolution // self.downsample  # W * H

        # if self.split in ['test', 'val']:
        if self.split == 'test':
            inputs = {}
            inputs['rgb'] = img
            inputs['depth'] = depth_map
            info = {'img_id': scene_idx,
                'img_size': np.array(img_size),
                'bbox_downsample_ratio': np.array([self.downsample]*2),
                'to_std_offset':np.array(to_std_offset),
                'is_empty':is_empty
                }

            return inputs, {}, info  # img / placeholder(fake label) / info

        # computed 3d projected box
        if (self.bbox2d_type == 'proj') and (cfg['dataset']['augment']['use_copy_paste'] and self.data_augmentation):  # default: proj
            for obj in objs:
                obj.box2d = obj.roi

        # data augmentation for labels
        if random_flip_flag:
            for obj in objs:
                [x1, _, x2, _] = obj.box2d
                obj.box2d[0], obj.box2d[2] = self.resolution[0] - x2, self.resolution[0] - x1
                obj.alpha = np.pi - obj.alpha
                obj.ry = np.pi - obj.ry
                if obj.alpha > np.pi:  obj.alpha -= 2 * np.pi  # check range
                if obj.alpha < -np.pi: obj.alpha += 2 * np.pi
                if obj.ry > np.pi:  obj.ry -= 2 * np.pi
                if obj.ry < -np.pi: obj.ry += 2 * np.pi

        # labels encoding
        heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32)  # C * H * W
        box2d_gt = np.zeros((self.max_objs, 4), dtype=np.float32)
        box2d_gt_head = np.zeros((self.max_objs, 4), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
        mask_paste = np.zeros((self.max_objs), dtype=np.uint8)
        object_num = len(objs) if len(objs) < self.max_objs else self.max_objs
        # more info
        # objID = np.zeros((self.max_objs, 1), dtype=np.int64)
        objID = np.ones((self.max_objs, 1), dtype=np.int64) * -1    # -1 denote no obj or dontcare
        cls_ids = np.zeros((self.max_objs), dtype=np.int64)
        occlusion = np.zeros((self.max_objs, 1), dtype=np.float32)
        truc = np.zeros((self.max_objs, 1), dtype=np.float32)
        occ_ratio = np.zeros((self.max_objs, 1), dtype=np.float32)

        for i in range(object_num):
            # filter objs by writelist
            # process 2d bbox & get 2d center
            bbox_2d = objs[i].box2d.copy()

            box2d_gt[i, :] = bbox_2d

            if objs[i].cls_type not in self.writelist:
                continue
            
            # NOTE: Occlusion Ratio Filtering
            # if objs[i].occ_ratio > cfg['dataset']['augment']['max_occ_raw_style']:
            #     continue
            # occ_ratio[i] = objs[i].occ_ratio

            # filter inappropriate samples
            _paste_flag = objs[i].paste_flag if objs[i].paste_flag is not None else False
            if _paste_flag:
                if objs[i].pos[-1] < 2: continue
            else:
                if objs[i].level_str == 'UnKnown' or objs[i].pos[-1] < 2: continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objs[i].pos[-1] > threshold:
                continue

            # ======== id ======== #   
            objID[i] = objs[i].ID

            # ======== occlusion, trucation ======== #
            occlusion[i] = objs[i].occlusion
            truc[i] = objs[i].trucation

            box2d_gt_head[i, :] = bbox_2d

            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d = bbox_2d / self.downsample

            # process 3d bbox & get 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                 dtype=np.float32)  # W * H
            center_3d = objs[i].pos + [0, -objs[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag:  # random flip for center3d
                center_3d[0] = self.resolution[0] - center_3d[0]
            cenrter_3d_raw = center_3d.copy()                   # NOTE: for alpha computing
            center_3d = center_3d / self.downsample

            # generate the center of gaussian heatmap [optional: 3d center or 2d center]
            center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)

            if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
            if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue

            # generate the radius of gaussian heatmap
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            if objs[i].cls_type in ['Van', 'Truck', 'DontCare']:
                draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                continue

            cls_id = self.cls2id[objs[i].cls_type]
            cls_ids[i] = cls_id
            draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

            # encoding 2d/3d offset & 2d size
            indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
            offset_2d[i] = center_2d - center_heatmap
            size_2d[i] = 1. * w, 1. * h

            # encoding depth
            depth[i] = objs[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objs[i].ry, cenrter_3d_raw[0])
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding 3d offset & size_3d
            offset_3d[i] = center_3d - center_heatmap
            src_size_3d[i] = np.array([objs[i].h, objs[i].w, objs[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objs[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            mask_2d[i] = 1
            mask_paste[i] = 1 if objs[i].paste_flag is not None else 0

        # collect return data
        inputs = {}
        inputs['rgb'] = img
        inputs['depth'] = depth_map

        targets = {
            'obj_num': object_num,
            'box2d_gt': box2d_gt,
            'box2d_gt_head': box2d_gt_head,
            'depth': depth,
            'size_2d': size_2d,
            'heatmap': heatmap,
            'offset_2d': offset_2d,
            'indices': indices,
            'size_3d': size_3d,
            'src_size_3d': src_size_3d,
            'offset_3d': offset_3d,
            'heading_bin': heading_bin,
            'heading_res': heading_res,
            'mask_2d': mask_2d,
            # more
            'mask_paste':mask_paste,
            'cls_ids': cls_ids,
            'objID': objID,
            'occlusion': occlusion,
            'truc': truc,
        }


        info = {'img_id': scene_idx,
                'iter_idx': iter_idx,
                'img_size': np.array(img_size),
                'bbox_downsample_ratio': np.array([self.downsample]*2),
                'to_std_offset':np.array(to_std_offset),
                'pts_rect2img_T':pts_rect2img_T,
                'pose_T':pose_T,
                'random_flip_flag':random_flip_flag,
                'is_empty':is_empty,
                }
        return inputs, targets, info


    


class KITTI_PostProcesser():
    def __init__(self):
        self._data_path_init()

        # base parmas
        self.cfg_aug = cfg['dataset']['augment']

        # NOTE: we use BGR format
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda().flip(0)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda().flip(0)
        # depth_mean :  0.12799497 depth_std :  0.18143456  (consider_empty=True, norm=True)
        # depth_mean :  32.638718  depth_std :  46.26581  (consider_empty=True, norm=False)
        self.depth_mean = torch.tensor([0.12799497, 0.12799497, 0.12799497], dtype=torch.float32).cuda()
        self.depth_std = torch.tensor([0.18143456, 0.18143456, 0.18143456], dtype=torch.float32).cuda()

        #====== filling params
        self._fill_kernel_size = 5
        # self._gaussian_kernel = _gaussian_kernel(self._fill_kernel_size, self._fill_kernel_size//2).cuda()
        self._gaussian_kernel = _gaussian_kernel(3, 1).cuda()
        self._max_hole = 10000
        self._max_ref = 1000
        self._n_ref2use = 3

        # get unified calib
        self.uni_intrinsic = UniIntrinsic(uni_size=[384, 1280])


    def _data_path_init(self):
        from data_paths import path_dict
        for key, value in path_dict.items():
            setattr(self, key, value)


    def __call__(self, batch_data, data_aug=True):
        inputs, targets, info = batch_data
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()

        if data_aug:
            inputs = self.scene_level_aug_cuda(inputs, info)

        inputs = self._norm(inputs)

        for key in targets.keys():
            targets[key] = targets[key].cuda()

        return inputs, targets, info


    def scene_level_aug_cuda(self, inputs, info):
        calibs = []
        img_sizes = []
        cam_params = []
        for idx, img_size in zip(info['img_id'], info['img_size']):
            w, h = img_size
            calib = self.get_calibration(idx)
            if self.cfg_aug.get('use_uni_intrinsic'):
                calib = self.uni_intrinsic(calib, h=h, w=w)
            else:
                calib, offset = self._to_std_size(calib, h=h, w=w)
            calibs.append(calib)     
            img_sizes.append(img_size)
            cam_params.append([calib.fu, calib.fv, calib.cu, calib.cv])
        img_sizes = torch.vstack(img_sizes).cuda()
        cam_params = torch.tensor(cam_params).float().cuda()
        P2s = torch.tensor([calib.P2 for calib in calibs]).cuda()

        img = inputs['rgb']
        depth = inputs['depth']
        _, h, w = depth.shape

        points, u_idx, v_idx = self.get_depth_point_cloud_cuda(depth, P2s, ret_idx2d=True)
        points = torch.permute(points, [0, 2, 1])       # (b, N, 3)
        dontcare_pts = (points[..., 2]==0)
        bs, n_pts, _ = points.shape

        points_color = img[:, v_idx, u_idx]             # (b, N, 3)

        #======= ego pose aug and cam moving along z-axis
        points = torch.cat([points, torch.ones(bs, n_pts, 1).cuda()], dim=2)
        T = info['pts_rect2img_T'].cuda().to(torch.float32)                     # (b, 4, 4)
        points = torch.bmm(points, T.permute(0, 2, 1))[..., :-1]
        points_depth = points[..., 2]
        points_img = torch.round((points[..., 0:2] / points[..., 2:3])).to(torch.int64)     # (b, N, 2)

        #======= inside pixels and no-black-edge pixels
        valid_pmask = \
            (points_img[..., 0] >= 0) & (points_img[..., 0] < w) & \
            (points_img[..., 1] >= 0) & (points_img[..., 1] < h)
        valid_pmask = (valid_pmask & ~dontcare_pts)     # (b, N)
        _batch_idx, _pixel_idx = torch.where(valid_pmask)

        #======= render new img and depth
        img_aug = torch.zeros_like(img)
        depth_aug = torch.zeros_like(depth)

        img_aug[_batch_idx, 
                points_img[_batch_idx, _pixel_idx, 1],
                points_img[_batch_idx, _pixel_idx, 0],] = points_color[valid_pmask]
        depth_aug[_batch_idx, 
                points_img[_batch_idx, _pixel_idx, 1],
                points_img[_batch_idx, _pixel_idx, 0],] = points_depth[valid_pmask]

        #======= fill scenes 
        ###=== Step 1 : Ground fill (only in aug_depth > 0)
        rgbd_aug = torch.cat([img_aug.float(), depth_aug[..., None]], dim=-1).permute(0, 3, 1, 2)

        
        # only fill at z_move > 0 scene
        bg_mask = (depth_aug == 0)
        fg_mask = ~bg_mask
        fg_idx = torch.argwhere(fg_mask)       # (b, 3)
        roi = []
        for i in range(bs):
            _fg_idx = fg_idx[fg_idx[:,0] == i][:, 1:]
            if len(_fg_idx) == 0:
                roi.append([0, 0, w-1, h-1])
            else:
                roi.append([_fg_idx[:,1].min(), _fg_idx[:,0].min(), _fg_idx[:,1].max(), _fg_idx[:,0].max()])
        roi = torch.tensor(roi).long().cuda()

        zmove_mask = info['pose_T'][:, 2, 3] > 0

        rgbd_aug[zmove_mask] = self._fill_ground(
            item=rgbd_aug[zmove_mask],
            hole_mask=bg_mask[zmove_mask],
            cam_params=cam_params[zmove_mask],
            max_hole=self._max_hole,
            max_ref=self._max_ref,
            n_ref2use=self._n_ref2use,
            roi=roi[zmove_mask],
            ret_plane_depth=False
        )


        ###=== Step 2 Pooling fill for every case
        bg_mask = (rgbd_aug[:, -1, ...] == 0)
        rgbd_aug = self._fill_hole(
            item=rgbd_aug,
            hole_mask=bg_mask,
            max_kernel_size=[self._fill_kernel_size],
            mean_kernel_size=None,
            min_kernel_size=None,
            final_kernel=self._gaussian_kernel
        ).permute(0, 2, 3, 1)


        #======== flip & crop
        _m = info['random_flip_flag']
        rgbd_aug[_m] = rgbd_aug[_m].flip(dims=[2])


        crop_mask = []
        for i in range(bs):
            if torch.rand(1).item() < self.cfg_aug['random_crop']:
                _crop_mask = torch.ones((h, w), dtype=torch.bool).cuda()
                u_scale = torch.randint(0, self.cfg_aug['crop_u_scale'], (1,)).item()
                v_scale = torch.randint(0, self.cfg_aug['crop_v_scale'], (1,)).item()
                u_start = torch.randint(0, u_scale, (1,)).item() if u_scale != 0 else 0
                v_start = torch.randint(0, v_scale, (1,)).item() if v_scale != 0 else 0
                _crop_mask[v_start:v_start+384-v_scale, u_start:u_start+1280-u_scale] = False
                crop_mask.append(_crop_mask)
            else:
                crop_mask.append(
                    torch.zeros((h, w), dtype=torch.bool).cuda()
                )
        crop_mask = torch.stack(crop_mask)
        rgbd_aug[crop_mask] = 0

        img_aug = rgbd_aug[..., :-1].to(torch.uint8)
        depth_aug = rgbd_aug[..., -1]

        inputs['rgb'] = img_aug
        inputs['depth'] = depth_aug

        return inputs
    

    def _fill_hole(self, item, hole_mask, max_kernel_size=[31], mean_kernel_size=[31], min_kernel_size=[31], final_kernel=None):
        """
        item: (b, c, h, w)  tensor
        hole_mask: (b, h, w)   tensor
        max_kernel_size, mean_kernel_size: list or None
        final_kernel: (k, k) or None
        """
        hole_b_idx, hole_r_idx, hole_c_idx = torch.where(hole_mask)

        _fills = []     
        if max_kernel_size is not None:
            for K in max_kernel_size:
                _fills.append(
                    F.max_pool2d(item, K, stride=1, padding=K//2))
        if mean_kernel_size is not None:
            for K in mean_kernel_size:
                _fills.append(
                    F.avg_pool2d(item, K, stride=1, padding=K//2))
        if min_kernel_size is not None:
            tmp = item.clone()
            tmp[hole_b_idx, :, hole_r_idx, hole_c_idx] = 1000.
            for K in min_kernel_size:
                result = -F.max_pool2d(-tmp, K, stride=1, padding=K//2)
                result[result==1000] = 0
                _fills.append(result)

        _fills = torch.stack(_fills)    # (k, b, c, h, w)

        if final_kernel is not None:
            channel = item.shape[1]
            s = final_kernel.shape[0]
            final_kernel = final_kernel[None, None, ...].repeat(channel, 1, 1, 1)
            tmp = item.clone()
            tmp[hole_b_idx, :, hole_r_idx, hole_c_idx] = _fills[:, hole_b_idx, :, hole_r_idx, hole_c_idx].mean(dim=1)
            tmp = F.conv2d(tmp, final_kernel, stride=1, padding=s//2, groups=channel)
            item[hole_b_idx, :, hole_r_idx, hole_c_idx] = tmp[hole_b_idx, :, hole_r_idx, hole_c_idx]
        else:
            item[hole_b_idx, :, hole_r_idx, hole_c_idx] = _fills[:, hole_b_idx, :, hole_r_idx, hole_c_idx].mean(dim=1)

        return item


    def _fill_ground(self, item, hole_mask, cam_params, cam_height=1.65, max_hole=10000, max_ref=1000, n_ref2use=1, roi=None, ret_plane_depth=False):
        """
        item: (b, c, h, w)  tensor
        hole_mask: (b, h, w)   tensor
        cam_params: (b, 4)  @ (fu, fv, cu, cv)
        roi: (b, 4) @ (u1, v1, u2, v2) or None
        max_hole, max_ref, ref_search_thres: NOTE: we set these to fix gpu memory usage
        """
        b, c, h, w = item.shape

        hole_idx = torch.argwhere(hole_mask)        # (n_hole, 3) @ (b_idx, v, u)
        if roi is not None:
            _m = (hole_idx[:, 2] >= roi[hole_idx[:, 0], 0]) & \
                (hole_idx[:, 2] < roi[hole_idx[:, 0], 2])  & \
                (hole_idx[:, 1] >= roi[hole_idx[:, 0], 1]) & \
                (hole_idx[:, 1] < roi[hole_idx[:, 0], 3])
            hole_idx = hole_idx[_m]
        else:
            roi = torch.tensor([0, 0, w-1, h-1]).float()[None, :].repeat(b, 1).cuda()

        # gen plane pos (b, h, w, 2) @ (x, z)
        # (b, 1, 1)
        fu, fv, cu, cv = \
            cam_params[:, 0][:,None,None], cam_params[:, 1][:,None,None], cam_params[:, 2][:,None,None], cam_params[:, 3][:,None,None]

        uu, vv = torch.meshgrid(
            torch.linspace(0, w-1, w),
            torch.linspace(0, h-1, h))
        uu = uu.T.to(torch.float32).unsqueeze(0).repeat(b, 1, 1).cuda()
        vv = vv.T.to(torch.float32).unsqueeze(0).repeat(b, 1, 1).cuda()    # (b, h, w)
        vv[:, cv.squeeze().long(), :] = torch.floor(cv) + 1

        plane_z = fv * cam_height / (vv - cv)
        plane_x = (uu - cu) * plane_z / fu  
        plane_xz = torch.cat([plane_z[..., None], plane_x[..., None]], dim=-1)

        # find edge (refences pixel)
        edge_mask = (F.max_pool2d(hole_mask.unsqueeze(1).float(), 3, stride=1, padding=1).squeeze(1) != 0) 
        edge_mask = (edge_mask & ~hole_mask)
        ref_idx = torch.argwhere(edge_mask)     # (n_ref, 3) @ (b_idx, v, u)
        
        ref_xz = plane_xz[ref_idx[:,0], ref_idx[:,1], ref_idx[:,2]]     # (n_ref, 2)
        hole_xz = plane_xz[hole_idx[:, 0], hole_idx[:, 1], hole_idx[:, 2]]  # (n_hole, 2)

        for b_idx in range(b):
            # init
            _m = (hole_idx[:, 0] == b_idx)
            _n_hole = _m.sum()
            if _n_hole == 0: continue
            _hole_idx = hole_idx[_m][:, 1:]    # (_n_hole, 2) @ (v, u)
            _hole_xz = hole_xz[_m]

            if _n_hole > max_hole:
                _m = torch.zeros(_n_hole, dtype=torch.bool)
                _m[torch.randperm(_n_hole)[:max_hole]] = True
                _hole_idx = _hole_idx[_m]
                _hole_xz = _hole_xz[_m]
                _n_hole = max_hole

            _m = (ref_idx[:, 0] == b_idx)
            _n_ref = _m.sum()
            if _n_ref == 0: continue
            _ref_idx = ref_idx[_m][:, 1:]      # (_n_ref, 2) @ (v, u)       
            _ref_xz = ref_xz[_m]

            if _n_ref > max_ref:
                _m = torch.zeros(_n_ref, dtype=torch.bool)
                _m[torch.randperm(_n_ref)[:max_ref]] = True
                _ref_idx = _ref_idx[_m]
                _ref_xz = _ref_xz[_m]
                _n_ref = max_ref

            # distance matrix (_n_hole, _n_ref)
            hr_dist = torch.norm(
                _hole_xz.unsqueeze(1) - _ref_xz.unsqueeze(0), dim=-1
            )

            fill_idx = torch.argsort(hr_dist, dim=1)
            fill_idx = fill_idx[:, 0:n_ref2use]     # (_n_hole, n_ref2use)
            fill_idx = _ref_idx[fill_idx]           # (_n_hole, n_ref2use, 2)

            item[b_idx, :, _hole_idx[:,0], _hole_idx[:,1]] = item[b_idx, :, fill_idx[...,0], fill_idx[...,1]].mean(dim=-1)

        if ret_plane_depth:
            return item, plane_z
        else:
            return item


    def _norm(self, inputs):
        img = inputs['rgb']         # (b, h, w, 3)
        img = img.to(torch.float32) / 255
        img = (img - self.mean) / self.std
        img = img.permute(0, 3, 1, 2)   # (b, 3, h, w)
        inputs['rgb'] = img

        if 'depth' in inputs.keys():
            depth = inputs['depth']     # (b, h, w)
            depth = depth.unsqueeze(-1).repeat(1, 1, 1, 3)
            depth /= 255
            depth = (depth - self.depth_mean) / self.depth_std
            depth = depth.permute(0, 3, 1, 2)
            inputs['depth'] = depth
        
        return inputs


    #====== some auxiliary function the same as kitti_augmenter
    def _to_std_size(self, item, std_size=[384, 1280], offset=None, h=None, w=None):
        if isinstance(item, Calibration):
            # _cu, _cv = std_size[1] / 2, std_size[0] / 2
            _cu, _cv = item.cu / w * std_size[1], item.cv / h * std_size[0]
            offset = [_cu - item.cu, _cv - item.cv]
            item.P2[0, 2] = _cu
            item.P2[1, 2] = _cv
            item.cu = _cu
            item.cv = _cv
            return item, offset
        elif isinstance(item, np.ndarray):
            if len(item.shape) == 3:    # img
                item_padded = np.zeros(tuple(std_size + [3]), dtype=item.dtype)
                h, w = item.shape[0], item.shape[1]
                item_padded[int(offset[1]):int(offset[1])+h, 
                            int(offset[0]):int(offset[0])+w] = item
            elif len(item.shape) == 4:  # mask
                item_padded = np.zeros(tuple(list(item.shape)[:2] + std_size), dtype=item.dtype)
                h, w = item.shape[2], item.shape[3]
                item_padded[...,
                            int(offset[1]):int(offset[1])+h, 
                            int(offset[0]):int(offset[0])+w] = item
            elif len(item.shape) == 2:  # depth
                item_padded = np.zeros(tuple(std_size), dtype=item.dtype)
                h, w = item.shape[0], item.shape[1]
                item_padded[int(offset[1]):int(offset[1])+h, 
                            int(offset[0]):int(offset[0])+w] = item
            return item_padded
        elif isinstance(item, Object3d):
            item.box2d += np.array(offset+offset)
            if item.roi is not None: item.roi += np.array(offset+offset)
            return item
    

    def eigen_idx_adapt(func):
        n_trainval_scenes = 7481
        def wrapper(self, idx, *args, **kwargs):
            if idx >= n_trainval_scenes: 
                idx -= n_trainval_scenes
                eigen = True
            else:
                eigen = False
            return func(self, idx, eigen=eigen, *args, **kwargs)
        return wrapper
    

    @eigen_idx_adapt
    def get_calibration(self, idx, noobj=False, eigen=False):
        if eigen:
            calib_file = os.path.join(self.data_dirs['eigen']['calib_dir'], '%06d.txt' % idx)
            layout_file = os.path.join(self.data_dirs['eigen']['layout_noobj_dir'], '%06d.npz' % idx)
        else:
            calib_file = os.path.join(self.data_dirs['trainval']['calib_dir'], '%06d.txt' % idx)
            if noobj:
                layout_file = os.path.join(self.data_dirs['trainval']['layout_noobj_dir'], '%06d.npz' % idx)
            else:
                layout_file = os.path.join(self.data_dirs['trainval']['layout_dir'], '%06d.npz' % idx)

        assert os.path.exists(calib_file)
        assert os.path.exists(layout_file)
        calib = Calibration(calib_file, layout_file)
        return calib


    def get_depth_point_cloud_cuda(self, depth_map, cam_p, min_v=0, flatten=True, in_cam0_frame=True, ret_idx2d=False):
        """Calculates the point cloud from a depth map given the camera parameters (Pytorch batch version)

        Args:
            depth_map: depth map (b, h, w)
            cam_p: camera p matrix  (b, 3, 4)
            
        Returns:
            point_cloud: (b, 3, N) point cloud
        """
        bs = depth_map.shape[0]
        depth_map_shape = depth_map.shape[1:3]

        if min_v > 0:
            # Crop top part
            depth_map[:, 0:min_v] = 0.0

        xx, yy = torch.meshgrid(
            torch.linspace(0, depth_map_shape[1] - 1, depth_map_shape[1]),
            torch.linspace(0, depth_map_shape[0] - 1, depth_map_shape[0]))

        xx = xx.cuda().T
        yy = yy.cuda().T

        # Calibration centre x, centre y, focal length
        centre_u = cam_p[:, 0, 2]
        centre_v = cam_p[:, 1, 2]
        focal_length = cam_p[:, 0, 0]

        i = xx.unsqueeze(0).repeat(bs, 1, 1) - centre_u[:, None, None]
        j = yy.unsqueeze(0).repeat(bs, 1, 1) - centre_v[:, None, None]

        # Similar triangles ratio (x/i = d/f)
        ratio = depth_map / focal_length[:, None, None]
        x = i * ratio
        y = j * ratio
        z = depth_map

        if in_cam0_frame:
            # Return the points in cam_0 frame
            # Get x offset (b_cam) from calibration: cam_p[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[:, 0, 3] / focal_length

            # valid_pixel_mask = depth_map > 0
            # x[valid_pixel_mask] += x_offset[:, None, None].repeat(1, x.shape[1], x.shape[2])[valid_pixel_mask]
            x += x_offset[:, None, None]

        # Return the points in the provided camera frame
        point_cloud_map = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], axis=1)

        if flatten:
            point_cloud = point_cloud_map.view(bs, 3, -1)
            if ret_idx2d:
                return point_cloud.to(torch.float32), xx.to(torch.int64).reshape(-1), yy.to(torch.int64).reshape(-1)
            else:
                return point_cloud.to(torch.float32)
        else:
            return point_cloud_map.to(torch.float32)


def _gaussian_kernel(kernel_size, sigma):
    # Create a 2D Gaussian kernel
    coords = torch.arange(kernel_size).float()
    coords -= (kernel_size - 1) / 2
    coords = coords.view(1, -1) ** 2 + coords.view(-1, 1) ** 2
    kernel = torch.exp(-0.5 * (coords / sigma ** 2))
    kernel /= kernel.sum()
    return kernel




