import os
import copy
import pickle
import cv2
import random
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.ndimage import distance_transform_edt, distance_transform_cdt
from sklearn.linear_model import RANSACRegressor

from kitti_utils import Calibration, Object3d, get_affine_transform, UniIntrinsic
import png
import numba
import tqdm


from experiments.config import cfg

class KITTI_Augmenter():
    def __init__(self, mode):
        #========== base requirements ===========#
        # aug cfg
        self.cfg_aug = cfg['dataset']['augment']
        self.bs = cfg['dataset']['batch_size']
        self.car_only = self.cfg_aug.get('db_car_only', False)
        
        # img info
        self.resolution = np.array([1280, 384])  # W * H
        
        # get unified calib
        self.uni_intrinsic = UniIntrinsic(uni_size=[self.resolution[1], self.resolution[0]])

        # depth complementor
        self.depth_ops = DepthOps()

        # basic path 
        self._data_path_init()

        train_split_file = os.path.join(self.data_dirs['trainval']['split_dir'], 'train.txt')
        self.train_scene_idx_list = [int(x.strip()) for x in open(train_split_file).readlines()]
        self.trainval_scene_idx_list = np.arange(0, self.n_trainval_scenes)
        self.val_scene_idx_list = np.setdiff1d(self.trainval_scene_idx_list, self.train_scene_idx_list)

        self.mode = mode
        if mode in ['val', 'test']:
            return

        #========== train mode requirements ===========#
        # load instance database
        self._init_instance_db()

        if self.cfg_aug['use_train_objs_only']:
            self._get_train_instance_db()

        if self.car_only:
            self._get_car_only_objs()

        self._get_hq_objs()
        self._get_hq_objs_stage_2()

        self._init_fov_mask()
        
        self.objs_hub_xz = np.array([obj.pos[[0, 2]] for obj in self.objs_hub])
        self.objs_hub_xz_grid = self._sample_map_xz2grid(self.objs_hub_xz)
        self.objs_hub_xz_grid_idx = self._sample_map_grid2idx(self.objs_hub_xz_grid)
        self.objs_mask = np.ones(len(self.objs_hub), dtype=np.bool_)
        
        # color augmentor
        self.photometric_augmentor = PhotometricAugmenter()

        
    def _data_path_init(self):
        from data_paths import path_dict
        for key, value in path_dict.items():
            setattr(self, key, value)

    def _init_instance_db(self, db_file=None):
        if db_file is None:
            with open(self.data_dirs['trainval']['instance_db_file'], 'rb') as file:
                instance_database = pickle.load(file)
        else:
            with open(db_file, 'rb') as file:
                instance_database = pickle.load(file)


        self.objs_hub = instance_database['objs_hub']
        self.scene_key = np.array(instance_database['scene_key'])
        self.cls_key = np.array(instance_database['cls_key'])  
        self.truc_key = np.array(instance_database['truc_key']) 
        self.occ_key = np.array(instance_database['occ_key']) 
        self.depth_key = np.array(instance_database['depth_key']) 
        self.depth_bin_key = np.array(instance_database['depth_bin_key']) 
        self.ID_key = np.array(instance_database['ID_key']) 

        #===== NEW
        for obj in self.objs_hub:
            obj.corners3d_rect = obj.generate_corners3d()

    def _get_car_only_objs(self):
        def mask_fun(item, mask):
            return item[mask]

        objs_mask = (self.cls_key == 'Car')

        self.objs_hub = [self.objs_hub[i] for i in range(len(objs_mask)) if objs_mask[i]]
        [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key] = list(map(mask_fun,
            [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key], [objs_mask]*7))

    def _get_hq_objs(self, pts_thres=200, depth_thres=50):
        def mask_fun(item, mask):
            return item[mask]
        
        objs_pts = np.array([len(obj.dense_points_mask2d) for obj in self.objs_hub])

        pts_mask = (objs_pts > pts_thres)
        depth_mask = (self.depth_key < depth_thres)

        objs_mask = pts_mask & depth_mask

        self.objs_hub = [self.objs_hub[i] for i in range(len(objs_mask)) if objs_mask[i]]
        [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key] = list(map(mask_fun, 
            [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key], [objs_mask]*7))
        objs_pts = objs_pts[objs_mask]

    def _get_hq_objs_stage_2(self, fg_thres=0.4, cls2check=['Car']):
        def mask_fun(item, mask):
            return item[mask]

        objs_pts = np.array([len(obj.dense_points_mask2d) for obj in self.objs_hub])
        objs_area = np.array([(obj.box2d[2]-obj.box2d[0])*(obj.box2d[3]-obj.box2d[1]) for obj in self.objs_hub])
        objs_fg = objs_pts / objs_area
        objs_mask = objs_fg > fg_thres

        cls_mask = np.array([obj.cls_type in cls2check for obj in self.objs_hub])
        objs_mask[~cls_mask] = True

        self.objs_hub = [self.objs_hub[i] for i in range(len(objs_mask)) if objs_mask[i]]
        [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key] = list(map(mask_fun,
            [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key], [objs_mask]*7))
        objs_pts = objs_pts[objs_mask]

        self.npts_key = objs_pts

    def _get_train_instance_db(self):
        def mask_fun(item, mask):
            return item[mask]
        
        train_objs_mask = np.array([
            (obj.ID // 50) in self.train_scene_idx_list for obj in self.objs_hub
        ])

        # update instance database trainval to train
        self.objs_hub = [self.objs_hub[i] for i in range(len(train_objs_mask)) if train_objs_mask[i]]
        [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key] = list(map(mask_fun, 
            [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key], [train_objs_mask]*7))


    def _get_sparse_objs(self, sparse_objs_ID, _used_scene_idx_list):
        def mask_fun(item, mask):
            return item[mask]
                
        mask_1 = np.isin(self.ID_key, sparse_objs_ID)
        mask_2 = np.isin(self.ID_key//50, _used_scene_idx_list.squeeze())
        mask = mask_1 | mask_2

        self.objs_hub = [self.objs_hub[i] for i in range(len(mask)) if mask[i]]
        [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key] = list(map(mask_fun, 
            [self.cls_key, self.scene_key, self.truc_key, self.occ_key, self.depth_key, self.depth_bin_key, self.ID_key], [mask]*7))

        self.objs_hub_xz = np.array([obj.pos[[0, 2]] for obj in self.objs_hub])
        self.objs_hub_xz_grid = self._sample_map_xz2grid(self.objs_hub_xz)
        self.objs_hub_xz_grid_idx = self._sample_map_grid2idx(self.objs_hub_xz_grid)
        self.objs_mask = np.ones(len(self.objs_hub), dtype=np.bool_)
    

    def _init_fov_mask(self):
        #==== hard encode reference img info
        cu = 610
        fu = 722
        img_w = 1242
        #====
        
        calib = self.get_calibration(0, noobj=True)
        z_range = calib.layout['x_range']
        x_range = calib.layout['y_range']
        voxel_size = calib.layout['voxel_size'][0]
        h, w = calib.layout['valid_map'].shape
        
        valid_area_center_x, valid_area_center_z = np.meshgrid(
            np.arange(x_range[0], x_range[1], voxel_size) + voxel_size/2,
            np.arange(z_range[0], z_range[1], voxel_size)[::-1] + voxel_size/2,
        )
        z_mask = ((valid_area_center_z > self.cfg_aug['dr_range'][0]) & (valid_area_center_z < self.cfg_aug['dr_range'][1]))
        fov_l = valid_area_center_z * -cu / fu
        fov_r = valid_area_center_z * (img_w - cu) / fu
       
        x_mask = (valid_area_center_x > (fov_l + 1)) & (valid_area_center_x < (fov_r - 1))      
        fov_mask = z_mask & x_mask
        
        self.fov_mask = fov_mask
        self.sample_map_info = {
            'z_range': z_range,
            'x_range': x_range,
            'voxel_size': voxel_size,
            'center_x': valid_area_center_x,
            'center_z': valid_area_center_z,
            'map_h': h,
            'map_w': w,
        }
        
        
    def load(self, data_aug=True, scene_idx=None, empty=None):
        if data_aug:
            if self.cfg_aug['use_copy_paste']:
                return self.load_aug_scene(scene_idx, empty)
            else:
                return self.load_nocp_scene(scene_idx)
        else:
            return self.load_plain_scene(scene_idx)
        

    def _load_testset_scene(self, scene_idx):
        img_file = os.path.join(self.data_dirs['test']['img_dir'], '%06d.png' % scene_idx)
        calib_file = os.path.join(self.data_dirs['test']['calib_dir'], '%06d.txt' % scene_idx)
        assert os.path.exists(img_file)
        assert os.path.exists(calib_file)
        dst_img = cv2.imread(img_file)
        dst_calib = Calibration(calib_file)

        # stdlize resize
        h, w, _ = dst_img.shape
        if self.cfg_aug.get('use_uni_intrinsic'):
            dst_calib, [dst_img] = self.uni_intrinsic(dst_calib, h, w, [dst_img])
            offset = [0, 0]
        else:
            dst_calib, offset = self._to_std_size(dst_calib, h=h, w=w)
            dst_img = self._to_std_size(dst_img, offset=offset)

        dst_objs = None
        dst_depth = None
        random_flip_flag = None
        return dst_objs, dst_img, dst_depth, dst_calib, [w, h], random_flip_flag, offset, np.eye(4), np.eye(4), False

    def load_plain_scene(self, scene_idx):
        # NOTE: for test set
        if self.mode == 'test':
            return self._load_testset_scene(scene_idx)

        dst_objs, dst_calib, dst_img, dst_mask, dst_depth = \
            self.get_scene(scene_idx, get_mask=True)

        # stdlize resize
        h, w, _ = dst_img.shape
        if self.cfg_aug.get('use_uni_intrinsic'):
            dst_calib, [dst_img, dst_depth, dst_objs] = self.uni_intrinsic(dst_calib, h, w, [dst_img, dst_depth, dst_objs])
            offset = [0, 0]     # placeholder
        else:
            dst_calib, offset = self._to_std_size(dst_calib, h=h, w=w)
            dst_img = self._to_std_size(dst_img, offset=offset)
            dst_mask = self._to_std_size(dst_mask, offset=offset)
            dst_depth = self._to_std_size(dst_depth, offset=offset)
            for obj in dst_objs: self._to_std_size(obj, offset=offset)

        random_flip_flag = False

        return dst_objs, dst_img, dst_depth, dst_calib, [w, h], random_flip_flag, offset, np.eye(4), np.eye(4)


    def load_nocp_scene(self, scene_idx):
        dst_objs, dst_calib, dst_img, dst_mask, dst_depth = \
            self.get_scene(scene_idx, get_mask=True)

        # stdlize resize
        h, w, _ = dst_img.shape

        if self.cfg_aug.get('use_uni_intrinsic'):
            dst_calib, [dst_img, dst_depth, dst_objs] = self.uni_intrinsic(dst_calib, h, w, [dst_img, dst_depth, dst_objs])
            offset = [0, 0]     # placeholder
        else:
            dst_calib, offset = self._to_std_size(dst_calib, h=h, w=w)
            dst_img = self._to_std_size(dst_img, offset=offset)
            dst_mask = self._to_std_size(dst_mask, offset=offset)
            dst_depth = self._to_std_size(dst_depth, offset=offset)
            for obj in dst_objs: self._to_std_size(obj, offset=offset)

        # img photometric augmentation
        dst_img = self.photometric_augmentor.scene_aug(dst_img, ret_norm_img=False)      # [0, 1]

        #===== Step 8 : Scene Level Augmentation(cam move, ego pose) =====#
        dst_objs, objs_paste, dst_img_aug, dst_depth_aug, pts_rect2img_T, pose_T = self.scene_level_aug(
            dst_objs=dst_objs,
            objs_paste=[],
            img=dst_img,
            dense_depth=dst_depth,
            calib=dst_calib,
            dst_mask=dst_mask,
            aug_depth=None,
            aug_pitch=None,
            aug_roll=None,
        )

        # random flip
        random_flip_flag = (np.random.random() < self.cfg_aug['random_flip'])

        return dst_objs, dst_img_aug, dst_depth_aug, dst_calib, [w, h], random_flip_flag, offset, pts_rect2img_T, pose_T


    def load_aug_scene(self, scene_idx, empty):                
        #===== Step 1 : Initialize =====#        
        _flag = 'empty' if empty else 'raw'
        n_sampled_objs = int(np.random.choice(list(np.arange(self.cfg_aug[f'n_paste_range_{_flag}'][0], 
                                                             self.cfg_aug[f'n_paste_range_{_flag}'][1])), 1))
        
        # load dst scene
        if empty:
            dst_objs, dst_calib, dst_img, dst_mask, dst_depth = \
                self.get_scene_empty(scene_idx)
        else:
            dst_objs, dst_calib, dst_img, dst_mask, dst_depth = \
                self.get_scene(scene_idx, get_mask=True)

        # sample objs in the scenes
        if self.cfg_aug['pos_resample']:
            paste_objs_hub_idx, paste_objs_ID, paste_objs_new_xz = self.sample_newxz_objs(n_sampled_objs, dst_calib.layout)
        else:
            paste_objs_hub_idx, paste_objs_ID, paste_objs_new_xz = self.sample_rawxz_objs(n_sampled_objs, dst_calib.layout)
            
        sampled_objs = [copy.deepcopy(self.objs_hub[idx]) for idx in paste_objs_hub_idx]
        sampled_objs_calib = [self.get_calibration(self.scene_key[idx]) for idx in paste_objs_hub_idx]

        # stdlize resize
        h, w, c = dst_img.shape

        if self.cfg_aug.get('use_uni_intrinsic'):
            dst_calib, [dst_img, dst_depth, dst_objs] = self.uni_intrinsic(dst_calib, h, w, [dst_img, dst_depth, dst_objs])
            offset = [0, 0]     # placeholder
        else:
            dst_calib, offset = self._to_std_size(dst_calib, h=h, w=w)
            dst_img = self._to_std_size(dst_img, offset=offset)
            dst_mask = self._to_std_size(dst_mask, offset=offset)
            dst_depth = self._to_std_size(dst_depth, offset=offset)
            if not empty:
                for obj in dst_objs: self._to_std_size(obj, offset=offset)
                                

        #===== Step 2 : color augmentation =====#
        if self.cfg_aug['color_aug']:
            for obj in sampled_objs:
                obj.dp_mask2d_color = self.photometric_augmentor.ins_aug(
                    img=obj.dp_mask2d_color[None, ...],            # (N, 3) -> (1, N, 3) h,w,3 adaptive
                    hsv_delta=self.cfg_aug['hsv_delta'],
                    thres=30
                )[0]

        #===== Step 3 : move sampled objs src to dst =====#
        objs_paste = []
        for i, (obj, calib) in enumerate(zip(sampled_objs, sampled_objs_calib)):
            obj = self.move_obj_src2dst(
                obj=obj,
                dr_xz=paste_objs_new_xz[i],
                src_calib=calib,
                dst_calib=dst_calib,
                dst_mask=dst_mask,
                )
            objs_paste.append(obj)


        #===== Step 4 : check whether the sampled objs locate in the valid area =====#
        if self.cfg_aug['obj2scene_style'] != 'soft':
            objs_paste = self.valid_area_filter(objs_paste, dst_calib)
        num_sampled_objs = len(objs_paste)

        #===== Step 5 : 3d bev collision detection =====#
        # check whether src(paste) objs collide with dst objs
        if not empty:
            src2dst_collision = self.box_collision_check(objs_paste, dst_objs)   # (num_sampled_objs, num_dst_objs)
            check = np.sum(src2dst_collision, axis=1)
            objs_paste = [objs_paste[i] for i in range(int(num_sampled_objs)) if check[i]==0]
        num_sampled_objs = len(objs_paste)
        # check whether collision occur between src objs 
        src2src_collision = self.box_collision_check(objs_paste, objs_paste)
        src2src_collision = src2src_collision & (~np.eye(num_sampled_objs, dtype=np.bool_))  # dontcare self collision
        if np.sum(src2src_collision != 0):
            # scan per obj, if collide with perious, discard
            check = [0]
            for i in range(1, num_sampled_objs):
                if (True in np.isin(check, np.where(src2src_collision[i])[0])):
                    continue
                else:
                    check.append(i)
            objs_paste = [objs_paste[i] for i in check]
            num_sampled_objs = len(objs_paste)


        #===== Step 6 :  2D foreground collision detection and select valid paste objs =====#
        # IN THIS VERSION, LEAVE TO STEP 8

        #==== Step 7 : object size augmentation =====#
        if self.cfg_aug.get('use_size_aug', False):
            for obj in objs_paste:
                self.obj_size_aug(obj=obj,          
                                  calib=dst_calib,
                                  )

        #==== Step 7 : Img Level augmentation =====#
        # img photometric augmentation
        dst_img = self.photometric_augmentor.scene_aug(dst_img, ret_norm_img=False)      # [0, 1]

        #===== Step 8 : Scene Level Augmentation(cam move, ego pose) =====#
        dst_objs, objs_paste, dst_img_aug, dst_depth_aug, pts_rect2img_T, pose_T = self.scene_level_aug(
            dst_objs=dst_objs,
            objs_paste=objs_paste,
            img=dst_img,
            dense_depth=dst_depth,
            calib=dst_calib,
            dst_mask=dst_mask,
            aug_depth=None,
            aug_pitch=None,
            aug_roll=None,
        )

        # random flip
        random_flip_flag = (np.random.random() < self.cfg_aug['random_flip'])

        # all_objs, rgb, depth, calib, raw_img_size, flip
        return dst_objs+objs_paste, dst_img_aug, dst_depth_aug, dst_calib, [w, h], random_flip_flag, offset, pts_rect2img_T, pose_T


    #========================== utils
    def scene_level_aug(
            self,
            dst_objs,
            objs_paste,
            img,
            dense_depth,
            calib,
            dst_mask,
            aug_depth=None,
            aug_pitch=None,
            aug_roll=None
            ):
        """
        P.S.:
        input points in lidar coord
        ops in rect coord
        args:
            objs_paste: n paste obj
            img: (h, w, 3)
            dense_depth: (h, w)
            calib: calibration
            aug_depth: move z value (batch_equal)
            aug_pitch, aug_roll: ego pose perturbation NOTE: pitch <-> r_x  roll <-> r_z
        ret: 
            objs_paste, img_aug, depth_aug
        """
        is_emtpy = True if len(dst_objs) == 0 else False
        num_paste_objs = len(objs_paste)

        if aug_depth is None:
            aug_depth = (np.random.rand() * self.cfg_aug['aug_depth_range'] * 2 - self.cfg_aug['aug_depth_range']) * \
                np.random.choice([0, 1], p=[1-self.cfg_aug['ego_pose_aug_prob'], self.cfg_aug['ego_pose_aug_prob']])
        if aug_pitch is None:
            aug_pitch = (np.random.rand() * self.cfg_aug['aug_pitch_range'] * 2 - self.cfg_aug['aug_pitch_range']) * \
                np.random.choice([0, 1], p=[1-self.cfg_aug['ego_pose_aug_prob'], self.cfg_aug['ego_pose_aug_prob']])
        if aug_roll is None:
            aug_roll = (np.random.rand() * self.cfg_aug['aug_roll_range'] * 2 - self.cfg_aug['aug_roll_range']) * \
                np.random.choice([0, 1], p=[1-self.cfg_aug['ego_pose_aug_prob'], self.cfg_aug['ego_pose_aug_prob']])

        pose_T = self.get_trans(rx=aug_pitch, ry=0, rz=aug_roll, tz=aug_depth)

        img_shape = dense_depth.shape


        dst_obj_pos = np.array([obj.pos for obj in dst_objs]) if not is_emtpy else np.random.randn(1, 3)
        paste_obj_pos = np.array([obj.pos for obj in objs_paste])     # (N, 3) @ rect
        if num_paste_objs != 0:
            paste_points = np.concatenate([obj.dense_points_mask2d for obj in objs_paste])      # @ lidar
        
        #======= get trans matrix
        homo_V2C = self.get_homo_trans(calib.V2C)
        homo_R0 = self.get_homo_trans(calib.R0)
        homo_P2 = self.get_homo_trans(calib.P2)
        pts_lidar2img_T = homo_P2 @ homo_R0 @ homo_V2C      
        pts_rect2img_T = homo_P2 @ pose_T

        #======= update info 
        if not is_emtpy:
            dst_obj_pos = self.apply_trans(dst_obj_pos, pose_T)
            for i, pos in enumerate(dst_obj_pos):
                dst_objs[i].pos = pos
                dst_objs[i].update_corners3d(pose_T)
                _, roi = dst_objs[i].gen_corners3d_2dproj(calib)
                dst_objs[i].roi = self.roi_clip(roi, img.shape[0], img.shape[1]).astype(np.int64)
        if num_paste_objs != 0:
            paste_obj_pos = self.apply_trans(paste_obj_pos, pose_T)
            for i, pos in enumerate(paste_obj_pos):
                objs_paste[i].pos = pos
                objs_paste[i].update_corners3d(pose_T)
                _, roi = objs_paste[i].gen_corners3d_2dproj(calib)
                objs_paste[i].roi = self.roi_clip(roi, img.shape[0], img.shape[1]).astype(np.int64)

        #======= proj paste points to image
        if num_paste_objs != 0:
            paste_points_img = self.apply_trans(paste_points, pts_lidar2img_T)
            paste_points_depth = paste_points_img[:, 2]
            paste_points_img = np.round((paste_points_img[:, 0:2] / paste_points_img[:, 2:3])).astype(np.int32)

            valid_paste_pmask = \
                (paste_points_img[:,0] >= 0) & (paste_points_img[:,0] < img_shape[1]) & \
                (paste_points_img[:,1] >= 0) & (paste_points_img[:,1] < img_shape[0])
            
            
            #======= render paste objs (use z-buffer) and {! get per obj occlusion}
            img_aug, depth_aug, objs_paste, dst_objs = self.render_paste_obj(
                objs_paste, 
                dst_objs,
                dst_mask,
                valid_paste_pmask, 
                paste_points_img, 
                img, 
                dense_depth, 
                paste_points_depth, 
                calib,
                add_shadow=self.cfg_aug['add_shadow'],
                pose_T=pose_T
            )
        else:
            img_aug = img
            depth_aug = dense_depth

        return dst_objs, objs_paste, img_aug, depth_aug, pts_rect2img_T, pose_T


    def move_obj_src2dst(
            self,
            obj,
            dr_xz, 
            src_calib, 
            # put box on road plane params
            dst_mask,
            dst_calib, 
            to_road_ref='top'       
            ):
        #===== Step 1 : Resample Depth =====#
        if (not np.allclose(dr_xz, np.array([-1,-1]))):      
            pos_cam = obj.pos
            pos_lidar = src_calib.rect_to_lidar(pos_cam.reshape(-1, 3)).squeeze()
            
            pos_cam[0] = dr_xz[0]
            pos_cam[2] = dr_xz[1]
            
            # new lidar_coord pos
            pos_lidar_new = src_calib.rect_to_lidar(pos_cam.reshape(-1, 3)).squeeze()
            mv_pos = pos_lidar - pos_lidar_new

            # update 
            obj.pos = pos_cam
            obj.points -= mv_pos
            obj.dense_points_mask2d -= mv_pos

        #===== Step 2 : put obj on the road plane =====#
        pos_cam = obj.pos
        cam_road_plane = dst_calib.layout['cam_plane']
        ca, cb, cc, cd = cam_road_plane
        new_cam_Y = (-cd - ca * pos_cam[0] - cc * pos_cam[2]) / cb      # ax+by+cz+d=0 y=(ax+cz+d) / (-b)
        pos_cam[1] = new_cam_Y

        lidar_road_plane = dst_calib.layout['lidar_plane']
        la, lb, lc, ld = lidar_road_plane

        assert to_road_ref in ['bottom', 'top', 'mean']
        if to_road_ref == 'bottom':
            _idx = np.argmin(obj.dense_points_mask2d[:, 2])
            pos_lidar = obj.dense_points_mask2d[_idx]
            new_lidar_Z = (la * pos_lidar[0] + lb * pos_lidar[1] + ld) / (-lc)    # z=(ax+by+d) / (-c)
        elif to_road_ref == 'top':
            _idx = np.argmax(obj.dense_points_mask2d[:, 2])
            pos_lidar = obj.dense_points_mask2d[_idx]
            new_lidar_Z = (la * pos_lidar[0] + lb * pos_lidar[1] + ld+obj.h) / (-lc)
        elif to_road_ref == 'mean':
            raise NotImplementedError

        pos_lidar_new = pos_lidar.copy()
        pos_lidar_new[2] = new_lidar_Z
        mv_pos = pos_lidar - pos_lidar_new

        # update
        obj.pos = pos_cam
        obj.points -= mv_pos
        obj.dense_points_mask2d -= mv_pos

        #===== Step 3 : roi clip, cut (and resize) =====#
        # raw roi denote roi before clip and cut(after resample depth and put road, roi may be trucated)
        obj.corners3d_rect = obj.generate_corners3d()       # update corners
        corners3d_proj, raw_roi = obj.gen_corners3d_2dproj(dst_calib)
        raw_roi = raw_roi.astype(np.int64)
        roi_mask = cv2.resize(obj.roi_mask.astype(np.uint8), (raw_roi[2]-raw_roi[0]+1, raw_roi[3]-raw_roi[1]+1)).astype(np.bool_)

        roi = self.roi_clip(raw_roi, dst_mask.shape[-2], dst_mask.shape[-1]).astype(np.int64)
        _, roi_mask = self.roi_cut(raw_roi, roi, roi_mask=roi_mask)
        # after cut, dont need resize
        obj.roi = roi
        obj.roi_mask = roi_mask

        return obj


    def render_paste_obj(self, objs_paste, dst_objs, dst_mask, valid_paste_pmask, paste_points_img, img_aug, depth_aug, paste_points_depth, calib, add_shadow=False, pose_T=None):
        #================ Step 0* : Render Shadow
        if add_shadow:
            self.render_shadow(img_aug, objs_paste, calib, 
                        alpha=self.cfg_aug['shadow_alpha'],     # 0.2
                        n_light=self.get_nlight(
                            theta_yx=self.cfg_aug['light_theta_yx'], 
                            theta_xz=self.cfg_aug['light_theta_xz']),
                        shadow_smooth=True,
                        pose_T=pose_T
                        )


        #================ Step 1 : Render per paste objects in slices 
        _slice_imgs = []
        _slice_depths = []
        _slice_masks = []
        _start = 0
        _end = 0
                
        paste_flags = []
        for obj in objs_paste:
            _end += obj.dp_mask2d_color.shape[0]
            _color = obj.dp_mask2d_color[valid_paste_pmask[_start:_end]]
            _img = np.zeros_like(img_aug)
            _depth = np.zeros_like(depth_aug) + 999999.
            _fg_mask = np.zeros_like(depth_aug, dtype=np.bool_)
            _pts_img = paste_points_img[_start:_end][valid_paste_pmask[_start:_end]]
            _pts_depth = paste_points_depth[_start:_end][valid_paste_pmask[_start:_end]]
        
            if len(_pts_img) == 0:
                _start = _end
                paste_flags.append(False)
                continue

            _img[_pts_img[:, 1], _pts_img[:, 0]] = _color
            _depth[_pts_img[:, 1], _pts_img[:, 0]] = _pts_depth
            _fg_mask[_pts_img[:, 1], _pts_img[:, 0]] = True

            _roi = [_pts_img[:,0].min(), _pts_img[:,1].min(),
                _pts_img[:,0].max(), _pts_img[:,1].max()]        # (x1, y1, x2, y2)

            _fg_roi_mask = self.fill_instance(
                img=_img[_roi[1]:_roi[3]+1, _roi[0]:_roi[2]+1],
                depth=_depth[_roi[1]:_roi[3]+1, _roi[0]:_roi[2]+1],
                mask=_fg_mask[_roi[1]:_roi[3]+1, _roi[0]:_roi[2]+1],
                k_size=3,
                iter=1,
                edge_refine=False
            )
            _fg_mask[_roi[1]:_roi[3]+1, _roi[0]:_roi[2]+1] = _fg_roi_mask

            if _fg_roi_mask.sum() < 200:
                _start = _end
                paste_flags.append(False)
                continue

            _slice_imgs.append(_img)
            _slice_depths.append(_depth)
            _slice_masks.append(_fg_mask)
            _start = _end
            
            obj.paste_flag = True
            paste_flags.append(True)
            
        objs_paste = [objs_paste[i] for i in range(len(objs_paste)) if paste_flags[i]]
        n_objs_paste = len(objs_paste)
        objs_paste_depth = np.array([obj.pos[2] for obj in objs_paste])
        
        
        #================ Step 1.5 : Merge Slice into array and reshape to vector & Sorted pasted objs
        if n_objs_paste != 0:
            # merge
            slice_imgs = np.stack(_slice_imgs, axis=0).reshape(n_objs_paste, -1, 3)      # (n, h*w, 3)
            slice_depths = np.stack(_slice_depths, axis=0).reshape(n_objs_paste, -1)     # (n, h*w)
            slice_masks = np.stack(_slice_masks, axis=0).reshape(n_objs_paste, -1)       # (n, h*w)
            
            # resort                        
            _priority = cfg['dataset']['augment']['occ_priority']
            assert _priority in ['c2d', 'd2c', 'mix']
            
            if _priority == 'c2d':
                sorted_idx = np.argsort(objs_paste_depth)
            elif _priority == 'd2c':
                sorted_idx = np.argsort(objs_paste_depth)[::-1]
            elif _priority == 'mix':
                sorted_idx = np.argsort(objs_paste_depth)
                if np.random.randn() > 0.5: sorted_idx = sorted_idx[::-1]
                
            objs_paste = [objs_paste[i] for i in sorted_idx]
            slice_imgs = slice_imgs[sorted_idx]
            slice_depths = slice_depths[sorted_idx]
            slice_masks = slice_masks[sorted_idx]
            objs_paste_depth = objs_paste_depth[sorted_idx]
        else:
            return img_aug, depth_aug, objs_paste, dst_objs
        
        
        #================ Step 2* : Occulusion Computation for each objects
        occ_thres = np.random.choice(cfg['dataset']['augment']['occ_thres_choice'])
        
        # add dst_objs-wise foreground reference
        _dst_cls_mask = np.array([(obj.cls_type in cfg['dataset']['writelist']) and (dst_mask[i].sum() != 0) and (obj.occlusion <= 2) for i, obj in enumerate(dst_objs)])
        dst_objs = [dst_objs[i] for i in range(len(_dst_cls_mask)) if _dst_cls_mask[i]]
        n_dst_objs = len(dst_objs)
        if n_dst_objs != 0:
            slice_masks_dst = dst_mask[_dst_cls_mask].reshape(len(dst_objs), -1)    # (n, h*w)        
            dst_objs_depth = np.array([obj.pos[-1] for obj in dst_objs])
            c2d_idx = np.argsort(dst_objs_depth)
            dst_objs = [dst_objs[i] for i in c2d_idx]
            slice_masks_dst = slice_masks_dst[c2d_idx]
            dst_objs_depth = dst_objs_depth[c2d_idx]
        else:
            dst_objs_depth = np.array([])
            slice_masks_dst = np.empty((0, depth_aug.size), dtype=np.bool_)
        
        # get fg idx & raw area
        dst_areas = np.zeros((n_dst_objs), dtype=np.int32)
        paste_areas = np.zeros((n_objs_paste), dtype=np.int32)
                
        dst_occ_level = np.array([obj.occlusion for obj in dst_objs])
        paste_occ_level = np.array([obj.occlusion for obj in objs_paste])
        occ_offset = cfg['dataset']['augment']['occ_offset_kitti_level']
        
        dst_fg_idx = []
        paste_fg_idx = []

        for i in range(n_dst_objs):
            _fg_idx = np.where(slice_masks_dst[i])[0]
            dst_fg_idx.append(_fg_idx)
            dst_areas[i] = len(_fg_idx)
        for i in range(n_objs_paste):
            _fg_idx = np.where(slice_masks[i])[0]
            paste_fg_idx.append(_fg_idx)
            paste_areas[i] = len(_fg_idx)
        
        dst_occ_areas = np.zeros((n_dst_objs), dtype=np.int32)
        paste_occ_areas = np.zeros((n_objs_paste), dtype=np.int32)
        
        bg_depth = depth_aug.copy().reshape(-1)
        paste_flags = []
        for i in range(n_objs_paste):
            _fg_idx = paste_fg_idx[i]
            _d = objs_paste[i].pos[-1]
            
            # occluded by background
            occ1_m = bg_depth[_fg_idx] < _d
            occ1 = occ1_m.sum()
            
            # occlude the dst objs
            occ_dst = slice_masks_dst[:, _fg_idx].sum(axis=1)
            
            # get occ ratio
            occ1_ratio = (occ1 + paste_occ_areas[i]) / paste_areas[i]
            occ2_ratio = (occ_dst + dst_occ_areas) / dst_areas
            
            # judge
            if (occ1_ratio > occ_thres) or np.any(occ2_ratio > occ_thres):
                paste_flags.append(False)
                continue
            
            #### update
            # hackimplement update bg_depth
            bg_depth[_fg_idx[~occ1_m]] = _d
            
            # update slice_masks_dst (to unseen)
            slice_masks_dst[:, _fg_idx] = False
            
            # update occ_areas
            dst_occ_areas += occ_dst
            
            # add new paste to dst
            slice_masks_dst = np.concatenate([slice_masks_dst, slice_masks[i:i+1]], axis=0)
            dst_occ_areas = np.append(dst_occ_areas, occ1 + paste_occ_areas[i])
            dst_areas = np.append(dst_areas, paste_areas[i])
            
            paste_flags.append(True)
            
        objs_paste = [objs_paste[i] for i in range(len(objs_paste)) if paste_flags[i]]
        slice_imgs = slice_imgs[paste_flags]
        slice_depths = np.concatenate([slice_depths[paste_flags], depth_aug.reshape(1, -1)], axis=0)
        n_objs_paste = len(objs_paste)
      
        #================ Step 3 : Global Z-buffer Rendering 
        std_h, std_w = 384, 1280

        if n_objs_paste != 0:
            overlap_mask = np.any(slice_masks, axis=0)          
            overlap_pixel_idx = np.argwhere(overlap_mask).squeeze(1)

            saved_sample_idx = np.argmin(slice_depths[:, overlap_mask], axis=0)
            _not_bg_mask = saved_sample_idx < n_objs_paste

            overlap_pixel_idx = overlap_pixel_idx[_not_bg_mask]
            saved_sample_idx = saved_sample_idx[_not_bg_mask]
            overlap_pixel_u = overlap_pixel_idx % std_w
            overlap_pixel_v = overlap_pixel_idx // std_w

            # refresh img and dense depth
            img_aug[overlap_pixel_v, overlap_pixel_u] = slice_imgs[saved_sample_idx, overlap_pixel_idx]
            depth_aug[overlap_pixel_v, overlap_pixel_u] = slice_depths[saved_sample_idx, overlap_pixel_idx]

        return img_aug, depth_aug, objs_paste, dst_objs


    def obj_size_aug(self, obj, calib, h_s=None, w_s=None, l_s=None):
        if h_s is None: 
            h_s = (np.random.rand() * self.cfg_aug['size_aug_h_scale'] * 2 - self.cfg_aug['size_aug_h_scale']) * \
                np.random.choice([0, 1], p=[1-self.cfg_aug['size_aug_prob'], self.cfg_aug['size_aug_prob']])
        if w_s is None: 
            w_s = (np.random.rand() * self.cfg_aug['size_aug_w_scale'] * 2 - self.cfg_aug['size_aug_w_scale']) * \
                np.random.choice([0, 1], p=[1-self.cfg_aug['size_aug_prob'], self.cfg_aug['size_aug_prob']])
        if l_s is None: 
            l_s = (np.random.rand() * self.cfg_aug['size_aug_l_scale'] * 2 - self.cfg_aug['size_aug_l_scale']) * \
                np.random.choice([0, 1], p=[1-self.cfg_aug['size_aug_prob'], self.cfg_aug['size_aug_prob']])

        # update label
        obj.h *= (1 + h_s)
        obj.w *= (1 + w_s)
        obj.l *= (1 + l_s)
                
        obj.corners3d_rect = obj.generate_corners3d()       # update corners
        corners3d_proj, raw_roi = obj.gen_corners3d_2dproj(calib)
        raw_roi = raw_roi.astype(np.int64)
        # roi_mask = cv2.resize(obj.roi_mask.astype(np.uint8), (raw_roi[2]-raw_roi[0]+1, raw_roi[3]-raw_roi[1]+1)).astype(np.bool_)

        roi = self.roi_clip(raw_roi, 384, 1280).astype(np.int64)
        # _, roi_mask = self.roi_cut(raw_roi, roi, roi_mask=roi_mask)
        # after cut, dont need resize
        obj.roi = roi
        # obj.roi_mask = roi_mask
                
        # get trans matrix
        R = np.array([[np.cos(obj.ry), 0, np.sin(obj.ry)],
                    [0, 1, 0],
                    [-np.sin(obj.ry), 0, np.cos(obj.ry)]])
        T0 = np.eye(4)
        T0[0:3, 0:3] = R
        T0[0:3, 3] = obj.pos
        
        T1 = np.eye(4)
        T1[0:3, 0:3] = calib.R0.T

        T2 = np.eye(4)
        T2[0:3] = calib.C2V

        T0_inv = self.inverse_rigid_trans(T0)
        T1_inv = self.inverse_rigid_trans(T1)
        T2_inv = self.inverse_rigid_trans(T2)

        T_imloc2lidar = T2 @ T1 @ T0
        T_lidar2imloc = T0_inv @ T1_inv @ T2_inv

        T_locs = np.eye(4) * np.array([[
            1 + l_s,
            1 + h_s,
            1 + w_s,
            1 
        ]])

        T_s = T_imloc2lidar @ T_locs @ T_lidar2imloc

        # update obj model
        pts = obj.dense_points_mask2d
        pts = self.cart_to_hom(pts)
        pts = (pts @ T_s.T)[:, 0:3]
        obj.dense_points_mask2d = pts
        
        # update lidar pts
        pts_lidar = obj.points
        pts_lidar = self.cart_to_hom(pts_lidar)
        pts_lidar = (pts_lidar @ T_s.T)[:, 0:3]
        obj.points = pts_lidar

    
    def sample_rawxz_objs(self, n_objs, layout):
        voxel_size = layout['voxel_size'][0]
        _map = (layout['valid_map'] == 1) * self.fov_mask
        
        _sample_area = np.argwhere(_map)
        _sample_area_center_x = self.sample_map_info['center_x'][_sample_area[:,0], _sample_area[:,1]]
        _sample_area_center_z = self.sample_map_info['center_z'][_sample_area[:,0], _sample_area[:,1]]

        _sample_area_grid_idx = self._sample_map_grid2idx(_sample_area[:, ::-1])

        _m = np.isin(self.objs_hub_xz_grid_idx, _sample_area_grid_idx)
        # _m = _m & objs_mask
        
        sampled_objs_hub_idx = list(np.random.choice(np.where(_m)[0], min(_m.sum(), n_objs), replace=False))
        sampled_newxz = self.objs_hub_xz[sampled_objs_hub_idx]

        sampled_objs_ID = self.ID_key[sampled_objs_hub_idx]

        return sampled_objs_hub_idx, sampled_objs_ID, sampled_newxz


    def sample_newxz_objs(self, n_objs, layout):
        voxel_size = layout['voxel_size'][0]
        _map = (layout['valid_map'] == 1) * self.fov_mask
        
        _sample_area = np.argwhere(_map)
        _sample_area_center_x = self.sample_map_info['center_x'][_sample_area[:,0], _sample_area[:,1]]
        _sample_area_center_z = self.sample_map_info['center_z'][_sample_area[:,0], _sample_area[:,1]]

        sampled_objs_hub_idx = []
        sampled_newxz = np.empty((0, 2))

        for i in range(n_objs):            
            _idx = np.random.choice(np.arange(len(_sample_area_center_z)))
            x_rs = _sample_area_center_x[_idx]
            z_rs = _sample_area_center_z[_idx]
            
            # random choice position
            z_rs = float(z_rs) + np.random.uniform(-voxel_size/2, voxel_size/2)
            x_rs = float(x_rs) + np.random.uniform(-voxel_size/2, voxel_size/2)
            _new_xz = np.array([[x_rs, z_rs]])

            # random choice object
            z_m = self.objs_hub_xz[:, 1] < (z_rs / (1 - self.cfg_aug['dr_ratio']))
            s_m = self.objs_hub_xz[:, 0] > 0 if x_rs > 0 else self.objs_hub_xz[:, 0] < 0
            obj_m = z_m & s_m
            
            if np.all(~obj_m): continue
            
            # random choice object
            idx = int(np.random.choice(np.where(obj_m)[0], 1))
            
            sampled_objs_hub_idx.append(idx)
            sampled_newxz = np.concatenate([sampled_newxz, _new_xz], axis=0)

        sampled_objs_ID = self.ID_key[sampled_objs_hub_idx]

        return sampled_objs_hub_idx, sampled_objs_ID, sampled_newxz



    #========================== KEY auxiliary utils
    def _sample_map_grid2idx(self, grid):
        """
        grid @ xz coord
        """
        h = self.sample_map_info['map_h']
        w = self.sample_map_info['map_w']
        out = grid[:, 1] * w + grid[:, 0]
        return out    
    
    def _sample_map_xz2grid(self, xz):
        h = self.sample_map_info['map_h']
        w = self.sample_map_info['map_w']
        out = ((np.clip(np.floor((xz - np.array([[self.sample_map_info['x_range'][0], self.sample_map_info['z_range'][0]]])) / self.sample_map_info['voxel_size']),
                                                     np.array([[0, 0]]), np.array([[w-1, h-1]])) * np.array([[1, -1]])) + np.array([[0, h-1]])).astype(np.int64)
        return out
    
    def valid_area_filter(self, objs_paste, dst_calib):
        n_objs = len(objs_paste)

        valid_map = dst_calib.layout['valid_map'] == 1
        h, w = valid_map.shape
        x_range = dst_calib.layout['x_range']
        y_range = dst_calib.layout['y_range']
        voxel_size = dst_calib.layout['voxel_size']

        objs_pts = np.array([np.concatenate([obj.generate_corners3d()[:4, [0, 2]], obj.pos[None, [0, 2]]], axis=0) for obj in objs_paste])
        objs_pts_grid = self._sample_map_xz2grid(objs_pts.reshape(n_objs*5, 2)).reshape(n_objs, 5, 2)
        
        pts_in_valid = valid_map[objs_pts_grid[..., 1], objs_pts_grid[..., 0]]      # (n_objs, 5)

        if self.cfg_aug['obj2scene_style'] == 'soft':
            objs_in_valid = pts_in_valid[:, -1]
        elif self.cfg_aug['obj2scene_style'] == 'mod':
            objs_in_valid = np.all(pts_in_valid[:, :-1], axis=-1)
        elif self.cfg_aug['obj2scene_style'] == 'hard':
            objs_in_valid = np.all(pts_in_valid, axis=-1)
            
        objs_paste = [obj for obj, _valid in zip(objs_paste, objs_in_valid) if _valid]

        return objs_paste
    
    def apply_trans(self, points, T):
        """
        args:
            points: (N, 3) or (N, 4)
            T: trans matrix (4, 4)
        ret:
            points: (N, 3) or (N, 4)
        """
        is_hom = points.shape[1]==4
        if not is_hom: points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))

        points = points @ T.T

        if not is_hom: points = points[:, :-1]
        return points


    def get_trans(self, rx=0, ry=0, rz=0, tx=0, ty=0, tz=0):
        """
        rx, ry, rz: angle
        t_x, t_y, t_z: x, y, z translation
        """
        rx = rx / 180 * np.pi
        ry = ry / 180 * np.pi
        rz = rz / 180 * np.pi

        R_x = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
        
        R = R_x @ R_y @ R_z
        t = np.array([tx, ty, tz])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t
        
        return T

    def get_homo_trans(self, T):
        h, w = T.shape
        homo_T = np.eye(4)
        homo_T[:h, :w] = T
        return homo_T
    

    def roi_clip(self, roi, h, w):
        roi_range = np.array([[0, w-1], [0, h-1], [0, w-1], [0, h-1]])
        roi = np.clip(roi, roi_range[:, 0], roi_range[:, 1])
        return roi

    def roi_cut(self, raw_roi, roi, roi_patch=None, roi_mask=None):
        """
        parmas:
        raw_roi: roi before clip
        roi: roi after clip, may meet truc case and need to update the visable patch
        roi_patch, roi_mask : before clip
        rets:
        roi_patch, roi_mask : after clip
        """
        # raw_roi = raw_roi.astype(np.int64)
        if np.array_equal(raw_roi, roi):
            return roi_patch, roi_mask

        x1_r, y1_r, x2_r, y2_r = raw_roi
        x1, y1, x2, y2 = roi

        x1_c, x2_c, y1_c, y2_c = \
            x1-x1_r, x2-x1_r, y1-y1_r, y2-y1_r

        if roi_patch is not None: roi_patch = roi_patch[y1_c:y2_c+1, x1_c:x2_c+1, :]
        if roi_mask is not None: roi_mask = roi_mask[y1_c:y2_c+1, x1_c:x2_c+1]  

        return roi_patch, roi_mask


    def get_nlight(self, theta_yx, theta_xz):
        if isinstance(theta_yx, list):
            theta_yx = np.random.uniform(theta_yx[0], theta_yx[1])
        if isinstance(theta_xz, list):
            theta_xz = np.random.uniform(theta_xz[0], theta_xz[1])

        xz_r = np.sin(theta_yx / 180 * np.pi)
        nx = xz_r * np.cos(theta_xz / 180 * np.pi)
        nz = xz_r * np.sin(theta_xz / 180 * np.pi)
        ny = -np.sqrt(1 - nx**2 - nz**2)
        return np.array([nx, ny, nz], dtype=np.float32)


    def render_shadow(self, img, objs, calib, alpha=0.2, n_light=None, light_dir_sample_range=30, shadow_smooth=False, pose_T=None):
        if isinstance(alpha, list):
            alpha = np.random.uniform(alpha[0], alpha[1])

        if n_light is None:
            xz_r_max = np.sin(light_dir_sample_range / 180 * np.pi)
            xz_r = np.random.uniform(0, xz_r_max)
            theta = np.random.uniform(0, 2*np.pi)
            nx = xz_r * np.cos(theta)
            nz = xz_r * np.sin(theta)
            ny = -np.sqrt(1 - nx**2 - nz**2)
            n_light = np.array([nx, ny, nz], dtype=np.float32)

        if shadow_smooth:
            shadow_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            shadow_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        cam_plane = calib.layout['cam_plane']
        a, b, c, d = cam_plane
        n_plane = np.array([a, b, c], dtype=np.float32)

        all_pts_lidar = np.concatenate([obj.points for obj in objs], axis=0)
        n_pts_per_obj = [0] + [len(obj.points) for obj in objs]
        objs_pts_idx = np.cumsum(n_pts_per_obj)

        homo_V2C = self.get_homo_trans(calib.V2C)
        homo_R0 = self.get_homo_trans(calib.R0)
        T = pose_T @ homo_R0 @ homo_V2C
        all_pts_rect = self.apply_trans(all_pts_lidar, T)
        
        _t = -(all_pts_rect @ n_plane[:, None] + d) / (n_light[None, :] @ n_plane[:, None])     
        all_proj_rect = all_pts_rect + n_light * _t
        all_proj_img, _ = calib.rect_to_img(all_proj_rect)                                      # proj uv
        all_proj_img = np.round(all_proj_img).astype(np.int32)
        
        for i in range(len(objs)):
            if self.cfg_aug.get('shadow@fine_obj', False) and (objs[i].occlusion > 0 or objs[i].trucation > 0): continue
            proj_img = all_proj_img[objs_pts_idx[i]:objs_pts_idx[i+1]]
            if len(proj_img) != 0:
                convex_hull_pts = cv2.convexHull(proj_img).squeeze(1).astype(np.int32)
                if shadow_smooth:
                    cv2.fillConvexPoly(shadow_mask, convex_hull_pts, alpha) #, cv2.LINE_AA)
                else:
                    cv2.fillConvexPoly(shadow_mask, convex_hull_pts, 1) #, cv2.LINE_AA)

        if shadow_smooth:
            # shadow smooth
            shadow_v, shadow_u = np.where(shadow_mask!=1)
            if len(shadow_v) != 0:
                v_min, v_max, u_min, u_max = shadow_v.min(), shadow_v.max(), shadow_u.min(), shadow_u.max()
                shadow_mask[v_min:v_max+1, u_min:u_max+1] = cv2.GaussianBlur(shadow_mask[v_min:v_max+1, u_min:u_max+1], (5, 5), -1)
                _m = (shadow_mask!=1)
                img[_m] = (img[_m] * shadow_mask[_m][:,None]).astype(np.uint8)
        else:
            shadow_mask = shadow_mask.astype(np.bool_)
            img[shadow_mask] = (img[shadow_mask] * alpha).astype(np.uint8)

        
    def fill_instance(self, img, depth, mask, k_size=3, iter=1, edge_refine=True):
        _fg, filled_mask = self.get_filled_mask(mask, k_size=k_size, iter=iter, edge_refine=edge_refine)
        distance, nearest_idx = distance_transform_cdt(~mask, metric='cityblock', return_indices=True)              
        filled_pixel = np.argwhere(filled_mask)
        nearest_pixel = nearest_idx[:, filled_mask]
        img[filled_pixel[:, 0], filled_pixel[:, 1]] = img[nearest_pixel[0], nearest_pixel[1]]
        depth[filled_pixel[:, 0], filled_pixel[:, 1]] = depth[nearest_pixel[0], nearest_pixel[1]]
        return _fg


    def get_filled_mask(self, mask, k_size=3, iter=2, edge_refine=True):
        kernel = np.ones((k_size, k_size)) 

        _fg = mask.copy()
        for _ in range(iter):
            _mask = cv2.filter2D(_fg.astype(np.float32), -1, kernel)
            if edge_refine:
                _bg = (_mask==0).astype(np.float32)
                _fg = cv2.filter2D(_bg, -1, np.ones((k_size, k_size))) == 0
            else:
                _fg = (_mask!=0)

        filled_mask = ~mask & _fg
    
        return _fg, filled_mask


    def update_plane(self, calib, pose_T):
        # for plane ax+by+cz+d=0
        # cam plane: y=ax+cz+d   
        # lidar plane: z=ax+by+d
        a, b, c, d = calib.layout['cam_plane']
        xx, zz = np.meshgrid(
            np.linspace(0, 3, 4),
            np.linspace(0, 3, 4)
        )
        yy = (a * xx + c * zz + d) / (-b)
        cam_pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)    # (16, 3)
        cam_pts = self.apply_trans(cam_pts, pose_T)
        lidar_pts = calib.rect_to_lidar(cam_pts)

        la, lb, lc, ld = self.fit_plane(lidar_pts, method='ls')
        ca, cc, cb, cd = self.fit_plane(cam_pts[:,[0,2,1]], method='ls')

        # update
        calib.layout['lidar_plane'] = [la, lb, lc, ld]
        calib.layout['cam_plane'] = [ca, cb, cc, cd]

        return calib


    def fit_plane(self, points, method='ransac'):
        """
        fit ax + by + cz + d = 0
        set z = ax + by + d (c=-1)
        """
        if method == 'ransac':
            model = RANSACRegressor(base_estimator=None, min_samples=300, residual_threshold=0.01, random_state=0)
            model.fit(points[:, :2], points[:, 2])
            a, b = model.estimator_.coef_
            d = model.estimator_.intercept_     
            return [a, b, -1, d]
        elif method == 'ls':
            A = np.column_stack((points[:, :2], np.ones(len(points))))  # (x,y,1)
            b = points[:, 2:3]      # z
            x = np.linalg.inv(A.T @ A) @ A.T @ b
            x = x.squeeze()
            return [x[0], x[1], -1, x[2]]


    
    #========================== load data auxiliary function
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


    def get_scene(self, idx, use_dense_points=True, get_points=False, get_mask=True):
        # get data
        calib = self.get_calibration(idx, noobj=False)
        img = self.get_img(idx, noobj=False)
        h, w ,_ = img.shape
        mask = self.get_mask_sparse(idx).reshape(-1, 1, h, w) if get_mask else np.zeros((0, 1, h, w), dtype=np.bool_)
        dense_depth = self.get_dense_depth(idx)

        if get_points:
            if use_dense_points:
                points_bp_rect, u_idx, v_idx = self.depth_ops.get_depth_point_cloud(dense_depth, calib.P2, in_cam0_frame=True, ret_idx2d=True)
                points_bp_rect = points_bp_rect.T
                points = calib.rect_to_lidar(points_bp_rect)
            else:
                points = self.get_lidar(idx)[:,0:3]
        
        objs = self.get_label(idx)
        objs = [obj for obj in objs if (obj.cls_type != 'DontCare')]

        if get_points:
            if use_dense_points:
                return points, objs, calib, img, mask, dense_depth, u_idx, v_idx
            else:        
                return points, objs, calib, img, mask, dense_depth
        else:
            return objs, calib, img, mask, dense_depth
    

    def get_scene_empty(self, idx, expand_size=0.5, use_dense_points=True, get_points=False):
        # get data
        calib = self.get_calibration(idx, noobj=True)
        img = self.get_img(idx, noobj=True)
        h, w ,_ = img.shape
        mask = np.zeros((0, 1, h, w), dtype=np.bool_) 
        dense_depth = self.get_dense_depth(idx, noobj=True)

        if get_points:
            if use_dense_points:
                points_bp_rect, u_idx, v_idx = self.depth_ops.get_depth_point_cloud(dense_depth, calib.P2, in_cam0_frame=True, ret_idx2d=True)
                points_bp_rect = points_bp_rect.T
                points = calib.rect_to_lidar(points_bp_rect)
            else:
                points = self.get_lidar(idx)[:,0:3]

            if not use_dense_points:
                # move out obj-points
                objs = self.get_label(idx)
                # handle DontCare 
                objs = [obj for obj in objs if obj.cls_type != 'DontCare']

                lidar_plane = calib.layout['lidar_plane']    # [a, b, c, d]  ax+by+cz+d=0, c=-1, z=ax+by+d, for lidar
                a, b, c, d = lidar_plane

                for obj in objs:
                    obj.h += expand_size
                    obj.w += expand_size
                    obj.l += expand_size
                instance_mask, T_imloc2lidar, T_lidar2imloc = get_points_in_boxes(points, objs, calib, bev_box=True)
                for i in range(len(objs)):
                    ins_points = points[(instance_mask == i).squeeze()]
                    ins_points[:, 2] = a * ins_points[:, 0] + b*ins_points[:, 1] + d  
                    points[(instance_mask == i).squeeze()] = ins_points


        if get_points:
            if use_dense_points:
                return points, [], calib, img, mask, dense_depth, u_idx, v_idx
            else:        
                return points, [], calib, img, mask, dense_depth
        else:
            return [], calib, img, mask, dense_depth
    

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.data_dirs['trainval']['lidar_dir'], '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return points
    

    def get_img(self, idx, noobj=False):
        if noobj:
            img_file = os.path.join(self.data_dirs['trainval']['img_empty_dir'], '%06d_mask001.png' % idx)
        else:
            img_file = os.path.join(self.data_dirs['trainval']['img_dir'], '%06d.png' % idx)
        assert os.path.exists(img_file)
        img = cv2.imread(img_file)
        return img
    
    def get_dense_depth(self, idx, noobj=False):
        if noobj:
            dense_depth_file = os.path.join(self.data_dirs['trainval']['depth_G2_empty_dir'], '%06d.png' % idx)
        else:
            dense_depth_file = os.path.join(self.data_dirs['trainval']['depth_G2_dir'], '%06d.png' % idx)

        assert os.path.exists(dense_depth_file)
        dense_depth = self.depth_ops.read_depth_map(dense_depth_file)
        return dense_depth
    
    def get_mask_sparse(self, idx):
        mask_file = os.path.join(self.data_dirs['trainval']['mask_dir'], '%06d.npz' % idx)
        assert os.path.exists(mask_file)
        mask = load_npz(mask_file).toarray()
        return mask


    def get_calibration(self, idx, noobj=False):
        calib_file = os.path.join(self.data_dirs['trainval']['calib_dir'], '%06d.txt' % idx)
        if noobj:
            layout_file = os.path.join(self.data_dirs['trainval']['layout_noobj_dir'], '%06d.npz' % idx)
        else:
            layout_file = os.path.join(self.data_dirs['trainval']['layout_dir'], '%06d.npz' % idx)

        assert os.path.exists(calib_file)
        assert os.path.exists(layout_file)
        calib = Calibration(calib_file, layout_file)
        return calib

    def get_objects_from_label(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        objects = [Object3d(line) for line in lines]
        return objects

    def get_label(self, idx):
        label_file = os.path.join(self.data_dirs['trainval']['label_dir'], '%06d.txt' % idx)
        id_file = os.path.join(self.data_dirs['trainval']['id_dir'], '%06d.txt' % idx)
        assert os.path.exists(label_file)
        assert os.path.exists(id_file)
        objects = self.get_objects_from_label(label_file)
        
        with open(id_file) as f:
            ids = f.read().strip().split('\n')
            for (id, obj) in zip(ids, objects):
                obj.ID = int(id)
        return objects
    

    #========================== DONTCARE auxiliary utils
    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def inverse_rigid_trans(self, Tr):
        """ 
        Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
        """
        # inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr = np.eye(4)  # 4x4

        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def get_points_in_boxes(self, scene_points, objs, calib, bev_box=False):
        """
        params:
        scene_points: (n, 3)
        objs: kitti objs list len=k
        calib: scene calib info
        bev_box: 
        rets:
        instance_mask: (n, 1) background -1 ; 0~k-1 each object
        """
        num_objs = len(objs)
        instance_mask = np.ones((scene_points.shape[0], 1)) * -1
        T_imloc2lidar = []
        T_lidar2imloc = []
        for i, obj in enumerate(objs):
            # step 1
            corners = obj.generate_corners3d()
            corners = calib.rect_to_lidar(corners)
            x_max, x_min = np.max(corners[:,0]), np.min(corners[:,0])
            y_max, y_min = np.max(corners[:,1]), np.min(corners[:,1])
            
            # step 2
            roi_mask = (
                (scene_points[:, 0] < x_max) &
                (scene_points[:, 0] > x_min) &
                (scene_points[:, 1] < y_max) &
                (scene_points[:, 1] > y_min))
            roi_points = scene_points[roi_mask]
            roi_indices = np.where(roi_mask)[0]

            # step 3
            R = np.array([[np.cos(obj.ry), 0, np.sin(obj.ry)],
                        [0, 1, 0],
                        [-np.sin(obj.ry), 0, np.cos(obj.ry)]])
            T0 = np.eye(4)
            T0[0:3, 0:3] = R
            T0[0:3, 3] = obj.pos

            T1 = np.eye(4)
            T1[0:3, 0:3] = calib.R0.T

            T2 = np.eye(4)
            T2[0:3] = calib.C2V

            T0_inv = self.inverse_rigid_trans(T0)
            T1_inv = self.inverse_rigid_trans(T1)
            T2_inv = self.inverse_rigid_trans(T2)

            T = T2 @ T1 @ T0
            T_inv = T0_inv @ T1_inv @ T2_inv
            # p_lidar = T2@T1@T0@p_image_local  p_image_local = inv(T0)@inv(T1)@inv(T2)@p_lidar
            roi_points_hom = self.cart_to_hom(roi_points)
            roi_points_imloc = (roi_points_hom @ T_inv.T)[:, 0:3]

            T_imloc2lidar.append(T)
            T_lidar2imloc.append(T_inv)

            # step 4
            if bev_box:
                inbox_mask = (
                    (roi_points_imloc[:,0] < obj.l/2) &
                    (roi_points_imloc[:,0] > -obj.l/2) &
                    (roi_points_imloc[:,2] < obj.w/2) &
                    (roi_points_imloc[:,2] > -obj.w/2))
            else:
                inbox_mask = (
                    (roi_points_imloc[:,0] < obj.l/2) &
                    (roi_points_imloc[:,0] > -obj.l/2) &
                    (roi_points_imloc[:,1] < 0) &
                    (roi_points_imloc[:,1] > -obj.h) &
                    (roi_points_imloc[:,2] < obj.w/2) &
                    (roi_points_imloc[:,2] > -obj.w/2))
            instance_indices = roi_indices[inbox_mask]

            # step 5
            instance_mask[instance_indices] = i

        return instance_mask, T_imloc2lidar, T_lidar2imloc

    def filter_outside_objs(self, objs_paste, dst_calib):
        """
        args:
        objs_paste: list of n src sampled objs
        dst_calib: calib of dst scene, including valid_map
        ret:
        objs_paste_valid: objs located in valid area
        """
        valid_map = dst_calib.layout['valid_map'] == 1
        h, w = valid_map.shape
        x_range = dst_calib.layout['x_range']
        y_range = dst_calib.layout['y_range']
        voxel_size = dst_calib.layout['voxel_size']

        objs_paste_valid = []
        for i, obj in enumerate(objs_paste):
            corners_lidar = obj.generate_corners3d_lidar(dst_calib)[:4, :2]   # (4, 2) corners in lidar bev
            corners_grid = np.zeros((4, 2), dtype=np.int64)       # judge corners locate in which grid at valid map
            # r,c <-> -x,-y
            corners_grid[:, 0] = h - 1 - ((corners_lidar[:, 0] - x_range[0]) / voxel_size[0]).astype(np.int64) 
            corners_grid[:, 1] = w - 1 - ((corners_lidar[:, 1] - y_range[0]) / voxel_size[1]).astype(np.int64)
            
            if (True in (corners_grid < np.array([[0, 0]]))) or (True in (corners_grid >= np.array([[h, w]]))):
                # 'real outside'
                continue

            corners_in_valid = valid_map[corners_grid[:,0], corners_grid[:,1]]  # (4,)
            if not False in corners_in_valid:
                objs_paste_valid.append(obj)
        
        return objs_paste_valid

    def _SAT_check(self, src_polygon, dst_polygon, is_rect=False):
        """
        Separating Axis Theorem for polygon collision detection
        params:
        src_polygon: (n, 2) for n vertices
        dst_polygon: (m, 2) for m vertices
        is_rect: if the two polygon is rectangle, it is easier
        ret:
        bool: True for collision

        reference: https://github.com/JuantAldea/Separating-Axis-Theorem/blob/master/python/separation_axis_theorem.py
        """
        # get edges (n+m, 2)
        edges = np.concatenate([
            np.roll(src_polygon, shift=-1, axis=0) - src_polygon,
            np.roll(dst_polygon, shift=-1, axis=0) - dst_polygon,
        ], axis=0)
        # get axes(normal of edges) (n+m, 2)
        axes = np.copy(edges)[:, ::-1]
        axes[:, 1] *= -1
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)

        # projection
        src_proj = axes @ src_polygon.T     # (n+m, n)
        dst_proj = axes @ dst_polygon.T     # (n+m, m)

        # find min max proj
        src_min_proj = np.min(src_proj, axis=1)
        src_max_proj = np.max(src_proj, axis=1)
        dst_min_proj = np.min(dst_proj, axis=1)
        dst_max_proj = np.max(dst_proj, axis=1)

        # judge if src and dst overlap at certain axis
        overlap = ((src_min_proj <= dst_max_proj) & (dst_min_proj <= src_max_proj))

        # only if every axis meet collision, the two polygon collision
        return not (False in overlap)

    def box_collision_check(self, src_objs, dst_objs, use_sp=False):
        """
        params:
        src_objs: n
        dst_objs: m
        use_sp: if use sparse mtx for collision_mtx
        ret:
        collision_mtx: (n, m) bool array, True for collision
        """
        #===== Step 1 : generate bev boxes =====#
        src_boxes = np.array(
            [obj.generate_corners3d()[[0,1,2,3], :][:, [0,2]] for obj in src_objs])     # (n, 4, 2)
        dst_boxes = np.array(
            [obj.generate_corners3d()[[0,1,2,3], :][:, [0,2]] for obj in dst_objs])     # (m, 4, 2)

        #===== Step 2 : Separating Axis Theorem for collision detection =====#
        n = src_boxes.shape[0]
        m = dst_boxes.shape[0]
        if use_sp:
            collision_mtx = csr_matrix((n, m), dtype=bool)
        else:
            collision_mtx = np.zeros((n, m), dtype=np.bool_)

        for i in range(n):
            for j in range(m):
                collision_mtx[i, j] = self._SAT_check(src_boxes[i], dst_boxes[j])

        return collision_mtx



class PhotometricAugmenter():
    def __init__(self):
        # color augmentation param (follow by monodistill)
        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self.eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

    def scene_aug(self, img, ret_norm_img=False):
        """
        img: BGR uint8 (h,w,3)
        """
        # random brightness, contrast, saturation        
        img = img.astype(np.float32) / 255
        self._scene_aug(img)
        if ret_norm_img:
            return img
        img = (img * 255).astype(np.uint8)

        return img
    
    def ins_aug(self, img, hsv_delta=[90, 128, 30], thres=30):
        img_aug = img.copy()
        img_aug_hsv = self.convert_color_factory('bgr', 'hsv')(img_aug)
        aug_mask = (img_aug_hsv[..., 2] < thres) | (img_aug_hsv[..., 2] > (255-thres))   

        for i in range(3):
            aug = int(np.random.uniform(-hsv_delta[i], hsv_delta[i]))
            if i == 0:
                img_aug_hsv[..., 0] = img_aug_hsv[..., 0] + aug
                img_aug_hsv[..., 0][img_aug_hsv[..., 0] >= 180] = img_aug_hsv[..., 0][img_aug_hsv[..., 0] >= 180] - 180
                img_aug_hsv[..., 0][img_aug_hsv[..., 0] < 0] = img_aug_hsv[..., 0][img_aug_hsv[..., 0] < 0] + 180
            else:
                img_aug_hsv[..., i] = img_aug_hsv[..., i] + aug
                if i == 1:
                    img_aug_hsv[..., i][img_aug_hsv[..., i] >= 255] = img_aug_hsv[..., i][img_aug_hsv[..., i] >= 255] - 255
                    img_aug_hsv[..., i][img_aug_hsv[..., i] < 0] = img_aug_hsv[..., i][img_aug_hsv[..., i] < 0] + 255
                elif i == 2:
                    img_aug_hsv[..., i] = np.clip(img_aug_hsv[..., i], thres, 255-thres)

        img_aug = self.convert_color_factory('hsv', 'bgr')(img_aug_hsv)
        img_aug[aug_mask] = img[aug_mask].copy()
        
        return img_aug

    def convert_color_factory(self, src, dst):
        code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')
        def convert_color(img):
            out_img = cv2.cvtColor(img, code)
            return out_img
        return convert_color
    
    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def lighting_(self, img, alphastd):
        alpha = self.data_rng.normal(scale=alphastd, size=(3, ))
        img += np.dot(self.eig_vec, self.eig_val * alpha)

    def blend_(self, alpha, image1, image2):
        image1 *= alpha
        image2 *= (1 - alpha)
        image1 += image2

    def saturation_(self, img, gs, gs_mean, var):
        alpha = 1. + self.data_rng.uniform(low=-var, high=var)
        self.blend_(alpha, img, gs[:, :, None])

    def brightness_(self, img, gs, gs_mean, var):
        alpha = 1. + self.data_rng.uniform(low=-var, high=var)
        img *= alpha

    def contrast_(self, img, gs, gs_mean, var):
        alpha = 1. + self.data_rng.uniform(low=-var, high=var)
        self.blend_(alpha, img, gs_mean)

    def _scene_aug(self, img):
        functions = [self.brightness_, self.contrast_, self.saturation_]
        random.shuffle(functions)

        gs = self.grayscale(img)
        gs_mean = gs.mean()
        for f in functions:
            f(img, gs, gs_mean, 0.4)
        self.lighting_(img, 0.1)


class DepthOps():
    def __init__(self):
        pass
        
    def project_pc_to_image(self, point_cloud, cam_p):
        """Projects a 3D point cloud to 2D points

        Args:
            point_cloud: (3, N) point cloud
            cam_p: camera projection matrix

        Returns:
            pts_2d: (2, N) projected coordinates [u, v] of the 3D points
        """

        pc_padded = np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0)
        pts_2d = np.dot(cam_p, pc_padded)

        pts_2d[0:2] = pts_2d[0:2] / pts_2d[2]
        return pts_2d[0:2]
    
    def read_depth_map(self, depth_map_path):
        depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
        depth_map = depth_image / 256.0

        # Discard depths less than 10cm from the camera
        depth_map[depth_map < 0.1] = 0.0

        return depth_map.astype(np.float32)
    
    def save_depth_map(self, save_path, depth_map,
                       version='cv2', png_compression=3):
        """Saves depth map to disk as uint16 png

        Args:
            save_path: path to save depth map
            depth_map: depth map numpy array [h w]
            version: 'cv2' or 'pypng'
            png_compression: Only when version is 'cv2', sets png compression level.
                A lower value is faster with larger output,
                a higher value is slower with smaller output.
        """

        # Convert depth map to a uint16 png
        depth_image = (depth_map * 256.0).astype(np.uint16)

        if version == 'cv2':
            ret = cv2.imwrite(save_path, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

            if not ret:
                raise RuntimeError('Could not save depth map')

        elif version == 'pypng':
            with open(save_path, 'wb') as f:
                depth_image = (depth_map * 256.0).astype(np.uint16)
                writer = png.Writer(width=depth_image.shape[1],
                                    height=depth_image.shape[0],
                                    bitdepth=16,
                                    greyscale=True)
                writer.write(f, depth_image)

        else:
            raise ValueError('Invalid version', version)
        
    def get_depth_point_cloud(self, depth_map, cam_p, min_v=0, flatten=True, in_cam0_frame=True, ret_idx2d=False):
        """Calculates the point cloud from a depth map given the camera parameters

        Args:
            depth_map: depth map
            cam_p: camera p matrix
            min_v: amount to crop off the top
            flatten: flatten point cloud to (3, N), otherwise return the point cloud
                in xyz_map (3, H, W) format. (H, W, 3) points can be retrieved using
                xyz_map.transpose(1, 2, 0)
            in_cam0_frame: (optional) If True, shifts the point cloud into cam_0 frame.
                If False, returns the point cloud in the provided camera frame
            ret_idx2d: If True, return the point proj pos in img (u, v)

        Returns:
            point_cloud: (3, N) point cloud
        """

        depth_map_shape = depth_map.shape[0:2]

        if min_v > 0:
            # Crop top part
            depth_map[0:min_v] = 0.0

        xx, yy = np.meshgrid(
            np.linspace(0, depth_map_shape[1] - 1, depth_map_shape[1]),
            np.linspace(0, depth_map_shape[0] - 1, depth_map_shape[0]))

        # Calibration centre x, centre y, focal length
        centre_u = cam_p[0, 2]
        centre_v = cam_p[1, 2]
        focal_length = cam_p[0, 0]

        i = xx - centre_u
        j = yy - centre_v

        # Similar triangles ratio (x/i = d/f)
        ratio = depth_map / focal_length
        x = i * ratio
        y = j * ratio
        z = depth_map

        if in_cam0_frame:
            # Return the points in cam_0 frame
            # Get x offset (b_cam) from calibration: cam_p[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[0, 3] / focal_length

            valid_pixel_mask = depth_map > 0
            x[valid_pixel_mask] += x_offset

        # Return the points in the provided camera frame
        point_cloud_map = np.asarray([x, y, z])

        if flatten:
            point_cloud = np.reshape(point_cloud_map, (3, -1))
            if ret_idx2d:
                return point_cloud.astype(np.float32), xx.astype(np.int32).reshape(-1), yy.astype(np.int32).reshape(-1)
            else:
                return point_cloud.astype(np.float32)
        else:
            return point_cloud_map.astype(np.float32)
        
    def get_point_colors(self, points, cam_p, image):
        """
        points: (N, 3) in rect coord
        """
        points_in_im = self.project_pc_to_image(points.T, cam_p)
        points_in_im_rounded = np.round(points_in_im).astype(np.int32)

        point_colors = image[points_in_im_rounded[1], points_in_im_rounded[0]]

        return point_colors

    def project_depths(self, point_cloud, cam_p, image_shape, max_depth=255.0, use_jit=True):
        """Projects a point cloud into image space and saves depths per pixel.

        Args:
            point_cloud: (3, N) Point cloud in cam0
            cam_p: camera projection matrix
            image_shape: image shape [h, w]
            max_depth: optional, max depth for inversion

        Returns:
            projected_depths: projected depth map
        """

        # Only keep points in front of the camera
        all_points = point_cloud.T

        # Save the depth corresponding to each point
        points_in_img = self.project_pc_to_image(all_points.T, cam_p)
        points_in_img_int = np.int32(np.round(points_in_img))

        # Remove points outside image
        valid_indices = \
            (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
            (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

        all_points = all_points[valid_indices]
        points_in_img_int = points_in_img_int[:, valid_indices]

        # Invert depths
        all_points[:, 2] = max_depth - all_points[:, 2]

        # Only save valid pixels, keep closer points when overlapping
        projected_depths = np.zeros(image_shape)
        valid_indices = [points_in_img_int[1], points_in_img_int[0]]
        
        if use_jit:
            projected_depths = jit_z_buffer(projected_depths, points_in_img_int, all_points)
        else:
            projected_depths[valid_indices] = [
                max(projected_depths[
                    points_in_img_int[1, idx], points_in_img_int[0, idx]],
                    all_points[idx, 2])
                for idx in range(points_in_img_int.shape[1])]

        projected_depths[valid_indices] = \
            max_depth - projected_depths[valid_indices]

        return projected_depths.astype(np.float32)
    

@numba.jit(nopython=True)
def jit_z_buffer(projected_depths, points_in_img_int, all_points):
    for idx in range(points_in_img_int.shape[1]):
        row_idx = points_in_img_int[1, idx]
        col_idx = points_in_img_int[0, idx]
        current_depth = projected_depths[row_idx, col_idx]
        new_depth = max(current_depth, all_points[idx, 2])
        projected_depths[row_idx, col_idx] = new_depth
    return projected_depths
    


if __name__ == '__main__':
    pass