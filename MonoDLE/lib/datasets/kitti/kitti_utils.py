''' some auxiliary functions for KITTI dataset '''
import numpy as np
import cv2
import random
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
import math

def get_img(img_dir, idx):
    img_file = os.path.join(img_dir, '%06d.png' % idx)
    assert os.path.exists(img_file)
    img = cv2.imread(img_file)
    return img

def get_img_noobj(img_dir, idx):
    img_file = os.path.join(img_dir, '%06d_mask001.png' % idx)
    assert os.path.exists(img_file)
    img = cv2.imread(img_file)
    return img

def get_mask(mask_dir, idx):
    mask_file = os.path.join(mask_dir, '%06d.npy' % idx)
    assert os.path.exists(mask_file)
    mask = np.load(mask_file, allow_pickle=True)
    return mask

def get_mask_sparse(mask_dir, idx):
    mask_file = os.path.join(mask_dir, '%06d.npz' % idx)
    assert os.path.exists(mask_file)
    mask = load_npz(mask_file).toarray()
    return mask

def get_calibration_old(calib_dir, road_plane_dir, idx):
    calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
    road_plane_file = os.path.join(road_plane_dir, '%06d.txt' % idx)
    assert os.path.exists(calib_file)
    assert os.path.exists(road_plane_file)
    calib = Calibration(calib_file, road_plane_file)
    return calib
    
def get_calibration(calib_dir, layout_dir, idx):
    calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
    layout_file = os.path.join(layout_dir, '%06d.npz' % idx)
    assert os.path.exists(calib_file)
    assert os.path.exists(layout_file)
    calib = Calibration(calib_file, layout_file)
    return calib

def get_lidar(lidar_dir, idx):
    lidar_file = os.path.join(lidar_dir, '%06d.bin' % idx)
    assert os.path.exists(lidar_file)
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    return points

def get_pillar(pillar_dir, idx, format='pkl'):
    assert format in ['pkl', 'npz']
    if format == 'pkl':
        pillar_file = os.path.join(pillar_dir, '%06d.pkl' % idx)
        assert os.path.exists(pillar_file)
        with open(pillar_file, 'rb') as f:
            pillar = pickle.load(f)
    elif format == 'npz':
        pillar_file = os.path.join(pillar_dir, '%06d.npz' % idx)
        assert os.path.exists(pillar_file)
        pillar = np.load(pillar_file, allow_pickle=True)
    
    pillar_info_file = os.path.join(pillar_dir, 'pillar_info.npz')
    assert os.path.exists(pillar_info_file)
    
    pillar_info = np.load(pillar_info_file, allow_pickle=True)
    
    return pillar, pillar_info


def get_label(label_dir, id_dir, idx):
    label_file = os.path.join(label_dir, '%06d.txt' % idx)
    id_file = os.path.join(id_dir, '%06d.txt' % idx)
    assert os.path.exists(label_file)
    assert os.path.exists(id_file)
    objects = get_objects_from_label(label_file)
    with open(id_file) as f:
        ids = f.read().strip().split('\n')
        for (id, obj) in zip(ids, objects):
            obj.ID = int(id)
    return objects


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


class Object3d(object):
    def __init__(self, line, is_aug=False):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        if not is_aug:
            self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()
        # sample id in dataset
        if is_aug:
            assert label.__len__() == 16
            self.ID = int(label[15])
        else:
            self.ID = -1
        
        self.depth_bin = self.get_depth_bins(self.pos[-1], mode='LID', 
                                             depth_min=2, depth_max=65, num_bins=80)
        # instance pointcloud
        self.points = None                  # sparse LiDAR points in 3Dbox
        self.dense_points_mask2d = None     # dense points from image-depth
        # dense points color
        self.dp_mask2d_color = None
        # object roi
        self.roi = None
        # object roi mask
        self.roi_mask = None
        self.paste_flag = None

        self.occ_ratio = 0.3 * self.occlusion

        self.corners3d_rect = self.generate_corners3d()


    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4


    def get_depth_bins(self, depth, mode, depth_min, depth_max, num_bins):
        """
        Converts depth map into bin_idx
        Args:
            depth : continious depth values
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
        Returns:
            bin_idx : Depth  bin_idx
        """
        if depth < depth_min:
            return -1
        if depth > depth_max:
            return -2

        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            bin_idx = ((depth - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            bin_idx = -0.5 + 0.5 * np.sqrt(1 + 8 * (depth - depth_min) / bin_size)
        elif mode == "SID":
            bin_idx = num_bins * (np.log(1 + depth) - np.log(1 + depth_min)) / \
                (np.log(1 + depth_max) - np.log(1 + depth_min))
        else:
            raise NotImplementedError

        return int(bin_idx)


    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d


    def update_corners3d(self, pose):
        pts_homo = np.hstack((self.corners3d_rect, np.ones((self.corners3d_rect.shape[0], 1), dtype=np.float32)))
        pts_homo = pts_homo @ pose.T
        self.corners3d_rect = pts_homo[:, :-1]


    def generate_corners3d_lidar(self, calib):
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        
        T0 = np.eye(4)
        T0[0:3, 0:3] = R
        T0[0:3, 3] = self.pos

        T1 = np.eye(4)
        T1[0:3, 0:3] = calib.R0.T

        T2 = np.eye(4)
        T2[0:3] = calib.C2V

        T = T2 @ T1 @ T0

        corners3d = calib.cart_to_hom(corners3d.T)
        corners3d = (corners3d @ T.T)[:, 0:3]

        return corners3d
    

    def gen_corners3d_2dproj(self, calib):
        # corners3d_rect = self.generate_corners3d()
        corners3d_proj, _ = calib.rect_to_img(self.corners3d_rect)
        box2d_proj = np.array([np.min(corners3d_proj[:,0]), 
                               np.min(corners3d_proj[:,1]), 
                               np.max(corners3d_proj[:,0]), 
                               np.max(corners3d_proj[:,1])])
        return corners3d_proj, box2d_proj

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d


    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str


    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str



###################  calibration  ###################

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

def get_road_plane_from_file(road_plane_file):
    with open(road_plane_file) as f:
        lines = f.readlines()
    obj = lines[-1].strip().split(' ')
    plane = np.array(obj, dtype=np.float32)
    return plane

def get_layout_from_file(layout_file):
    layout = np.load(layout_file, allow_pickle=True)
    layout = {key:layout[key] for key in layout.files}
    return layout



class UniIntrinsic():
    def __init__(self, uni_size=[384, 1280]):
        self.uni_h = uni_size[0]
        self.uni_w = uni_size[1]

        #======= to hard
        # self.P2 = np.array([
        #     [ 7.215377e+02, 0.000000e+00, self.uni_w/2, 4.485728e+01],
        #     [ 0.000000e+00, 7.215377e+02, self.uni_h/2, 2.163791e-01],
        #     [ 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])

        # self.cu = self.P2[0, 2]
        # self.cv = self.P2[1, 2]
        # self.fu = self.P2[0, 0]
        # self.fv = self.P2[1, 1]
        # self.tx = self.P2[0, 3] / (-self.fu)
        # self.ty = self.P2[1, 3] / (-self.fv)

        #======== soft implementation (only considerate focal and centeroid)
        self.fu = 721.53771973
        self.fv = 721.53771973
        self.cu = 609.55932617 / 1242 * self.uni_w
        self.cv = 172.85400391 / 375 * self.uni_h


    def __call__(self, calib, h=None, w=None, items=[], ret_affine=False):
        """
        trans all into uni_intrinsic
        calib: raw calib
        h, w: raw size
        items: img / depth / objs / (masks, dontcare)
        """
        assert type(calib) == Calibration
        assert type(items) == list

        _fu, _fv, _cu, _cv = calib.fu, calib.fv, calib.cu, calib.cv

        # update calib into unified intrinsic
        # calib.P2 = self.P2
        calib.fu, calib.fv, calib.cu, calib.cv = self.fu, self.fv, self.cu, self.cv
        calib.P2[0, 0], calib.P2[1, 1], calib.P2[0, 2], calib.P2[1, 2] = calib.fu, calib.fv, calib.cu, calib.cv
        calib.P2[0, 3] = calib.tx * (-calib.fu)
        calib.P2[1, 3] = calib.ty * (-calib.fv)

        # get affine transformation
        s_fu = self.fu / _fu
        s_fv = self.fv / _fv

        src_pts = np.array(
            [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

        # self.cu - _cu*s_fu, self.cu + (w-_cu)*s_fu
        # self.cv - _cv*s_fv, self.cv + (h-_cv)*s_fv
        dst_pts = np.array([
            [-_cu*s_fu, -_cv*s_fv],
            [(w-_cu)*s_fu, -_cv*s_fv],
            [(w-_cu)*s_fu,(h-_cv)*s_fv],
            [-_cu*s_fu, (h-_cv)*s_fv]
        ])
        dst_pts = (np.tile(np.array([self.cu, self.cv]), (4, 1)) + dst_pts).astype(np.float32)
        
        affine_T = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])     # (2, 3)

        if len(items) == 0:
            if ret_affine:
                return calib, affine_T
            else:
                return calib

        # update items from raw intrinsic to unified intrinsic
        uni_items = []
        for item in items:
            if isinstance(item, np.ndarray):
                # image / depth
                _item = cv2.warpAffine(item, affine_T, (self.uni_w, self.uni_h))
                uni_items.append(_item)
            elif isinstance(item, list):
                # objs
                for i in range(len(item)):
                    box2d_hom = np.array([[item[i].box2d[0], item[i].box2d[2]],
                                        [item[i].box2d[1], item[i].box2d[3]],
                                        [1, 1]])
                    box2d = (affine_T @ box2d_hom).T.reshape(-1)
                    item[i].box2d = box2d

                    if item[i].roi is not None:
                        roi_hom = np.array([[item[i].roi[0], item[i].roi[2]],
                                            [item[i].roi[1], item[i].roi[3]],
                                            [1, 1]])
                        roi = (affine_T @ roi_hom).T.reshape(-1)
                        item[i].roi = roi
                uni_items.append(item)

        if ret_affine:
            return calib, uni_items, affine_T
        else:
            return calib, uni_items




class Calibration(object):
    def __init__(self, calib_file, layout_file=None):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
        self.C2V = self.inverse_rigid_trans(self.V2C)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

        if layout_file is not None:
            # road plane [a, b, c, d] for plane parameters
            # layout = {'x_range','y_range','z_range','voxel_size','lidar_plane','cam_plane','valid_map'}
            self.layout = get_layout_from_file(layout_file)


    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        pts_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_rect)))
        pts_ref = self.cart_to_hom(pts_ref)  # nx4
        return np.dot(pts_ref, np.transpose(self.C2V))

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth


    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)

        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

        return ry

    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)

        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return alpha



###################  affine trainsform  ###################

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# color aug
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


# more for debug
def angle2class(angle, num_heading_bin=12):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


if __name__ == '__main__':
    import copy
    calib = Calibration('/data/KITTI/object/training/calib/000010.txt',
                        '/data/KITTI/object/training/layout_0.5/000010.npz')
    
    img = cv2.imread('/data/KITTI/object/training/image_2/000010.png')
    
    h, w, _ = img.shape
    std_size=[384, 1280]
    item = copy.deepcopy(calib)

    _cu, _cv = item.cu / w * std_size[1], item.cv / h * std_size[0]
    offset = [_cu - item.cu, _cv - item.cv]
    item.P2[0, 2] = _cu
    item.P2[1, 2] = _cv
    item.cu = _cu
    item.cv = _cv

    print('debug')
