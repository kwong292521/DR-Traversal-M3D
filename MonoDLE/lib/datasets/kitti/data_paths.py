import os

trainval_data_dir = './data/trainval'
testset_dir = './data/test'

data_dirs = {
    'trainval':{
        'img_dir' : os.path.join(trainval_data_dir, 'image_2'),
        'img_empty_dir' : os.path.join(trainval_data_dir, 'kitti_scenes_empty'),
        'calib_dir' : os.path.join(trainval_data_dir, 'calib'),
        'label_dir' : os.path.join(trainval_data_dir, 'label_2'),
        'id_dir' : os.path.join(trainval_data_dir, 'objs_ID'),
        'mask_dir' : os.path.join(trainval_data_dir, 'mask_rect'),
        'lidar_dir' : os.path.join(trainval_data_dir, 'velodyne'),
        'depth_G2_dir' : os.path.join(trainval_data_dir, 'depth_2_G2_rect'),
        'depth_G2_empty_dir' : os.path.join(trainval_data_dir, 'depth_2_empty_G2_rect_filled'),
        'layout_dir' : os.path.join(trainval_data_dir, 'layout_0.5_merge'),
        'layout_noobj_dir' : os.path.join(trainval_data_dir, 'layout_noobj_0.5_merge_filter'),
        
        'split_dir' : os.path.join(trainval_data_dir, 'ImageSets'),

        'instance_db_file' : os.path.join(trainval_data_dir, 'kitti_instance_database_tiny.pkl'),
        
        'ped_scenes_file': os.path.join(trainval_data_dir, 'ped_scenes.txt'),
        'sparse_objs_ID_file': os.path.join(trainval_data_dir, 'sparse_objs_ID.txt'),
        'nobjs_ratio_train_idx_file': os.path.join(trainval_data_dir, 'nobjs_ratio_train_idx.pkl'),
    },
    'test':{
        'img_dir': os.path.join(testset_dir, 'image_2'),
        'calib_dir': os.path.join(testset_dir, 'calib')
    }
}

n_trainval_scenes = 7481
n_train_scenes = 3712

import builtins  
def get_path_vars():
    global_vars = globals()  
    library_names = dir(builtins)  

    _vars = {var: value for var, value in global_vars.items() if var not in library_names and not var.startswith("__")}
    del _vars['os']
    del _vars['builtins']
    del _vars['get_path_vars']

    return _vars

path_dict = get_path_vars()

