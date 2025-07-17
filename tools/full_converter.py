import mmcv
import numpy as np

# info_path = './full_nuscenes/full_nuscenes_infos_val.pkl'
train_info_path = './full_nuscenes/full_nuscenes_infos_train.pkl'
output_path = './full_nuscenes/full_nuscenes_infos_train_with_proj.pkl'
infos_data = mmcv.load(train_info_path)
infos = infos_data['infos']

for info in infos:
    for cam, cam_info in info['cams'].items():
        lidar2cam_rot = np.array(cam_info['sensor2lidar_rotation']).T
        lidar2cam_trans = -lidar2cam_rot @ np.array(cam_info['sensor2lidar_translation'])
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_rot
        lidar2cam_rt[:3, 3] = lidar2cam_trans

        intrinsic = np.array(cam_info['cam_intrinsic'])
        proj = np.eye(4)
        proj[:3, :3] = intrinsic
        lidar2image = proj @ lidar2cam_rt

        sensor2lidar_rot = np.array(cam_info['sensor2lidar_rotation'])
        sensor2lidar_trans = np.array(cam_info['sensor2lidar_translation'])

        lidar2camera = np.eye(4)
        lidar2camera[:3, :3] = lidar2cam_rot
        lidar2camera[:3, 3] = lidar2cam_trans

        camera2lidar = np.eye(4)
        camera2lidar[:3, :3] = sensor2lidar_rot
        camera2lidar[:3, 3] = sensor2lidar_trans

        cam_info['lidar2image_matrix'] = lidar2image.tolist()
        cam_info['lidar2camera'] = lidar2camera.tolist()
        cam_info['camera2lidar'] = camera2lidar.tolist()
        cam_info['img_aug_matrix'] = np.eye(4).tolist()

mmcv.dump(dict(infos=infos, metadata=dict(version='v1.0-trainval')), output_path)