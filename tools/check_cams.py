import pickle
from mmdet3d.datasets import build_dataset
import mmcv, numpy as np
from mmcv import Config
# 🔄 NuScenes info pkl 파일 경로
info= './data/nuscenes/nuscenes_infos_val.pkl'  # ← 경로를 실제 사용 중인 것으로 변경
# info = mmcv.load('./full_nuscenes/full_nuscenes_infos_val_with_proj.pkl')
# cfg = Config.fromfile('configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml')
# cfg.data.val.ann_file = './full_nuscenes/full_nuscenes_infos_val.pkl'  # ← 경로를 실제 사용 중인 것으로 변경
# ds = build_dataset(cfg.data.val)
# _ = ds[0]

# print(f"# Samples: {len(info['infos'])}")
first = info['infos'][0]
print("✅ First sample keys:", first.keys())

# required_keys = [
#     'lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation',
#     'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp'
# ]
# missing = [k for k in required_keys if k not in first]
# print("✅ Missing required keys:", missing if missing else "None")

print("\n✅ First CAM_FRONT keys:", first['cams']['CAM_FRONT'].keys())