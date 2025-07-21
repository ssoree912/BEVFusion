#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract camera-only BEV Q/K attention maps from BEVFusion
Usage:
    nohup python tools/extract_camera_qk_attention.py \
        --cfg configs/nuscenes/det/qk/camera.yaml \
        --info full_nuscenes/full_nuscenes_infos_val_with_proj.pkl \
        --weight pretrained/camera-only-det.pth \
        --out_dir results/qk_attn/camera > logs/camera_fpn_qk.log 2>&1 &
"""

import os, argparse, math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from torch.utils.data import DataLoader
import logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir/'extract_camera_qk_attention.log'),
        logging.StreamHandler()
    ]
)

# --- 1) argument ---
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',      type=str, required=True)
parser.add_argument('--info',     type=str, required=True)
parser.add_argument('--weight',   type=str, required=True)
parser.add_argument('--out_dir',  type=str, required=True)
parser.add_argument('--debug',    action='store_true')
args = parser.parse_args()

# --- 2) camera-only pipeline & dataset ---
try:
    from mmdet.datasets.builder import PIPELINES
except ImportError:
    # 만약 MMDet3D 용으로 분리된 경우
    from mmdet3d.datasets.builder import PIPELINES

@PIPELINES.register_module()
class FixCamLidarMeta:
    def __call__(self, results):
        # Fix camera-lidar metadata keys to ensure consistent naming
        if 'camera2lidar' in results:
            results['cam2lidar'] = results.pop('camera2lidar')
        if 'lidar2camera' in results:
            results['lidar2cam'] = results.pop('lidar2camera')
        return results

@PIPELINES.register_module()
class ImageToNumpy:
    def __call__(self, results):
        if 'img' in results:
            results['img'] = [np.asarray(img, dtype=np.float32) for img in results['img']]
        return results

@PIPELINES.register_module()
class StackMVNormalize:
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean if mean is not None else [123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array(std if std is not None else [58.395, 57.12, 57.375], dtype=np.float32)

    def __call__(self, results):
        imgs = results.get('img', [])
        norm_imgs = [(img.astype(np.float32) - self.mean) / self.std for img in imgs]
        results['img'] = norm_imgs
        return results

@PIPELINES.register_module()
class PackCameraMeta:
    def __call__(self, results):
        # Pack camera metadata into a DataContainer for batch processing
        meta_keys = [
            'camera_intrinsics', 'camera2ego', 'cam2lidar',
            'lidar2cam', 'lidar2image', 'img_aug_matrix',
            'lidar2ego', 'timestamp'
        ]
        metas = {}
        for key in meta_keys:
            if key in results:
                metas[key] = results[key]
        results['camera_meta'] = DataContainer(metas, cpu_only=True)
        return results

camera_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='FixCamLidarMeta'),
    dict(type='ImageToNumpy'),
    dict(type='StackMVNormalize'),
    dict(type='Collect3D',
         keys=['img'],
         meta_keys=[
             'camera_intrinsics','camera2ego','camera2lidar',
             'lidar2camera','lidar2image','img_aug_matrix',
             'lidar2ego','timestamp'
         ]),
    dict(type='PackCameraMeta'),
]

simple_dataset = dict(
    type         = 'Custom3DDataset',
    dataset_root = '',
    ann_file     = args.info,
    pipeline     = camera_pipeline,
    modality     = dict(use_camera=True, use_lidar=False,
                        use_radar=False, use_map=False, use_external=False),
    test_mode    = True,
)

cfg = Config.fromfile(args.cfg)
cfg.model.pretrained = None
cfg.data.val  = simple_dataset
cfg.data.test = simple_dataset
cfg.dataset_type = 'Custom3DDataset'

dataset = build_dataset(cfg.data.val)
if args.debug:
    # pipeline 디버그 출력 원하면 여기에 추가
    exit(0)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda batch: batch[0]  # batch가 [sample_dict] 이렇게 하나만 오니까 [0]으로 꺼내면 OK
)

# --- 3) 모델 로드 (camera encoder만) ---
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, args.weight, map_location='cpu')
model.cuda().eval()

# --- 4) Q/K attention 계산 헬퍼 ---
def compute_qk_attn(bev_feat: torch.Tensor):
    """
    bev_feat: (B*N, C, H, W)
    returns A: (B*N, H*W, H*W)
    """
    Bn, C, H, W = bev_feat.shape
    # flatten spatial dims
    q = bev_feat.flatten(2).permute(0, 2, 1)    # (Bn, H*W, C)
    k = bev_feat.flatten(2)                    # (Bn, C, H*W)
    # scaled dot-product
    attn = torch.softmax((q @ k) / math.sqrt(C), dim=-1)
    return attn

# --- 5) 루프 & 저장 ---
save_dir = Path(args.out_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# 전체 뷰 수 계산 (배치 수 × 뷰 수)
num_batches = len(loader)
first_batch = next(iter(loader))
N_cam = len(first_batch['img'])
total_cams = num_batches * N_cam
counter = 0

for idx, batch in enumerate(loader):
    # Convert list of numpy images to tensor (B=1, N, C, H, W)
    imgs_np = batch['img']  # list of N numpy arrays shape (H, W, 3)
    imgs_tensor = torch.stack(
        [torch.from_numpy(np.transpose(im, (2, 0, 1))) for im in imgs_np],
        dim=0
    ).unsqueeze(0).cuda(non_blocking=True)
    img = imgs_tensor  # shape (1, N, 3, H, W)
    B, N, C, H, W = img.shape

    # for each view separately
    for cam in range(N):
        counter += 1
        single = img[:, cam]            # (1,3,H,W)
        with torch.no_grad():
            # camera 백본 + 넥 → BEV features list
            flat = single  # already tensor
            feats = model.encoders.camera.backbone(flat)
            bev_feats = model.encoders.camera.neck(feats)
        # bev_feats 는 list(scale1, scale2,...), we pick 첫 번째
        bev1 = bev_feats[0]               # (1, C1, h, w)
        A    = compute_qk_attn(bev1)      # (1, hw, hw)

        fn = save_dir / f"{idx:04d}_cam{cam:02d}_qk_attn.pt"
        torch.save(A.cpu(), fn)

        remaining = total_cams - counter
        logging.info(f"[{counter}/{total_cams}] saved → {fn}   (남은: {remaining})")

logging.info(f"✅ Done. 총 map 개수: {counter}")