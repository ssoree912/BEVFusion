#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract camera-only BEV Q/K attention maps from BEVFusion
Usage:
    nohup python tools/extract_camera_bev_qk_attention.py \
        --cfg configs/nuscenes/det/qk/camera.yaml \
        --info full_nuscenes/full_nuscenes_infos_val_with_proj.pkl \
        --weight pretrained/camera-only-det.pth \
        --out_dir results/qk_attn/bev/camera > logs/camera_bev_qk.log 2>&1 &
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
        logging.FileHandler(log_dir/'extract_camera_bev_qk_attention.log'),
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
            'img_shape', 'pad_shape',
            'camera_intrinsics', 'camera2ego', 'cam2lidar',
            'lidar2cam', 'lidar2image', 'img_aug_matrix',
            'lidar2ego', 'timestamp'
        ]
        metas = {}
        for key in meta_keys:
            if key in results:
                # Pop the DataContainer and extract its raw data
                dc = results.pop(key)
                metas[key] = dc.data
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
             'img_shape', 'pad_shape',
             'camera_intrinsics', 'camera2ego', 'cam2lidar',
             'lidar2cam', 'lidar2image', 'img_aug_matrix',
             'lidar2ego', 'timestamp'
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

# Monkey-patch bev_pool to log geometry and feature shapes
vtrans = model.encoders.camera.vtransform
orig_bev_pool = vtrans.bev_pool
def bev_pool_debug(self, geom, x):
    # Debug: match x and geom grid dims
    B, N, D, fH, fW, C = x.shape
    # geom shape is [B, N, D, Y, X, 3]
    _, _, Dg, Y, X, _ = geom.shape
    if D != Dg:
        print(f"[WARN bev_pool] depth dim mismatch: x D={D}, geom D={Dg}")
    if fH != Y or fW != X:
        # Reshape and interpolate features to [B*N*D, C, fH, fW]
        x_flat = x.permute(0,1,2,5,3,4).reshape(B*N*D, C, fH, fW)
        x_flat = F.interpolate(x_flat, size=(Y, X), mode='bilinear', align_corners=False)
        # Restore to [B, N, D, Y, X, C]
        x = x_flat.view(B, N, D, C, Y, X).permute(0,1,2,4,5,3)
        # print(f"[DEBUG bev_pool] Resized x to {tuple(x.shape)} to match geom grid {(Y, X)}")
    # print(f"[DEBUG bev_pool] geom.shape={tuple(geom.shape)}, x.shape={tuple(x.shape)}")
    return orig_bev_pool(geom, x)
# Bind the debug version
vtrans.bev_pool = bev_pool_debug.__get__(vtrans, type(vtrans))

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
    logging.info(f"=== Start batch {idx+1}/{num_batches}, total views per batch: {N_cam} ===")
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
        logging.info(f"--- Processing view {cam+1}/{N_cam} of batch {idx+1} (global view {counter+1}/{total_cams}) ---")
        counter += 1
        single = img[:, cam]            # (1,3,H,W)
        with torch.no_grad():
            flat = single  # alias current view tensor for backbone input
            # 1) 이미지 평면 feature 생성 (backbone + FPN)
            feats = model.encoders.camera.backbone(flat)
            # Extract features and ensure a single tensor is passed to vtransform
            neck_out = model.encoders.camera.neck(feats)
            if isinstance(neck_out, (tuple, list)):
                # print(f"[DEBUG] neck_out length = {len(neck_out)}")
                # for idx_n, feat_n in enumerate(neck_out):
                    # print(f"[DEBUG] neck_out[{idx_n}].shape = {feat_n.shape}")
                # TODO: choose the correct FPN output index that matches BEV grid (e.g., 2 for P4)
                img_feats = neck_out[-1]  # temporary fallback to last element
            else:
                img_feats = neck_out

            # Ensure img_feats has a camera view dimension for vtransform
            if img_feats.dim() == 4:
                # [B, C, H, W] → [B, 1, C, H, W]
                img_feats = img_feats.unsqueeze(1)

            # 2) BEV 변환을 통해 BEV feature 추출
            #    PackCameraMeta로 모아둔 metadata를 꺼내서 텐서로 변환
            meta = batch['camera_meta'].data
            mats = {
                k: torch.as_tensor(meta[k], dtype=torch.float32, device='cuda').unsqueeze(0)
                for k in meta
            }
            # Remove timestamp from metadata to avoid geometry dimension issues
            mats.pop('timestamp', None)
            # Provide default identity for lidar_aug_matrix if not present
            if 'lidar_aug_matrix' not in mats or mats['lidar_aug_matrix'] is None:
                # Create (B=1, N, 4, 4) identity matrices for lidar augmentation
                mats['lidar_aug_matrix'] = torch.eye(4, dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0).expand(1, N, 4, 4).clone()

            # Ensure lidar2ego has a per-view dimension
            if 'lidar2ego' in mats:
                if mats['lidar2ego'].dim() == 3:
                    # Expand single lidar2ego matrix to per-camera views
                    mats['lidar2ego'] = mats['lidar2ego'].unsqueeze(1).expand(1, N, 4, 4)

            # Restrict per-view meta tensors to the current camera view
            for k in ['camera2ego', 'cam2lidar', 'lidar2cam', 'lidar2image',
                      'camera_intrinsics', 'img_aug_matrix', 'lidar_aug_matrix',
                      'lidar2ego']:
                mats[k] = mats[k][:, cam:cam+1]

            # Debug: print shape info before vtransform
            # print(f"[DEBUG] img_feats.shape = {img_feats.shape}")
            # for key, tensor in mats.items():
            #     print(f"[DEBUG] {key}.shape = {tuple(tensor.shape)}")

            bev = model.encoders.camera.vtransform(
                img_feats,
                points=None,
                radar=None,
                camera2ego=mats['camera2ego'],
                lidar2ego=mats.get('lidar2ego'),
                lidar2camera=mats.get('lidar2cam'),
                lidar2image=mats.get('lidar2image'),
                camera_intrinsics=mats['camera_intrinsics'],
                camera2lidar=mats.get('cam2lidar'),
                img_aug_matrix=mats.get('img_aug_matrix'),
                lidar_aug_matrix=mats['lidar_aug_matrix'],
                img_shape=mats['img_shape'][0],
                pad_shape=mats['pad_shape'][0]
            )

            # 3) BEV Q/K attention 계산
            A = compute_qk_attn(bev)

        fn = save_dir / f"{idx:04d}_cam{cam:02d}_qk_attn.pt"
        torch.save(A.cpu(), fn)

        remaining = total_cams - counter
        logging.info(f"SAVED [{counter}/{total_cams}] → {fn} (remaining {remaining})")

logging.info(f"✅ Done. 총 map 개수: {counter}")