#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#camera bev qk attention 추출 스크립트
"""
Extract camera-only BEV Q/K attention maps from BEVFusion
Usage:
    nohup python tools/extract_camera_bev_qk_attention.py \
        --cfg configs/nuscenes/det/qk/camera.yaml \
        --info full_nuscenes/full_nuscenes_infos_val_with_proj.pkl \
        --weight pretrained/camera-only-det.pth \
        --out_dir results/qk_attn/bev/camera > logs/cam_bev_qk.out 2>&1 &
"""

import os, argparse, math
from pathlib import Path

import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader
import logging

# ─── 로그 세팅 ─────────────────────────────────────
log_dir = Path('logs'); log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir/'extract_camera_bev_qk_attention.log'),
        logging.StreamHandler()
    ]
)

# ─── 파라미터 파싱 ────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',      type=str, required=True)
parser.add_argument('--info',     type=str, required=True)
parser.add_argument('--weight',   type=str, required=True)
parser.add_argument('--out_dir',  type=str, required=True)
parser.add_argument('--debug',    action='store_true')
args = parser.parse_args()

# ─── pipeline 등록 (생략: 원본 코드와 동일) ────────────
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
# ─── 데이터셋 준비 ───────────────────────────────────
cfg = Config.fromfile(args.cfg)
cfg.model.pretrained = None

simple_dataset = dict(
    type         = 'Custom3DDataset',
    dataset_root = '',
    ann_file     = args.info,
    pipeline     = camera_pipeline,  # 위에서 정의한 pipeline
    modality     = dict(use_camera=True, use_lidar=False,
                        use_radar=False, use_map=False, use_external=False),
    test_mode    = True,
)
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
    collate_fn=lambda batch: batch[0]
)

# ─── 모델 로드 ───────────────────────────────────────
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, args.weight, map_location='cpu')
model.cuda().eval()

# ─── Q/K 계산 헬퍼 ───────────────────────────────────
def compute_qk_attn(bev_feat: torch.Tensor):
    """
    bev_feat: (B, C, H, W)
    returns  : (B, H*W, H*W)
    """
    B, C, H, W = bev_feat.shape
    q = bev_feat.flatten(2).permute(0, 2, 1)   # (B, H*W, C)
    k = bev_feat.flatten(2)                    # (B, C, H*W)
    return torch.softmax((q @ k) / math.sqrt(C), dim=-1)

# ─── 출력 디렉터리 ─────────────────────────────────
save_dir = Path(args.out_dir); save_dir.mkdir(parents=True, exist_ok=True)

# ─── Resume support ────────────────────────────────
# Collect indices that were already processed (files existing in out_dir)
existing_indices = {
    int(p.stem.split('_')[0])
    for p in save_dir.glob("*_bev_qk_attn.pt")
    if p.stem.split('_')[0].isdigit()
}
if existing_indices:
    logging.info(f"Resume mode: {len(existing_indices)} attention maps already exist "
                 f"in {save_dir}. They will be skipped.")

total = len(loader)
todo_total = total - len(existing_indices)
processed = 0    # number of NEW attention maps we create this run
overall_done = len(existing_indices)   # already extracted before this run
logging.info(f"총 frame 수: {total}")

# ─── 추출 루프 ───────────────────────────────────────
for idx, batch in enumerate(loader):
    # Skip frames whose attention map already exists (resume capability)
    out_path = save_dir / f"{idx:04d}_bev_qk_attn.pt"
    if idx in existing_indices:
        # logging.debug(f"[{idx:04d}] already extracted – skipping.")
        continue

    # 1) Multi-view 이미지를 (1, N, C, H, W) tensor로 변환
    imgs_np = batch['img']
    imgs_tensor = torch.stack([
        torch.from_numpy(np.transpose(im, (2,0,1))) 
        for im in imgs_np
    ], dim=0).unsqueeze(0).cuda(non_blocking=True)  # shape = (1, N, 3, H, W)

    B, N, C, H, W = imgs_tensor.shape

    # 2) BEV feature 획득 (멀티뷰 → BEV)
    with torch.no_grad():
        # a) flatten all views into batch 차원
        flat = imgs_tensor.view(B*N, C, H, W)
        # b) backbone & neck
        feats = model.encoders.camera.backbone(flat)
        # feats: list of feature maps [(B*N, C1, h1, w1), ...]
        bev_feats = model.encoders.camera.neck(feats)
        # bev_feats[0]: (B*N, C_bev, H_bev, W_bev)
        bev = bev_feats[0].view(B, N, bev_feats[0].shape[1],
                                bev_feats[0].shape[2],
                                bev_feats[0].shape[3])
        # c) 뷰 축(1) 평균 → (B, C_bev, H_bev, W_bev)
        bev = bev.mean(dim=1)

        # 3) Q/K attention 계산
        attn = compute_qk_attn(bev)  # (B, HW, HW)

    # 4) 저장
    fn = save_dir / f"{idx:04d}_bev_qk_attn.pt"
    torch.save(attn.cpu(), fn)

    processed += 1
    remaining = todo_total - processed
    overall_done += 1
    msg = f"[{overall_done}/{total}] saved → {fn}   (남은: {remaining})"
    logging.info(msg)
    print(msg, flush=True)

logging.info(f"✅ 완료! 이번 실행에서 {processed}개, 전체 {len(existing_indices)+processed}/{total}개 추출됨.")