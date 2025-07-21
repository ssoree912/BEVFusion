#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract LiDAR-only BEV Q/K attention maps from BEVFusion.

예시:
    nohup python tools/extract_lidar_bev_qk_attention.py \
        --cfg configs/nuscenes/det/transfusion/secfpn/lidar/lidar_only.yaml \
        --info full_nuscenes/full_nuscenes_infos_val.pkl \
        --weight pretrained/lidar-only-det.pth \
        --out_dir results/qk_attn/bev/lidar > logs/lidar_bev_qk.log 2>&1 &
"""
import os, argparse, math, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

# ───────────── spconv 체크 ────────────────────────────────────────────────
try:
    import spconv  # noqa: F401
except ImportError as e:
    raise RuntimeError(
        "❗ spconv 이 설치되지 않았습니다. 예:\n"
        "   pip install spconv-cu113==2.3.6\n"
        "또는 README 를 참고해 소스 빌드하세요."
    ) from e

# ───────────── 로깅 ────────────────────────────────────────────────────────
log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "extract_lidar_bev_qk_attention.log"),
              logging.StreamHandler()],
)

# ───────────── CLI 인자 ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True, help="config .yaml")
parser.add_argument("--info", required=True, help="nuscenes_infos_val.pkl")
parser.add_argument("--weight", required=True, help="pretrained lidar ckpt")
parser.add_argument("--out_dir", required=True, help="output dir for .pt attn")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ───────────── 데이터셋 (LiDAR-only) ──────────────────────────────────────
cfg = Config.fromfile(args.cfg)

simple_dataset = dict(
    type="NuScenesDataset",
    dataset_root="",
    ann_file=args.info,
    pipeline=[
        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
        dict(type="DefaultFormatBundle3D", classes=[]),
        dict(type="Collect3D", keys=["points"], meta_keys=["lidar2ego"]),
    ],
    modality=dict(use_lidar=True, use_camera=False,
                  use_radar=False, use_map=False, use_external=False),
    test_mode=True,
)
cfg.data.val = simple_dataset
cfg.data.test = simple_dataset
cfg.model.pretrained = None

# ── 탐지를 쓰지 않으므로 head/assigner 제거 ───────────────────────────────
for key in ["head", "fusion_head", "bbox_head", "pts_bbox_head",
            "pts_bbox", "roi_head", "heads"]:
    cfg.model.pop(key, None)
cfg.model.train_cfg = None
cfg.model.test_cfg = None
cfg.model.heads = {}

dataset = build_dataset(cfg.data.val)
loader  = build_dataloader(dataset, samples_per_gpu=1,
                           workers_per_gpu=4, dist=False, shuffle=False)

# ───────────── 모델 로드 ──────────────────────────────────────────────────
model = build_model(cfg.model, test_cfg=None)
load_checkpoint(model, args.weight, map_location="cpu")
model.cuda().eval()

# ───────────── Q/K 계산 헬퍼 ──────────────────────────────────────────────
@torch.no_grad()
def compute_qk_attn(bev):                 # bev: (B,C,H,W)
    B, C, H, W = bev.shape
    q = bev.flatten(2).permute(0, 2, 1)   # (B, H*W, C)
    k = bev.flatten(2)                    # (B, C, H*W)
    return torch.softmax((q @ k) / math.sqrt(C), dim=-1)  # (B, H*W, H*W)

# ───────────── 출력 디렉터리 ──────────────────────────────────────────────
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

total = len(loader)
for idx, batch in enumerate(loader):
    with torch.no_grad():
        # ── 1) points 텐서 → CUDA ───────────────────────────────────────
        pts_dc = batch["points"]
        pts = pts_dc.data[0] if isinstance(pts_dc.data, list) else pts_dc.data
        if isinstance(pts, list): pts = pts[0]
        pts = pts.cuda(non_blocking=True).contiguous()             # (N,5)

        # ── 2) voxelize --------------------------------------------------
        voxels, coors, batch_sz = model.encoders.lidar.voxelize(pts)
        if coors.shape[1] == 3:  # prepend batch idx (0)
            coors = torch.cat([torch.zeros((coors.size(0), 1),
                                            dtype=coors.dtype,
                                            device=coors.device), coors], 1)

        # ── 3) voxel encoder (있으면) ------------------------------------
        if hasattr(model.encoders.lidar, "voxel_encoder") and model.encoders.lidar.voxel_encoder:
            vox_feat, coors, batch_sz = model.encoders.lidar.voxel_encoder(voxels, coors, batch_sz)
        else:
            vox_feat = voxels.mean(1)

        # ---- normalize batch_sz to plain int ---------------------------------
        if not isinstance(batch_sz, int):
            if torch.is_tensor(batch_sz):
                if batch_sz.numel() == 1:
                    batch_sz = int(batch_sz.item())
                else:
                    # fallback: infer from coordinates (max batch idx + 1)
                    batch_sz = int(coors[:, 0].max().item() + 1)
            else:
                batch_sz = int(batch_sz)

        # ── 4) BEV feature ----------------------------------------------
        if hasattr(model.encoders.lidar, "middle_encoder") and model.encoders.lidar.middle_encoder:
            bev = model.encoders.lidar.middle_encoder(vox_feat, coors.int(), batch_sz)  # (B,C,H,W)
        else:
            bev = model.encoders.lidar.backbone(vox_feat, coors.int(), batch_sz)
            if hasattr(bev, "dense"): bev = bev.dense()          # (B,C,Z,H,W) or (B,C,H,W)
            if bev.dim() == 5: bev = bev.squeeze(2)              # Z 축 제거

        # --- ensure bev is 4‑D (B,C,H,W) ---------------------------------
        # Case 1: (B,C,Z,H,W) → squeeze Z
        if bev.dim() == 5:
            bev = bev.squeeze(2)

        # Case 2: (C,H,W) → add batch dim
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)

        # Case 3: (B,C,W) → insert H=1 between C and W
        if bev.dim() == 3:  # after previous step could still be 3‑D
            bev = bev.unsqueeze(2)

        # Case 4: (C,N) or (N,C) → reshape to (1,C,N,1)
        if bev.dim() == 2:
            if bev.shape[0] <= 64:        # assume channels first
                bev = bev.unsqueeze(0).unsqueeze(-1)
            else:                         # assume batch first
                bev = bev.unsqueeze(-1).unsqueeze(-1)

        # Final guard: prepend batch dims until 4‑D
        while bev.dim() < 4:
            bev = bev.unsqueeze(0)

        # if H==1 and W>1, still acceptable; transformer_fpn handles that

        # --- final dimension sanity check --------------------------------
        if bev.dim() == 3:          # (C,H,W) or (B,C,W)
            bev = bev.unsqueeze(0)  # add batch
        if bev.dim() == 2:          # (C,N)
            bev = bev.unsqueeze(0).unsqueeze(-1)  # (1,C,N,1)

        assert bev.dim() == 4, f"Unexpected bev dim {bev.dim()}: {bev.shape}"
        # print(f"[idx {idx}] FINAL bev before neck  shape={bev.shape}, dim={bev.dim()}")
        # print(f"    sizes : B={bev.shape[0] if bev.dim()>0 else '∅'}, "
        #       f"C={bev.shape[1] if bev.dim()>1 else '∅'}, "
        #       f"H={bev.shape[2] if bev.dim()>2 else '∅'}, "
        #       f"W={bev.shape[3] if bev.dim()>3 else '∅'}")
        # --- 5) neck (TransformerFPN) -----------------------------------
        # TransformerFPN config expects two levels whose spatial strides
        # are [2, 4] w.r.t. the original SECOND BEV.  Down‑sample accordingly
        # to cut token count and avoid OOM in the self‑attention block.

        # (a) Level‑0 feature : 128ch, stride‑2  (≈ H/2 × W/2)
        feat_l0 = F.avg_pool2d(bev[:, :128], kernel_size=2, stride=2)   # (B,128,⌊H/2⌋,⌊W/2⌋)

        # (b) Level‑1 feature : 256ch, stride‑4  (≈ H/4 × W/4)
        feat_l1 = F.avg_pool2d(bev,           kernel_size=4, stride=4)   # (B,256,⌊H/4⌋,⌊W/4⌋)

        feats_in = [feat_l0, feat_l1]          # list ⇢ TransformerFPN
        feats_out = model.decoder.neck(feats_in)[0]   # (B,80,h,w)
        bev = feats_out                               # rename for downstream

        # ── 6) Q/K attention -------------------------------------------
        attn = compute_qk_attn(bev)             # (B, H*W, H*W)

    torch.save(attn.cpu().half(), out_dir / f"{idx:04d}_lidar_qk_attn.pt")

    remaining = total - idx - 1
    logging.info(f"[{idx+1}/{total}] saved attention.  Remaining: {remaining}")