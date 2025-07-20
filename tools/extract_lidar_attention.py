#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract camera-only Swin attention maps from BEVFusion.
Usage:
    python tools/extract_camera_attention.py              # 정상 추출
    python tools/extract_camera_attention.py --debug      # 1개 샘플 디버그
"""
import argparse, os, copy,inspect,torch
from pathlib import Path
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
import logging

logging.basicConfig(
    filename='logs/extract_lidar_attention.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ───── prepare output directory ─────
parser = argparse.ArgumentParser()
parser.add_argument("--cfg",
    default="configs/nuscenes/det/transfusion/secfpn/lidar/lidar_only.yaml")
parser.add_argument("--info",
    default="full_nuscenes/full_nuscenes_infos_val_with_proj.pkl")
parser.add_argument("--weight",
    default="pretrained/lidar-only-det.pth")
parser.add_argument("--out_dir",
    default="results/attention/camera")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

save_dir = Path(args.out_dir).expanduser().resolve()
save_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Config & dataset --------------------------------------------------------------------------- #
cfg = Config.fromfile(args.cfg)
cfg.model.pretrained = None
# ensure CenterPointBBoxCoder has required args
if hasattr(cfg.model, 'heads') and 'object' in cfg.model.heads:
    bbox_cfg = cfg.model.heads.object.bbox_coder
    # pull pc_range and voxel_size from lidar encoder settings
    pc_range = cfg.model.encoders.lidar.voxelize.point_cloud_range
    voxel_size = cfg.model.encoders.lidar.voxelize.voxel_size
    bbox_cfg['pc_range'] = pc_range
    bbox_cfg['voxel_size'] = voxel_size

lidar_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='PointsRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='DefaultFormatBundle3D', classes=[
        'car','truck','construction_vehicle','bus','trailer',
        'barrier','motorcycle','bicycle','pedestrian','traffic_cone'
    ]),
    dict(type='Collect3D', keys=['points'], meta_keys=['lidar2ego']),
]

simple_dataset = dict(
    type       = "Custom3DDataset",
    ann_file   = args.info,
    pipeline   = lidar_pipeline,
    modality   = dict(use_camera=False, use_lidar=True,
                      use_radar=False, use_map=False, use_external=False),
    test_mode  = True,
)
cfg.data.val  = copy.deepcopy(simple_dataset)
cfg.data.test = copy.deepcopy(simple_dataset)
cfg.dataset_type = "Custom3DDataset"

dataset = build_dataset(cfg.data.val, default_args={'dataset_root': ''})

# --------------------------------------------------------------------------- #
# 3. (선택) 디버그: 첫 샘플 파이프라인 확인 후 종료
# --------------------------------------------------------------------------- #
def debug_first_sample(ds):
    print("\n[DEBUG] raw get_data_info keys:", list(ds.data_infos[0].keys()))
    input_dict = ds.get_data_info(0)
    print("\n[DEBUG] input_dict field dtypes:")
    for k, v in input_dict.items():
        t = f"list(len={len(v)})" if isinstance(v, list) else type(v).__name__
        print(f"  • {k:<16} : {t}")

    ds.pre_pipeline(input_dict)
    try:
        processed = ds.pipeline(input_dict)
        print("\n[DEBUG] after pipeline:")
        for k, v in processed.items():
            print(f"  • {k:<8} : {type(v)}")
    except Exception as e:
        print("\n[DEBUG] pipeline error →", e)
    print("\n[DEBUG] finish.")
    exit(0)

if args.debug:
    debug_first_sample(dataset)

# --------------------------------------------------------------------------- #
# 4. DataLoader
# --------------------------------------------------------------------------- #
loader = build_dataloader(dataset,
                          samples_per_gpu=1,
                          workers_per_gpu=0,
                          dist=False,
                          shuffle=False)

# --------------------------------------------------------------------------- #
# 5. Model & ShiftWindowMSA attention hook
# --------------------------------------------------------------------------- #
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, args.weight, map_location="cpu")
model.cuda().eval()

from torch.nn import MultiheadAttention

attn_buf = {}
def lidar_attn_hook(mod, _inp, out):
    # out 이 (output, attn_weights) 형태일 때
    if isinstance(out, tuple) and len(out)==2:
        attn_buf[mod.__class__.__name__] = out[1].detach().cpu()

for idx, batch in enumerate(loader):
    # unwrap DataContainer
    raw = batch['points']
    if isinstance(raw, DataContainer):
        raw = raw.data
    # if still wrapped in a list, drill down
    while isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    # now raw should be a Tensor
    pts_tensor = raw.cuda(non_blocking=True)
    pts_list = [pts_tensor]

    attn_buf.clear()

    # build dummy camera matrices
    device = pts_list[0].device
    B = len(pts_list)
    empty_cam = torch.empty((B, 0, 4, 4), device=device)
    lidar_aug = torch.eye(4, device=device).unsqueeze(0).expand(B, 1, 4, 4)

    with torch.no_grad():
        _ = model(
            points=pts_list,
            img=None,
            camera2ego=empty_cam,
            lidar2ego=empty_cam,
            lidar2camera=empty_cam,
            lidar2image=empty_cam,
            camera_intrinsics=empty_cam,
            camera2lidar=empty_cam,
            img_aug_matrix=empty_cam,
            lidar_aug_matrix=lidar_aug,
            metas=[{}], depths=None,
            return_loss=False, rescale=True,
        )

    # 3) now your MultiheadAttention hooks will have fired
    if not attn_buf:
        logger.warning(f"⚠️ batch {idx}: no attention collected")
    else:
        for mod_name, attn in attn_buf.items():
            path = save_dir / f"{idx:04d}_{mod_name}.pt"
            torch.save(attn.half(), path, _use_new_zipfile_serialization=False)

    logger.info(f"completed batch {idx+1}/{len(loader)}")

print("✓ all done →", save_dir)