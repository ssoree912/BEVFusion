#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract camera-only Swin attention maps from BEVFusion.
Usage:
    python tools/extract_camera_attention.py              # 정상 추출
    python tools/extract_camera_attention.py --debug      # 1개 샘플 디버그
"""
import argparse, os, copy,inspect
from pathlib import Path
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

# ❶ 새 변환 클래스 정의 — tools/extract_camera_attention.py 상단에 추가
from mmdet.datasets import PIPELINES
from pyquaternion import Quaternion
import mmcv, numpy as np, torch
from pyquaternion import Quaternion
# from mmcv.utils import Registry
from mmcv.utils import Registry

# Logging setup
import logging
# Setup logging to file and console
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir/'extract_camera_attention.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# from .transforms_3d_extra import StackMultiViewImage 


# ────────────────────────────────
# 1. util : dict → 4×4 float32
# ────────────────────────────────
@PIPELINES.register_module()
class PackCameraMeta:
    """Collect3D 이후 호출 – meta 를 img_metas 로 묶어 줌"""
    def __call__(self, results):
        from mmcv.parallel import DataContainer

        meta = {}
        # Collect3D 가 cpu_only DataContainer 로 넣어둔 값 꺼내기
        src_dc = results.pop('metas')          # DataContainer([...])
        src = src_dc.data[0] if isinstance(src_dc.data, list) else src_dc.data
        meta.update(src)                       # 기존 항목 유지

        # 우리가 추가로 만든 키들도 meta 로 이동
        for k in [
            'camera_intrinsics','camera2ego','camera2lidar',
            'lidar2camera','lidar2image','img_aug_matrix',  'lidar2ego',    'timestamp'
        ]:
            if k in results:
                meta[k] = results.pop(k).data   # DC → 텐서/값

        # 최종 img_metas 로 저장 (batch collate 에서 그대로 묶임)
        results['img_metas'] = DataContainer(meta, cpu_only=True)
        return results
# ❶ PIL  →  np.ndarray(float32, H×W×3  BGR)
@PIPELINES.register_module()
class ImageToNumpy:
    def __call__(self, results):
        results['img'] = [
            np.asarray(img, dtype=np.float32)      # (H,W,3) BGR
            for img in results['img']
        ]
        return results

@PIPELINES.register_module()
class StackMVNormalize:
    """List[np.ndarray(H,W,3)] → torch.float32 (N,3,H,W), 이미지를
       0-1 스케일로 바꾸고 mean/std 로 정규화한다."""
    def __init__(self,
                 mean=(0.485,0.456,0.406),
                 std =(0.229,0.224,0.225)):
        self.mean = torch.tensor(mean).view(1,3,1,1)
        self.std  = torch.tensor(std ).view(1,3,1,1)

    def __call__(self, results):
        imgs = []
        for img in results['img']:               # np.ndarray H,W,3 (BGR)
            # BGR → RGB  (& 0-1 float32)
            img = img[..., ::-1].astype(np.float32) / 255.0
            img = torch.from_numpy(img.transpose(2,0,1))     # C,H,W
            imgs.append(img)
        imgs = torch.stack(imgs, 0)               # N,C,H,W
        imgs = (imgs - self.mean) / self.std
        results['img'] = imgs
        return results
# ---------------------------------------------------------------------------

@PIPELINES.register_module()
class FixCamLidarMeta:
    """cams 딕셔너리의 모든 RT·K·증강 행렬을 (4,4) float32 로 표준화"""
    @staticmethod
    def _to_44(mat_like):
        A = np.asarray(mat_like, dtype=np.float32)

        if A.shape == (4,):             # quaternion
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = Quaternion(A).rotation_matrix
            return M

        if A.shape == (3, 3):
            M = np.eye(4, dtype=np.float32);  M[:3, :3] = A;  return M
        if A.shape == (3, 4):
            M = np.pad(A, ((0, 1), (0, 0)));  M[3, 3] = 1;     return M
        if A.shape == (4, 4):
            return A.astype(np.float32)

        raise ValueError(f'Unsupported shape {A.shape}')

    def __call__(self, results):
        cams  = results['cams']                # OrderedDict
        order = sorted(cams.keys())            # 고정 순서

        def cam2ego_matrix(cam_info):
            """sensor2ego_rotation(quat) + sensor2ego_translation → 4×4"""
            q = cam_info['sensor2ego_rotation']      # [w, x, y, z]
            t = cam_info['sensor2ego_translation']   # [3]
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = Quaternion(q).rotation_matrix
            M[:3,  3] = t
            return M

        pull = lambda k, default=None: [
            self._to_44(cams[c].get(k, default if default is not None else np.eye(4)))
            for c in order]

        results.update(
            image_paths      = [cams[c]['data_path'] for c in order],

            camera_intrinsics= [self._to_44(cams[c]['cam_intrinsic']) for c in order],

            camera2ego       = [cam2ego_matrix(cams[c])               for c in order],

            lidar2image      = pull('lidar2image_matrix'),   # (3,4) → 4×4
            lidar2camera     = pull('lidar2camera'),         # 이미 4×4
            camera2lidar     = pull('camera2lidar'),         # 이미 4×4
            img_aug_matrix   = pull('img_aug_matrix', np.eye(4, dtype=np.float32)),
        )

        # 더 이상 'cams' 가 필요 없으면 제거
        results.pop('cams', None)
        return results
# --------------------------------------------------------------------------- #
# 1. 파라미터
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--cfg",
    default="configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml")
parser.add_argument("--info",
    default="full_nuscenes/full_nuscenes_infos_val_with_proj.pkl")
parser.add_argument("--weight",
    default="pretrained/camera-only-det.pth")
parser.add_argument("--out_dir",
    default="results/attention/camera")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# --------------------------------------------------------------------------- #
# 2. Config & dataset --------------------------------------------------------------------------- #
cfg = Config.fromfile(args.cfg)
cfg.model.pretrained = None



camera_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='FixCamLidarMeta'),  
    dict(type='ImageToNumpy'), # 메타 필드 전개
    dict(type='StackMVNormalize'),         # List → Tensor & 정규화
    dict(type='Collect3D',                 # 텐서는 stack=True, 나머지는 cpu_only
         keys=['img'],
        meta_keys=[
         # 우리가 forward 에서 필요로 하는 것 전부!
         'camera_intrinsics', 'camera2ego', 'camera2lidar',
         'lidar2camera', 'lidar2image', 'img_aug_matrix',
         'lidar2ego', 'timestamp'
     ]),
    dict(type='PackCameraMeta') ,
]

simple_dataset = dict(
    type       = "Custom3DDataset",
    dataset_root = "",
    ann_file   = args.info,
    pipeline   = camera_pipeline,
    modality   = dict(use_camera=True, use_lidar=False,
                      use_radar=False, use_map=False, use_external=False),
    test_mode  = True,
)
cfg.data.val  = copy.deepcopy(simple_dataset)
cfg.data.test = copy.deepcopy(simple_dataset)
cfg.dataset_type = "Custom3DDataset"

dataset = build_dataset(cfg.data.val)

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

# Progress tracking setup
num_batches = len(loader)
# Peek first batch to get number of camera views
first_batch = next(iter(loader))
N_cam = first_batch['img'].shape[1]
total_cams = num_batches * N_cam
counter = 0
# Reset loader iterator
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

# --------------------------------------------------------------------------- #
# 5. Model & ShiftWindowMSA attention hook
# --------------------------------------------------------------------------- #
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, args.weight, map_location="cpu")
model.cuda().eval()

from mmdet.models.backbones.swin import ShiftWindowMSA

# Enable returning attention in ShiftWindowMSA
for m in model.modules():
    if isinstance(m, ShiftWindowMSA):
        m.return_attn = True

# Buffer for attention maps
attn_buf = {}

# Hook to capture attn output from ShiftWindowMSA
def attn_hook(mod, _inp, out):
    # out may be a tuple (x, attn) or a single attn tensor
    if isinstance(out, tuple) and len(out) > 1:
        attn = out[1]
    else:
        attn = out
    attn_buf[mod.name] = attn.detach().cpu()

# Register hook on every ShiftWindowMSA module
for name, m in model.named_modules():
    if isinstance(m, ShiftWindowMSA):
        m.name = name
        m.register_forward_hook(attn_hook)


# Python 스크립트 안에서
depthnet = model.encoders.camera.vtransform.depthnet

# 1) depth_conv_* 이름을 가진 하위 모듈 이름 수집
level_names = [
    name for name, _ in depthnet.named_children()
    if name.startswith('depth_conv_')
]
# 숫자 순서대로 정렬
level_names.sort(key=lambda x: int(x.split('_')[-1]))

logger.info("DepthNet에서 사용 중인 depth_conv 레벨: %s", level_names)
# → 예) ['depth_conv_1', 'depth_conv_2', 'depth_conv_3']

# 2) 각 레벨별 블록 수 확인
for lvl in level_names:
    module = getattr(depthnet, lvl)
    num_blocks = len(module)
    logger.info("  • %s  has  %d  blocks", lvl, num_blocks)
    
# 3) (옵션) 마지막 레벨(depth_conv_3)의 출력 채널 수 = depth bin 수
conv3 = depthnet.depth_conv_3[0]  # 첫 번째 Conv2d 레이어
logger.info("Depth bin 개수: %d", conv3.out_channels)


vtrans = model.encoders.camera['vtransform']
orig_fwd = vtrans.forward  
def forward_no_radar(x,  points=None,  # ★ radar=None 제거
                     radar=None,                         # ★ default 추가
                     camera2ego=None, lidar2ego=None,     # 이하 동일
                     lidar2camera=None, lidar2image=None,
                     camera_intrinsics=None, camera2lidar=None,
                     img_aug_matrix=None, lidar_aug_matrix=None,
                     *extra, **kw):                    # ★  ← **kw 추가
    """pass-through for any extra positional/keyword args (depth_loss, depths…)"""
    return orig_fwd(
        x, points,radar,
        camera2ego=camera2ego, lidar2ego=lidar2ego,
        lidar2camera=lidar2camera, lidar2image=lidar2image,
        camera_intrinsics=camera_intrinsics,
        camera2lidar=camera2lidar,
        img_aug_matrix=img_aug_matrix,
        lidar_aug_matrix=lidar_aug_matrix,
        *extra, **kw                                # ★  그대로 전달
    )

vtrans.forward = forward_no_radar
def blank4x4(batch, n_cam):
    """(B,N,4,4) 단위행렬 텐서 생성"""
    eye = torch.eye(4, dtype=torch.float32)
    return eye.expand(batch, n_cam, 4, 4).clone()
# 카메라-전용 forward 헬퍼
def forward_camera_only(model, img, metas):
    """
    img   : (B, N, 3, H, W) GPU tensor
    metas : dict (img_metas)
    """
    B, N, _, H, W = img.shape
    device = img.device
    cam_mod = model.encoders.camera  # ModuleDict(camera.backbone, neck, vtransform)

    # helper: list of (4×4) ndarray → (B,N,4,4) tensor
    def stack_mats(key):
        t = torch.stack([
            torch.as_tensor(m, dtype=torch.float32, device=device)
            for m in metas[key]
        ], dim=0)                    # (N,4,4)
        return t.unsqueeze(0).expand(B, -1, 4, 4).clone()  # (B,N,4,4)

    mats = {
        k: stack_mats(k)
        for k in [
            'camera_intrinsics', 'camera2ego', 'camera2lidar',
            'lidar2camera', 'lidar2image', 'img_aug_matrix', 'lidar2ego'
        ]
    }
    with torch.no_grad():
        # backbone + neck 만 실행해서 hook이 self-attention map을 모읍니다.
        flat = img.view(B * N, 3, img.shape[3], img.shape[4])
        feats = model.encoders.camera.backbone(flat)
        _     = model.encoders.camera.neck(feats)
    if not attn_buf:
         print(f'⚠️  batch {idx}: attn_buf EMPTY → hook가 안 불렸는지 확인')

  
# 6. Run & 저장
# --------------------------------------------------------------------------- #
# ───── 0) 준비부 ────────────────────────────────────────────────
save_dir = Path(args.out_dir).expanduser().resolve()
save_dir.mkdir(parents=True, exist_ok=True)

# (GPU 한정) 단위행렬 텐서 만드는 헬퍼
def eye44(B, N):
    return torch.eye(4, dtype=torch.float32, device='cuda').expand(B, N, 4, 4).clone()

# ───── DataLoader loop ─────────────────────────────────────

for idx, batch in enumerate(loader):
    img = batch['img'].cuda(non_blocking=True)  # (B, N_cam, 3, H, W)
    B, N, C, H, W = img.shape

    for cam in range(N):
        attn_buf.clear()
        # single‐view 이미지
        single = img[:, cam]        # shape (B,3,H,W)

        # backbone 통과만 시켜서 hook으로 attn 수집
        with torch.no_grad():
            _ = model.encoders.camera.backbone(single)

        counter += 1
        remaining = total_cams - counter
        # 한 줄 덮어쓰기로 진행 상황 출력
        logger.info(f"Progress: {counter}/{total_cams} processed, {remaining} remaining")

        if not attn_buf:
            continue

        # attention 저장
        for layer, attn in attn_buf.items():
            fn = save_dir / f"{idx:04d}_cam{cam}_{layer}.pt"
            torch.save(attn.half(), fn, _use_new_zipfile_serialization=False)

    # 배치가 끝나면 버퍼 초기화
    attn_buf.clear()

# 마지막 줄 깨끗하게
logger.info("✓ all done → %s", save_dir)