# tools/extract_camera_attention.py  ★ robust camera-only ★
# ===============================================================
import os, copy, torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models.backbones.swin import SwinBlock
from mmdet3d.datasets import build_dataset
from mmcv.parallel import DataContainer
from typing import Dict, Any

# ------------------------------------------------ 0. SwinBlock patch
orig_forward = SwinBlock.forward
def fwd_attn(self, x):
    feat = orig_forward(self, x)
    B,L,C = x.shape
    qkv = self.attn.qkv(x).reshape(B,L,3,self.attn.num_heads,C//self.attn.num_heads).permute(2,0,3,1,4)
    q,k = qkv[0], qkv[1]
    attn = (q @ k.transpose(-2,-1)) * self.attn.scale
    if self.attn.attn_mask is not None:
        attn = attn + self.attn.attn_mask
    return feat, attn.softmax(-1)
SwinBlock.forward = fwd_attn

# ------------------------------------------------ 1. base config
cfg = Config.fromfile(
    'configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml')
cfg.model.pretrained = None

# ------------------------------------------------ 2. 파이프라인 재귀 탐색
def find_first_pipeline(node: Any):
    """ConfigDict or list tree에서 'type':'LoadMultiViewImageFromFiles' 가
       포함된 리스트를 만나면 그 리스트를 반환."""
    if isinstance(node, list):
        if any(isinstance(x, dict) and
               x.get('type') == 'LoadMultiViewImageFromFiles' for x in node):
            return node
    if isinstance(node, dict):
        for v in node.values():
            res = find_first_pipeline(v)
            if res is not None:
                return res
    return None

raw_pipe = find_first_pipeline(cfg)
# assert raw_pipe is not None, '카메라 파이프라인을 찾지 못했습니다.'


camera_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ImageNormalize',
         mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    dict(type='Collect3D',
         keys=['img'],
         meta_keys=['camera_intrinsics','camera2ego',
                    'camera2lidar','img_aug_matrix'])
]

# ------------------------------------------------ 3. 새 NuScenesDataset(camera only) 정의
simple_dataset = dict(
    type='NuScenesDataset',
    dataset_root='',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',  # ← 필요시 경로 수정
    pipeline=camera_pipeline,
    modality=dict(use_camera=True, use_lidar=False,
                  use_radar=False, use_map=False, use_external=False),
    test_mode=True,
)

cfg.data.val  = copy.deepcopy(simple_dataset)
cfg.data.test = copy.deepcopy(simple_dataset)
cfg.dataset_type = 'NuScenesDataset'
ds = build_dataset(cfg.data.val)
print(ds[0].keys())   
# ------------------------------------------------ 4. img_filename 교정
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
_old_get = NuScenesDataset.get_data_info
def _new_get(self, idx):
    info = _old_get(self, idx)
    if 'image_paths' in info:
        # 이미 올바른 이름이면 그대로 두기
        pass
    elif 'file_name' in info:
        info['image_paths']  = info['file_name']      # ★ 추가
        info['img_filename'] = info.pop('file_name')
    elif 'img_filename' in info:
        info['image_paths'] = info['img_filename']    # ★ 추가
    return info
NuScenesDataset.get_data_info = _new_get

# ------------------------------------------------ 5. model & hook
from mmdet3d.models import build_model
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, 'pretrained/camera-only-det.pth', map_location='cpu')
model.cuda().eval()

attn_buf: Dict[str, torch.Tensor] = {}
for n,m in model.named_modules():
    if isinstance(m, SwinBlock):
        m.register_forward_hook(lambda mod, _inp, out, k=n: attn_buf.setdefault(k, out[1].cpu()))

# ------------------------------------------------ 6. data loader
from mmdet3d.datasets import build_dataset, build_dataloader
dataset = build_dataset(cfg.data.val)
loader  = build_dataloader(dataset, 1, 0, dist=False, shuffle=False)
print(dataset[0].keys())
# ------------------------------------------------ 7. run & save
import numpy as np
def meta_tensor(metas_dc, key):
    # ▲ metas_dc: DataContainer → 내부 list 꺼내기
    if isinstance(metas_dc, DataContainer):
        metas = metas_dc.data[0]          # batch=1 이므로 0번째
    else:
        metas = metas_dc                 # 이미 list

    mats = []
    for m in metas:
        val = m.get(key, None)
        if val is None:
            mats.append(torch.eye(4))    # 4×4 I
        else:
            mats.append(torch.tensor(val))
    return torch.stack(mats).float().cuda()

# --------------------------- 7. run & save  (교체)
 # 7. run & save – 메타 추출 부분만 교체
os.makedirs('./outputs/attention/camera', exist_ok=True)

for idx, data in enumerate(loader):
    # 텐서 GPU 이동
    data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}

    # ✅ metas 추출 : 'metas' 없으면 'img_metas' 사용
    metas_dc = data.pop('metas', None)
    if metas_dc is None:
        metas_dc = data.pop('img_metas')           # 반드시 존재
    metas = metas_dc.data[0]                       # batch=1 → list

    def meta_tensor(key):
        import numpy as np
        mats = [torch.tensor(m.get(key, np.eye(4))).float() for m in metas]
        return torch.stack(mats).cuda()


# --------------------------- 7. run & save  (수정안)
os.makedirs('./outputs/attention/camera', exist_ok=True)
for idx, data in enumerate(loader):
    # dataloader에서 나온 데이터를 GPU로 이동
    data = {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in data.items()}

    # ✅ 'img_metas'가 DataContainer 안에 있으므로 풀어주기만 하면 됨
    # 이전에 작성하셨던 방식이 올바른 방식입니다.
    if 'metas' in data:
         data['img_metas'] = data.pop('metas').data[0]
    else:
         data['img_metas'] = data['img_metas'].data[0]
    
    # Camera-only 모델을 위한 placeholder
    data['points']    = [None]

    # ✅ 모델 호출은 간단하게 **data 로 전달
    with torch.no_grad():
        # 프레임워크가 data 딕셔너리 안에서 필요한 모든 것을 처리합니다.
        _ = model(return_loss=False, rescale=True, **data)

    for layer, mat in attn_buf.items():
        torch.save(mat.half(), f'./outputs/attention/camera/{idx:04d}_{layer}.pt')
    attn_buf.clear()
    print(f'[{idx+1}/{len(loader)}] attention saved.')