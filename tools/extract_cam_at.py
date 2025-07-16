# 표준 import 구문
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models.builder import build_detector #
from mmdet3d.datasets import build_dataloader, build_dataset

# ----------------- Attention Hook 및 전역 변수 -----------------

attention_maps = []

def get_attention_hook(module, input, output):
    """Swin Transformer의 Attention 모듈에서 Attention Score를 추출하는 Forward Hook 함수."""
    attention_maps.append(output.detach().cpu())

# ------------------------- 메인 실행 함수 -------------------------

def main():
    # --- 💡 경로 설정 (이 부분을 수정하세요) ---
    config_path = './onfigs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml'
    checkpoint_path = './pretrained/camera-only-det.pth'  # 사용자 설정 필요
    gpu_id = 0
    # ---------------------------------------------

    # --- 1. 설정 파일 및 장치 준비 ---
    cfg = Config.fromfile(config_path)
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading config from: {config_path}")

    # --- 2. 모델 빌드 및 체크포인트 로드 ---
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. 데이터 로더 준비 ---
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    print("Dataloader prepared.")
    
    # --- 4. 어텐션 추출을 위한 Hook 등록 ---
    handles = []
    camera_backbone = model.encoders.camera.backbone
    
    for stage in camera_backbone.stages:
        for block in stage.blocks:
            handle = block.attn.register_forward_hook(get_attention_hook)
            handles.append(handle)
            
    print(f"Registered {len(handles)} forward hooks in Swin Transformer blocks.")

    # --- 5. 모델 추론 및 Attention 추출 실행 ---
    data_sample = next(iter(data_loader))
    
    for key in data_sample.keys():
        if isinstance(data_sample[key], list) and len(data_sample[key]) > 0 and isinstance(data_sample[key][0], torch.Tensor):
            for i in range(len(data_sample[key])):
                data_sample[key][i] = data_sample[key][i].to(device)

    print("\nRunning inference to trigger hooks...")
    with torch.no_grad():
        model(return_loss=False, **data_sample)

    print("Inference complete. Attention maps have been extracted.")
    
    # --- 6. Hook 제거 ---
    for handle in handles:
        handle.remove()
    print("All hooks have been removed.")

    # --- 7. 추출된 결과 확인 ---
    print(f"\n--- Extraction Results ---")
    print(f"Total number of attention maps extracted: {len(attention_maps)}")
    
    A_RGB_T = attention_maps
    
    for i, attn_map in enumerate(A_RGB_T):
        print(f"Map {i+1:02d} shape: {attn_map.shape}")

if __name__ == '__main__':
    main()