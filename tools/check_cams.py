import pickle

# 🔄 NuScenes info pkl 파일 경로
info_path = './data/nuscenes/nuscenes_infos_val.pkl'  # ← 경로를 실제 사용 중인 것으로 변경

# ✅ 파일 로드
with open(info_path, 'rb') as f:
    infos = pickle.load(f)

# ✅ 첫 번째 샘플 확인
first_info = infos['infos'][0]  # 또는 infos[0]인 경우도 있음
cams = first_info['cams']

# ✅ cams 구조 출력
for cam_name, cam_info in cams.items():
    print(f"[{cam_name}]")
    for key, value in cam_info.items():
        print(f"  {key}: {type(value)}")