import os

# 경로 설정
OUT_DIR = 'results'
PKL_PATH = os.path.join(OUT_DIR, 'results_nusc.bbox.pkl')
JSON_PATH = os.path.join(OUT_DIR, 'results_nusc.bbox.json')

# Step. Convert to json
cmd_convert = f"""
python tools/create_submission.py \
    --pkl_path {PKL_PATH} \
    --submission_path {JSON_PATH} \
    --dataroot data/nuscenes \
    --version v1.0-mini
"""
print(f"[Run] {cmd_convert}")
os.system(cmd_convert)

print(f"\n✅ Done! Check your results here:\n- PKL: {PKL_PATH}\n- JSON: {JSON_PATH}")