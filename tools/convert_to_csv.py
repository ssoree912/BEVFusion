import os
import pickle
import pandas as pd

pkl_path = 'results/results_nusc.bbox.pkl'
csv_path = 'results/results_nusc.bbox.csv'

print(f"[INFO] Loading pkl: {pkl_path}")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

records = []

for frame_id, frame in enumerate(data):
    boxes_3d = frame['boxes_3d'].tensor.cpu().numpy()  # (N, 9)
    scores_3d = frame['scores_3d'].cpu().numpy()       # (N,)
    labels_3d = frame['labels_3d'].cpu().numpy()       # (N,)

    num_boxes = boxes_3d.shape[0]

    for i in range(num_boxes):
        box = boxes_3d[i]
        record = {
            'frame_id': frame_id,
            'box_id': i,
            'score': scores_3d[i],
            'label': labels_3d[i],
            'x': box[0],
            'y': box[1],
            'z': box[2],
            'dx': box[3],
            'dy': box[4],
            'dz': box[5],
            'yaw': box[6],
            'vx': box[7],
            'vy': box[8],
        }
        records.append(record)

df = pd.DataFrame(records)
df.to_csv(csv_path, index=False)

print(f"\n✅ Done! Saved CSV to {csv_path}")