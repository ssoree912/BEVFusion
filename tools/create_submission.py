# tools/create_submission.py

import pickle
import argparse
from nuscenes.eval.detection.scripts.export_submission import export

def parse_args():
    parser = argparse.ArgumentParser(description='Convert pkl to nuscenes json')
    parser.add_argument('--pkl_path', type=str, required=True)
    parser.add_argument('--submission_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] Loading pkl: {args.pkl_path}")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"[INFO] Converting to JSON: {args.submission_path}")
    export(data, args.submission_path)

    print(f"✅ Done! Saved to {args.submission_path}")

if __name__ == '__main__':
    main()