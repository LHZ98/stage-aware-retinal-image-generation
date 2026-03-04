"""
Compute per-class (diagnosis 0-4) VTI/Width stats from geometry_case_data.json + aptos2019 CSVs.
Output: geometry_stats_by_class.json
"""

import os
import json
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
geometry_path = os.path.join(script_dir, "geometry_case_data.json")
project_root = os.path.dirname(os.path.dirname(script_dir))
default_label_dir = os.path.join(project_root, "datasets", "datasets", "aptos2019")


def main():
    p = argparse.ArgumentParser(description="Compute per-class VTI/Width stats")
    p.add_argument("--geometry", type=str, default=geometry_path, help="path to geometry_case_data.json")
    p.add_argument("--label_dir", type=str, default=default_label_dir, help="dir with train_1.csv, valid.csv, test.csv")
    p.add_argument("--out", type=str, default=None, help="output JSON path (default: geometry_stats_by_class.json in same dir as geometry)")
    args = p.parse_args()

    # 1. case_id -> diagnosis from CSVs
    case_to_diagnosis = {}
    for name, fname in [("train", "train_1.csv"), ("val", "valid.csv"), ("test", "test.csv")]:
        csv_path = os.path.join(args.label_dir, fname)
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, "r") as f:
            lines = f.readlines()
        if not lines:
            continue
        header = lines[0].strip().split(",")
        idx_id = header.index("id_code") if "id_code" in header else 0
        idx_diag = header.index("diagnosis") if "diagnosis" in header else 1
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) <= max(idx_id, idx_diag):
                continue
            case_id = parts[idx_id].strip()
            try:
                diagnosis = int(parts[idx_diag].strip())
            except ValueError:
                continue
            case_to_diagnosis[case_id] = diagnosis

    if not case_to_diagnosis:
        print("No case_id->diagnosis from CSVs. Check --label_dir and CSV columns (id_code, diagnosis).")
        return

    # 2. Load geometry
    if not os.path.isfile(args.geometry):
        print(f"File not found: {args.geometry}")
        return
    with open(args.geometry, "r") as f:
        geometry_data = json.load(f)

    # 3. Group by class
    from collections import defaultdict
    class_data = defaultdict(list)
    for case in geometry_data:
        case_id = case.get("case_id")
        if case_id not in case_to_diagnosis:
            continue
        diagnosis = case_to_diagnosis[case_id]
        class_data[diagnosis].append({
            "VTI": float(case["VTI"]),
            "Width": float(case["Width"]),
            "Density": float(case.get("Density", 0)),
        })

    # 4. Per-class stats: count, mean, std, min, max
    import numpy as np
    out = {}
    for cls in sorted(class_data.keys()):
        arr = class_data[cls]
        vti = np.array([x["VTI"] for x in arr])
        width = np.array([x["Width"] for x in arr])
        density = np.array([x["Density"] for x in arr])
        out[str(cls)] = {
            "count": int(len(arr)),
            "VTI": {
                "mean": float(np.mean(vti)),
                "std": float(np.std(vti)) if len(vti) > 1 else 0.0,
                "min": float(np.min(vti)),
                "max": float(np.max(vti)),
                "median": float(np.median(vti)),
            },
            "Width": {
                "mean": float(np.mean(width)),
                "std": float(np.std(width)) if len(width) > 1 else 0.0,
                "min": float(np.min(width)),
                "max": float(np.max(width)),
                "median": float(np.median(width)),
            },
            "Density": {
                "mean": float(np.mean(density)),
                "std": float(np.std(density)) if len(density) > 1 else 0.0,
                "min": float(np.min(density)),
                "max": float(np.max(density)),
            },
        }

    out_path = args.out or os.path.join(os.path.dirname(args.geometry), "geometry_stats_by_class.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Written: {out_path}")
    print(f"{len(case_to_diagnosis)} cases with diagnosis; {sum(len(v) for v in class_data.values())} matched in geometry.")
    for cls in sorted(out.keys(), key=lambda x: int(x)):
        s = out[cls]
        print(f"  Class {cls}: n={s['count']}, VTI mean={s['VTI']['mean']:.4f} (±{s['VTI']['std']:.4f}), "
              f"Width mean={s['Width']['mean']:.4f} (±{s['Width']['std']:.4f})")


if __name__ == "__main__":
    main()
