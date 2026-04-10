#!/usr/bin/env python3
"""
Analyze object candidates in a decision_topk.json.

Usage:
  python test_scripts/analyze-object.py \
    --input output_process/.../decision_topk.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def box_iou_xyzwhd(a: np.ndarray, b: np.ndarray) -> float:
    """
    Axis-aligned IoU for boxes in [cx, cy, cz, w, h, d].
    """
    # convert to xyzxyz
    a_min = a[:3] - a[3:] / 2.0
    a_max = a[:3] + a[3:] / 2.0
    b_min = b[:3] - b[3:] / 2.0
    b_max = b[:3] + b[3:] / 2.0

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_size[0] * inter_size[1] * inter_size[2])

    a_vol = float(np.prod(np.maximum(a[3:], 0.0)))
    b_vol = float(np.prod(np.maximum(b[3:], 0.0)))
    union = a_vol + b_vol - inter_vol
    if union <= 1e-9:
        return 0.0
    return inter_vol / union


def build_clusters(points: np.ndarray, threshold: float) -> List[List[int]]:
    """
    Simple connected-components clustering by pairwise distance threshold.
    """
    n = len(points)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze decision_topk object candidates")
    parser.add_argument(
        "--input",
        type=str,
        default="output_process/20260410-162646/scene=00800-TEEsavR23oF/episode=0/task=0/decisions/dec_010/decision_topk.json",
        help="Path to decision_topk.json",
    )
    parser.add_argument(
        "--cluster_distance",
        type=float,
        default=1.0,
        help="Distance threshold (meters) for grouping potential same-object candidates",
    )
    parser.add_argument(
        "--near_pair_distance",
        type=float,
        default=0.8,
        help="Distance threshold (meters) to report near pairs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output report json path (default: <input_dir>/analysis_report.json)",
    )
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    data = json.loads(in_path.read_text(encoding="utf-8"))
    objs = data.get("object_topk_by_score", [])
    if not objs:
        print("No object candidates found.")
        return

    scores = np.array([float(o.get("score", 0.0)) for o in objs], dtype=float)
    centers = np.array([o.get("center_xyz", [0.0, 0.0, 0.0]) for o in objs], dtype=float)
    boxes = np.array([o.get("box_xyzwhd", [0.0] * 6) for o in objs], dtype=float)
    ids = [int(o.get("object_id_in_memory", -1)) for o in objs]
    counts = [float(o.get("count", 0.0)) for o in objs]

    n = len(objs)
    dist_mat = np.zeros((n, n), dtype=float)
    iou_mat = np.zeros((n, n), dtype=float)
    near_pairs: List[Dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            dist_mat[i, j] = dist_mat[j, i] = d
            iou = box_iou_xyzwhd(boxes[i], boxes[j])
            iou_mat[i, j] = iou_mat[j, i] = iou
            if d <= args.near_pair_distance:
                near_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "obj_i": ids[i],
                        "obj_j": ids[j],
                        "distance": d,
                        "iou": iou,
                        "score_i": float(scores[i]),
                        "score_j": float(scores[j]),
                    }
                )

    clusters = build_clusters(centers, threshold=args.cluster_distance)
    cluster_rows = []
    for cidx, c in enumerate(clusters):
        c_scores = scores[c]
        c_center = centers[c].mean(axis=0)
        cluster_rows.append(
            {
                "cluster_id": cidx,
                "size": len(c),
                "members_topk_idx": c,
                "members_object_id": [ids[k] for k in c],
                "mean_score": float(c_scores.mean()),
                "max_score": float(c_scores.max()),
                "mean_center_xyz": [float(x) for x in c_center.tolist()],
            }
        )

    report = {
        "input_path": str(in_path),
        "decision_num": data.get("decision_num"),
        "num_objects_topk": n,
        "score_stats": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        },
        "count_stats": {
            "min": float(np.min(counts)),
            "max": float(np.max(counts)),
            "mean": float(np.mean(counts)),
        },
        "interpretation": {
            "high_scores_note": (
                "Top-k scores都接近1并不代表同一物体；通常是候选排序头部都很高，"
                "需结合center距离和box IoU判断是否重复/同物体。"
            ),
            "same_object_heuristic": (
                f"若两候选中心距 <= {args.near_pair_distance:.2f}m 且 IoU较高，"
                "更可能是同一物体的重复候选。"
            ),
        },
        "near_pairs_by_distance": near_pairs,
        "clusters_by_center_distance": cluster_rows,
    }

    out_path = Path(args.output).resolve() if args.output else in_path.parent / "analysis_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # concise terminal summary
    print(f"[analyze-object] input: {in_path}")
    print(f"[analyze-object] output: {out_path}")
    print(
        "[analyze-object] scores: "
        f"min={report['score_stats']['min']:.4f}, "
        f"mean={report['score_stats']['mean']:.4f}, "
        f"max={report['score_stats']['max']:.4f}, "
        f"std={report['score_stats']['std']:.4f}"
    )
    print(f"[analyze-object] near_pairs (d<={args.near_pair_distance}m): {len(near_pairs)}")
    print(f"[analyze-object] clusters (threshold={args.cluster_distance}m): {len(cluster_rows)}")
    top_clusters = sorted(cluster_rows, key=lambda x: x["size"], reverse=True)[:5]
    for c in top_clusters:
        print(
            f"  - cluster#{c['cluster_id']} size={c['size']} "
            f"obj_ids={c['members_object_id']} mean_score={c['mean_score']:.4f}"
        )


if __name__ == "__main__":
    main()

