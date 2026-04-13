import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SR/AR and SPL from vlmcor result json files.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="output_logs/anchor/vlmcor/refhm3d_seq_vlmcor_refine1_*.json",
        help="Glob pattern for result json files.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[temp-analysis] no files matched: {args.pattern}")
        return

    total = 0
    sr_sum = 0.0
    spl_sum = 0.0

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("sequence", [])
        n = len(seq)
        cur_sr = sum(float(x.get("sr", 0.0)) for x in seq)
        cur_spl = sum(float(x.get("spl", 0.0)) for x in seq)
        total += n
        sr_sum += cur_sr
        spl_sum += cur_spl
        print(
            f"[temp-analysis] {os.path.basename(fp)}: tasks={n}, "
            f"avg_ar(sr)={(cur_sr / n if n else 0.0):.6f}, avg_spl={(cur_spl / n if n else 0.0):.6f}"
        )

    avg_ar = sr_sum / total if total else 0.0
    avg_spl = spl_sum / total if total else 0.0
    print("-" * 72)
    print(f"[temp-analysis] files={len(files)}, total_tasks={total}")
    print(f"[temp-analysis] overall_avg_ar(sr)={avg_ar:.6f}")
    print(f"[temp-analysis] overall_avg_spl={avg_spl:.6f}")


if __name__ == "__main__":
    main()

