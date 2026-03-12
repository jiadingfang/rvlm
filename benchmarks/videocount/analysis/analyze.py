"""
Analyze and compare evaluation results from JSONL result files.

Usage:
    uv run python -m benchmarks.videocount.analysis.analyze \
        --results benchmarks/videocount/results/naive_16f_gpt_4o.jsonl

    # Compare multiple runs:
    uv run python -m benchmarks.videocount.analysis.analyze \
        --results results/naive_8f*.jsonl results/naive_16f*.jsonl results/rvlm*.jsonl
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from benchmarks.videocount.metrics import report


def compare(result_paths: list[str], threshold: int = 0) -> None:
    """Print a side-by-side comparison table."""
    all_metrics = {}
    for path in result_paths:
        name = Path(path).stem
        m = report(path, threshold=threshold)
        if m:
            all_metrics[name] = m

    if len(all_metrics) < 2:
        return

    keys = ["exact_match", "within_1", "within_2", "within_5", "mae", "rmse", "n_valid", "n_error"]
    labels = ["Exact", "Within-1", "Within-2", "Within-5", "MAE", "RMSE", "N valid", "N error"]

    col_w = 12
    header = f"{'Metric':>12}" + "".join(f"{n[:col_w]:>{col_w}}" for n in all_metrics)
    print("\n" + "=" * len(header))
    print("COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, label in zip(keys, labels):
        row = f"{label:>12}"
        for name, m in all_metrics.items():
            val = m.get(key)
            if val is None:
                row += f"{'—':>{col_w}}"
            elif isinstance(val, float) and key not in ("mae", "rmse", "n_valid", "n_error"):
                row += f"{val:>{col_w}.1%}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.2f}"
            else:
                row += f"{int(val):>{col_w}}"
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Analyze RVLM eval results")
    parser.add_argument("--results", nargs="+", required=True,
                        help="JSONL result files or glob patterns")
    parser.add_argument("--threshold", type=int, default=0,
                        help="MVC accuracy threshold (default: 0 = exact match)")
    args = parser.parse_args()

    # Expand globs
    paths = []
    for pattern in args.results:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])

    if not paths:
        print("No result files found.")
        sys.exit(1)

    if len(paths) == 1:
        report(paths[0], threshold=args.threshold)
    else:
        for p in paths:
            report(p, threshold=args.threshold)
        compare(paths, threshold=args.threshold)


if __name__ == "__main__":
    main()
