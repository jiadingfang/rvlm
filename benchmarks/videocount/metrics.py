"""
Metrics for Molmo2-VideoCountEval video object counting benchmark.

MVC (Molmo2-VideoCount) accuracy: exact match (threshold=0).
The Molmo2 tech report evaluates with exact integer match.
We also report within-k for analysis.
"""
import json
import math
from pathlib import Path


def mvc_accuracy(predictions: list[int], ground_truth: list[int], threshold: int = 0) -> float:
    """Fraction of examples where |pred - gt| <= threshold."""
    assert len(predictions) == len(ground_truth)
    correct = sum(abs(p - g) <= threshold for p, g in zip(predictions, ground_truth))
    return correct / len(predictions) if predictions else 0.0


def mae(predictions: list[int], ground_truth: list[int]) -> float:
    """Mean absolute error."""
    assert len(predictions) == len(ground_truth)
    return sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(predictions)


def rmse(predictions: list[int], ground_truth: list[int]) -> float:
    """Root mean squared error."""
    assert len(predictions) == len(ground_truth)
    mse = sum((p - g) ** 2 for p, g in zip(predictions, ground_truth)) / len(predictions)
    return math.sqrt(mse)


def within_k_accuracy(predictions: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of examples where |pred - gt| <= k."""
    return mvc_accuracy(predictions, ground_truth, threshold=k)


def report(results_path: str | Path, threshold: int = 0) -> dict:
    """
    Load a JSONL results file and compute all metrics.

    Each line in the JSONL must have: expected (int), predicted (int).
    predicted == -1 means parse error.

    Returns a nested dict of metrics and prints a summary table.
    """
    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return {}

    # Separate parseable vs error
    valid = [r for r in records if r.get("predicted", -1) != -1 and r.get("error") is None]
    errors = [r for r in records if r.get("predicted", -1) == -1 or r.get("error") is not None]

    preds = [r["predicted"] for r in valid]
    gts = [r["expected"] for r in valid]
    n = len(records)
    n_valid = len(valid)
    n_error = len(errors)

    print(f"\n{'='*60}")
    print(f"Results: {Path(results_path).name}")
    print(f"{'='*60}")
    print(f"Total examples : {n}")
    print(f"Parseable      : {n_valid} ({100*n_valid/n:.1f}%)")
    print(f"Parse errors   : {n_error} ({100*n_error/n:.1f}%)")

    if not valid:
        print("No valid predictions to evaluate.")
        return {"n": n, "n_valid": 0, "n_error": n_error}

    mvc = mvc_accuracy(preds, gts, threshold=threshold)
    _mae = mae(preds, gts)
    _rmse = rmse(preds, gts)
    w1 = within_k_accuracy(preds, gts, 1)
    w2 = within_k_accuracy(preds, gts, 2)
    w5 = within_k_accuracy(preds, gts, 5)
    over = sum(p > g for p, g in zip(preds, gts)) / n_valid
    under = sum(p < g for p, g in zip(preds, gts)) / n_valid
    exact = sum(p == g for p, g in zip(preds, gts)) / n_valid

    print(f"\n--- Overall Metrics (on {n_valid} valid) ---")
    print(f"Exact match    : {exact:.3f} ({100*exact:.1f}%)")
    print(f"MVC accuracy (threshold={threshold}): {mvc:.3f} ({100*mvc:.1f}%)")
    print(f"Within-1       : {w1:.3f} ({100*w1:.1f}%)")
    print(f"Within-2       : {w2:.3f} ({100*w2:.1f}%)")
    print(f"Within-5       : {w5:.3f} ({100*w5:.1f}%)")
    print(f"MAE            : {_mae:.2f}")
    print(f"RMSE           : {_rmse:.2f}")
    print(f"Over-count rate: {over:.3f} ({100*over:.1f}%)")
    print(f"Under-count    : {under:.3f} ({100*under:.1f}%)")

    # By count range
    print(f"\n--- Breakdown by Ground-Truth Count Range ---")
    ranges = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"),
              (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 9999, "51+")]
    print(f"  {'Range':>8}  {'n':>5}  {'Exact':>7}  {'Within1':>8}  {'MAE':>7}")
    range_metrics = {}
    for lo, hi, lbl in ranges:
        idx = [i for i, g in enumerate(gts) if lo <= g <= hi]
        if not idx:
            continue
        rp = [preds[i] for i in idx]
        rg = [gts[i] for i in idx]
        rexact = sum(p == g for p, g in zip(rp, rg)) / len(rp)
        rw1 = within_k_accuracy(rp, rg, 1)
        rmae_ = mae(rp, rg)
        print(f"  {lbl:>8}  {len(idx):>5}  {rexact:>6.1%}  {rw1:>7.1%}  {rmae_:>7.2f}")
        range_metrics[lbl] = {"n": len(idx), "exact": rexact, "within_1": rw1, "mae": rmae_}

    # By video source
    sources = set(r.get("video_source", "unknown") for r in valid)
    if len(sources) > 1:
        print(f"\n--- Breakdown by Video Source ---")
        print(f"  {'Source':>12}  {'n':>5}  {'Exact':>7}  {'Within1':>8}  {'MAE':>7}")
        for src in sorted(sources):
            idx = [i for i, r in enumerate(valid) if r.get("video_source") == src]
            rp = [preds[i] for i in idx]
            rg = [gts[i] for i in idx]
            rexact = sum(p == g for p, g in zip(rp, rg)) / len(rp)
            rw1 = within_k_accuracy(rp, rg, 1)
            rmae_ = mae(rp, rg)
            print(f"  {src:>12}  {len(idx):>5}  {rexact:>6.1%}  {rw1:>7.1%}  {rmae_:>7.2f}")

    # By clip duration bucket
    durations = [r.get("clip_duration") for r in valid]
    if any(d is not None for d in durations):
        print(f"\n--- Breakdown by Clip Duration ---")
        print(f"  {'Duration':>12}  {'n':>5}  {'Exact':>7}  {'Within1':>8}  {'MAE':>7}")
        dur_ranges = [(0, 15, "0-15s"), (15, 30, "15-30s"), (30, 60, "30-60s"), (60, 9999, "60s+")]
        for lo, hi, lbl in dur_ranges:
            idx = [i for i, d in enumerate(durations) if d is not None and lo <= d < hi]
            if not idx:
                continue
            rp = [preds[i] for i in idx]
            rg = [gts[i] for i in idx]
            rexact = sum(p == g for p, g in zip(rp, rg)) / len(rp)
            rw1 = within_k_accuracy(rp, rg, 1)
            rmae_ = mae(rp, rg)
            print(f"  {lbl:>12}  {len(idx):>5}  {rexact:>6.1%}  {rw1:>7.1%}  {rmae_:>7.2f}")

    print(f"{'='*60}\n")

    return {
        "n": n, "n_valid": n_valid, "n_error": n_error,
        "exact_match": exact,
        "mvc_accuracy": mvc,
        "within_1": w1, "within_2": w2, "within_5": w5,
        "mae": _mae, "rmse": _rmse,
        "over_count_rate": over, "under_count_rate": under,
        "by_range": range_metrics,
    }
