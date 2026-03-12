"""
Phase 0: Inspect allenai/Molmo2-VideoCountEval dataset structure.
Answers the open questions from plan/phase0_dataset_baseline.md.
"""
import json
import sys
from collections import Counter

import numpy as np


def inspect():
    from datasets import load_dataset

    print("=" * 70)
    print("Loading allenai/Molmo2-VideoCountEval ...")
    print("=" * 70)

    ds = load_dataset("allenai/Molmo2-VideoCountEval", trust_remote_code=True)
    print(f"\nDataset splits: {list(ds.keys())}")

    for split_name, split in ds.items():
        print(f"\n{'='*70}")
        print(f"SPLIT: {split_name}  ({len(split)} examples)")
        print(f"{'='*70}")

        # Schema
        print("\n--- Schema (features) ---")
        for col, feat in split.features.items():
            print(f"  {col}: {feat}")

        # First 3 examples (non-binary fields)
        print("\n--- First 3 examples ---")
        for i in range(min(3, len(split))):
            ex = split[i]
            print(f"\n  Example {i}:")
            for k, v in ex.items():
                if isinstance(v, bytes):
                    print(f"    {k}: <bytes len={len(v)}>")
                elif isinstance(v, dict) and "bytes" in v:
                    b = v.get("bytes", b"")
                    print(f"    {k}: <video/image dict, bytes len={len(b) if b else 'None'}, path={v.get('path','?')}>")
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                    print(f"    {k}: list[{type(v[0]).__name__}] len={len(v)}, sample={v[:4]}")
                else:
                    val_str = str(v)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"    {k}: {val_str}")

        # Count/answer distribution
        print("\n--- Count / answer distribution ---")
        count_vals = []
        for ex in split:
            for key in ["count", "answer", "label", "n_objects", "num_objects", "target"]:
                if key in ex:
                    try:
                        count_vals.append(int(ex[key]))
                    except (TypeError, ValueError):
                        count_vals.append(ex[key])
                    break

        if count_vals:
            if all(isinstance(c, int) for c in count_vals):
                arr = np.array(count_vals)
                print(f"  n={len(arr)}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}, median={np.median(arr):.1f}")
                bins = [0, 1, 2, 3, 5, 10, 20, 50, 9999]
                labels = ["0", "1", "2", "3-5", "6-10", "11-20", "21-50", "50+"]
                for lo, hi, lbl in zip(bins, bins[1:], labels):
                    if lo == 0:
                        n = np.sum(arr == 0)
                    elif hi == 9999:
                        n = np.sum(arr > lo)
                    else:
                        n = np.sum((arr >= lo) & (arr <= hi))
                    print(f"    [{lbl:>6}]: {n:4d} ({100*n/len(arr):.1f}%)")
            else:
                c = Counter(str(v) for v in count_vals[:20])
                print(f"  Sample (non-integer?): {dict(c)}")

        # Source / metadata distribution
        print("\n--- Metadata field distributions ---")
        for key in ["source", "video_source", "category", "split", "dataset", "type"]:
            if key in split.column_names:
                vals = [ex[key] for ex in split]
                c = Counter(vals)
                print(f"  {key}: {dict(c.most_common(10))}")

        # Video field presence
        print("\n--- Video field analysis ---")
        video_keys = [k for k in split.column_names if "video" in k.lower() or "frame" in k.lower() or "image" in k.lower()]
        print(f"  Video-related columns: {video_keys}")

        if len(split) > 0:
            ex0 = split[0]
            for vk in video_keys:
                v = ex0[vk]
                if isinstance(v, dict):
                    print(f"  {vk} (dict keys): {list(v.keys())}")
                    if "bytes" in v and v["bytes"]:
                        print(f"    bytes length: {len(v['bytes'])}")
                    if "path" in v:
                        print(f"    path: {v['path']}")
                elif isinstance(v, list):
                    print(f"  {vk}: list of {len(v)} items")
                    if v:
                        print(f"    first item type: {type(v[0]).__name__}")
                elif isinstance(v, bytes):
                    print(f"  {vk}: raw bytes, len={len(v)}")
                else:
                    print(f"  {vk}: {type(v).__name__} = {str(v)[:100]}")

        # Question / text fields
        print("\n--- Question/text field samples ---")
        text_keys = [k for k in split.column_names if "question" in k.lower() or "query" in k.lower() or "text" in k.lower() or "prompt" in k.lower()]
        print(f"  Text-related columns: {text_keys}")
        for ex in split.select(range(min(5, len(split)))):
            for tk in text_keys:
                print(f"  [{tk}]: {str(ex[tk])[:200]}")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    return ds


if __name__ == "__main__":
    ds = inspect()
