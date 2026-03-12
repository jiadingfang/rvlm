"""
Phase 0: Naive single-shot VLM baseline for video object counting.

Strategy: sample N frames uniformly from the clip, encode as base64,
send in a single API call asking for a count. No REPL, no iteration.

Usage:
    uv run python -m benchmarks.videocount.baselines.naive_singleshot \
        --model gpt-4o --n-frames 16 --max-examples 50 \
        --dev-only --output results/naive_16f_dev.jsonl
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from benchmarks.videocount.dataset import cache_video, load_dataset_split, load_manifest


def sample_frames_uniform(video_path: str, n: int) -> list[np.ndarray]:
    """Sample n frames uniformly from the video. Returns RGB uint8 arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n = min(n, total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def frame_to_base64(frame: np.ndarray, max_side: int = 512) -> str:
    """Resize frame and encode as base64 JPEG."""
    from PIL import Image

    img = Image.fromarray(frame)
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def parse_count(response: str) -> int:
    """Extract integer from model response. Returns -1 on failure."""
    matches = re.findall(r"\b\d+\b", response)
    if not matches:
        return -1
    # Prefer the last integer (models tend to put final answer last)
    return int(matches[-1])


def call_openai(frames: list[np.ndarray], question: str, model: str, max_tokens: int = 64) -> tuple[str, dict]:
    """Call OpenAI API with frames + question. Returns (response_text, usage_dict)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    content = []
    for frame in frames:
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    content.append({
        "type": "text",
        "text": (
            f"These are {len(frames)} uniformly sampled frames from a video clip. "
            f"{question} "
            "Answer with a single integer only. Do not explain."
        ),
    })

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "model": model,
    }
    return text, usage


def call_anthropic(frames: list[np.ndarray], question: str, model: str, max_tokens: int = 64) -> tuple[str, dict]:
    """Call Anthropic API with frames + question."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    content = []
    # Anthropic limit: 20 images per request
    for frame in frames[:20]:
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({
        "type": "text",
        "text": (
            f"These are {len(frames[:20])} uniformly sampled frames from a video clip. "
            f"{question} "
            "Answer with a single integer only. Do not explain."
        ),
    })

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text.strip()
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "model": model,
    }
    return text, usage


def call_gemini(frames: list[np.ndarray], question: str, model: str, max_tokens: int = 64) -> tuple[str, dict]:
    """Call Gemini API with frames as inline images."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    parts = []
    for frame in frames:
        b64 = frame_to_base64(frame)
        img_bytes = base64.b64decode(b64)
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    parts.append(types.Part.from_text(
        text=f"These are {len(frames)} uniformly sampled frames from a video clip. "
        f"{question} "
        "Answer with a single integer only. Do not explain."
    ))

    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    text = resp.text
    if text is None:
        raise RuntimeError(f"Gemini returned empty response (finish_reason={resp.candidates[0].finish_reason if resp.candidates else 'unknown'})")
    text = text.strip()
    usage = {
        "input_tokens": resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0,
        "output_tokens": resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0,
        "model": model,
    }
    return text, usage


PROVIDER_DISPATCH = {
    "gpt-4o": ("openai", call_openai),
    "gpt-4o-mini": ("openai", call_openai),
    "gpt-4.1": ("openai", call_openai),
    "gpt-4.1-mini": ("openai", call_openai),
    "claude-opus-4-6": ("anthropic", call_anthropic),
    "claude-sonnet-4-6": ("anthropic", call_anthropic),
    "claude-haiku-4-5-20251001": ("anthropic", call_anthropic),
    "gemini-2.0-flash": ("gemini", call_gemini),
    "gemini-2.5-flash": ("gemini", call_gemini),
    "gemini-2.5-flash-preview-04-17": ("gemini", call_gemini),
    "gemini-2.5-pro": ("gemini", call_gemini),
    "gemini-3-flash-preview": ("gemini", call_gemini),
}


def run_naive_baseline(
    model: str = "gpt-4o",
    n_frames: int = 16,
    split: str = "val",
    max_examples: int | None = None,
    dev_only: bool = False,
    dev_indices_path: str = "benchmarks/videocount/dev_indices.json",
    output_path: str | None = None,
    cache_dir: str = "/tmp/rvlm_videos",
    resume: bool = True,
    verbose: bool = True,
) -> str:
    """
    Run the naive single-shot baseline. Returns path to results JSONL.
    """
    if model not in PROVIDER_DISPATCH:
        raise ValueError(f"Unknown model '{model}'. Supported: {list(PROVIDER_DISPATCH)}")
    _, call_fn = PROVIDER_DISPATCH[model]

    # Determine output path
    if output_path is None:
        tag = f"naive_{n_frames}f_{model.replace('-', '_').replace('.', '_')}"
        output_path = f"benchmarks/videocount/results/{tag}.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset_split(split)

    # Filter to dev set or max_examples
    if dev_only:
        with open(dev_indices_path) as f:
            indices = json.load(f)
        ds = ds.select(indices)
    elif max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    # Load existing results for resume
    done_ids = set()
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r.get("example_id"))
                except Exception:
                    pass
        if verbose and done_ids:
            print(f"Resuming: {len(done_ids)} examples already done")

    total = len(ds)
    n_done = 0
    n_error = 0

    with open(output_path, "a") as out_f:
        for i, ex in enumerate(ds):
            example_id = ex["video_id"] + f"_{ex['clip_start']:.0f}_{ex['clip_end']:.0f}"
            if example_id in done_ids:
                continue

            video_path = cache_video(ex, cache_dir=cache_dir, verbose=False)
            if video_path is None:
                record = {
                    "example_id": example_id,
                    "video_id": ex["video_id"],
                    "question": ex["question"],
                    "label": ex["label"],
                    "expected": ex["count"],
                    "predicted": -1,
                    "raw_response": None,
                    "error": "video_unavailable",
                    "video_source": ex["video_source"],
                    "clip_duration": ex["clip_end"] - ex["clip_start"],
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_error += 1
                if verbose:
                    print(f"[{i+1}/{total}] SKIP (unavailable): {ex['video_id']}")
                continue

            t0 = time.time()
            try:
                frames = sample_frames_uniform(video_path, n=n_frames)
                if not frames:
                    raise RuntimeError("No frames extracted")
                raw_response, usage = call_fn(frames, ex["question"], model)
                predicted = parse_count(raw_response)
                elapsed = time.time() - t0
                record = {
                    "example_id": example_id,
                    "video_id": ex["video_id"],
                    "question": ex["question"],
                    "label": ex["label"],
                    "expected": ex["count"],
                    "predicted": predicted,
                    "raw_response": raw_response,
                    "error": None,
                    "execution_time": elapsed,
                    "usage": usage,
                    "video_source": ex["video_source"],
                    "clip_duration": ex["clip_end"] - ex["clip_start"],
                    "n_frames": len(frames),
                }
                n_done += 1
                if verbose:
                    status = "✓" if abs(predicted - ex["count"]) <= 1 else "✗"
                    print(f"[{i+1}/{total}] {status} expected={ex['count']:3d} predicted={predicted:3d} | {ex['label'][:40]}")
            except Exception as e:
                elapsed = time.time() - t0
                record = {
                    "example_id": example_id,
                    "video_id": ex["video_id"],
                    "question": ex["question"],
                    "label": ex["label"],
                    "expected": ex["count"],
                    "predicted": -1,
                    "raw_response": None,
                    "error": str(e),
                    "execution_time": elapsed,
                    "video_source": ex["video_source"],
                    "clip_duration": ex["clip_end"] - ex["clip_start"],
                }
                n_error += 1
                if verbose:
                    print(f"[{i+1}/{total}] ERROR: {e} | {ex['video_id']}")

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    if verbose:
        print(f"\nDone. {n_done} evaluated, {n_error} skipped/errored. Results: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Naive single-shot VLM baseline")
    parser.add_argument("--model", default="gpt-4o", choices=list(PROVIDER_DISPATCH))
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dev-only", action="store_true", help="Run on 50-example dev set only")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--cache-dir", default="/tmp/rvlm_videos")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_naive_baseline(
        model=args.model,
        n_frames=args.n_frames,
        max_examples=args.max_examples,
        dev_only=args.dev_only,
        output_path=args.output,
        cache_dir=args.cache_dir,
        resume=not args.no_resume,
        verbose=True,
    )


if __name__ == "__main__":
    main()
