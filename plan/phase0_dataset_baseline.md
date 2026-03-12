# Phase 0: Dataset Exploration & Naive Baseline

## Goal

Understand the exact format of `allenai/Molmo2-VideoCountEval` before writing any framework
code, and establish a naive single-shot VLM baseline to measure what RVLM adds.

## Tasks

### 0.1 — Install prerequisites

```bash
uv add datasets huggingface_hub opencv-python-headless pillow decord yt-dlp
```

Add to `pyproject.toml` as an optional extra:
```toml
[project.optional-dependencies]
video = ["opencv-python-headless", "pillow", "decord", "yt-dlp"]
```

### 0.2 — Dataset inspection script

Create `scripts/inspect_dataset.py`:
- Load `allenai/Molmo2-VideoCountEval` with `datasets.load_dataset`
- Print schema (column names, dtypes)
- Print 3 example rows with all fields
- Report: video source distribution (YouTube vs MammalNet), video length stats
  (min/mean/max duration), count value distribution (histogram), question type variety

Key questions to answer:
- Are videos stored as bytes, file paths, or YouTube URLs?
- Is the ground truth a scalar integer, a list of points, or something else?
- What is the exact MVC accuracy threshold (off-by-1? exact match?)?
  Cross-check with the Molmo2 tech report (arXiv 2601.10611, §Evaluation).
- Do questions ask for scalar counts only, or also "where"/"which frame"?
- How many examples are in the validation split?

### 0.3 — Video caching utility

Create `benchmarks/videocount/dataset.py`:

```python
def load_dataset_split(split="validation") -> datasets.Dataset:
    """Load Molmo2-VideoCountEval from HuggingFace."""

def cache_video(example: dict, cache_dir: str = "/tmp/rvlm_videos") -> str:
    """
    Ensure the video for this example is available locally.
    - If video is stored as bytes: write to cache_dir/<id>.mp4
    - If video is a YouTube URL: use yt-dlp to download (skip if already cached)
    Returns the local file path.
    """
```

Cache to disk so repeated eval runs do not re-download.

### 0.4 — Naive single-shot baseline

Create `benchmarks/videocount/baselines/naive_singleshot.py`:

Strategy: sample N frames uniformly, encode as base64, make one GPT-4o call asking
"How many <object> are in this video? Answer with a single integer."

Run configurations:
- `naive-8f`: 8 frames uniform
- `naive-16f`: 16 frames uniform
- `naive-32f`: 32 frames uniform

For each: record predicted count, expected count, and latency per example.

```python
def run_naive_baseline(
    dataset,
    model: str = "gpt-4o",
    n_frames: int = 16,
    output_path: str = "results/naive_16f.jsonl",
) -> dict:
    """Run single-shot baseline and write per-example JSONL results."""
```

### 0.5 — Baseline metrics report

Create `benchmarks/videocount/metrics.py`:

```python
def mvc_accuracy(predictions, ground_truth, threshold: int = 1) -> float:
    """Fraction of examples where |pred - gt| <= threshold."""

def mae(predictions, ground_truth) -> float:
    """Mean absolute error."""

def within_k_accuracy(predictions, ground_truth, k: int) -> float:
    """Fraction where |pred - gt| <= k."""

def report(results_path: str) -> None:
    """
    Load JSONL results file and print:
    - Overall MVC accuracy (threshold=1)
    - MAE, RMSE
    - Within-1, within-2, within-5 accuracy
    - Breakdown by video source (YouTube / MammalNet)
    - Breakdown by count range: [0-5], [6-20], [21+]
    - Failure mode analysis: over-count rate vs under-count rate
    """
```

## Deliverables

- [ ] Schema documented (update this file with findings)
- [ ] `scripts/inspect_dataset.py` runnable
- [ ] `benchmarks/videocount/dataset.py` with caching
- [ ] `benchmarks/videocount/metrics.py` with all metric functions
- [ ] Naive baseline results for 8f / 16f / 32f on GPT-4o
- [ ] Confirmed: MVC accuracy threshold definition

## Findings (Confirmed by Inspection)

- **533 examples**, single `val` split (no train split in this HF dataset)
- **No embedded video bytes** — only `video_id` (YouTube ID), `clip_start`, `clip_end` (seconds)
- **MammalNet IDs are also YouTube IDs** — all 533 downloadable via yt-dlp
- **Ground truth**: `count` field, `int64`, range 0–58, mean 12.15, median 10.0
- **Clip durations**: mean 30.4s, max 63s — no clip exceeds 2 minutes
- **Fields**: `video_id`, `question`, `label`, `count`, `category`, `video_duration`, `video_source`, `clip_start`, `clip_end`
- **Categories**: object (426), action/event (58), animal (49)
- **Sources**: youtube (484), MammalNet (49)
- **Availability**: ~93% downloadable; ~3% unavailable (deleted/private), ~3% other failures
- **Unique video IDs**: 501 (some videos have multiple examples with different questions)
- **Estimated download size**: ~1.5–2 GB for all clips at 720p
- **MVC accuracy threshold**: exact match (threshold=0) based on Molmo2 paper
  (also report within-1 for comparison with looser benchmarks)
- **Dev set**: 50 stratified examples saved to `benchmarks/videocount/dev_indices.json`
