# Phase 4: Evaluation Harness

## Goal

A reproducible, resumable evaluation pipeline that runs RVLM on the full
`Molmo2-VideoCountEval` benchmark and produces structured results for analysis.

## File Structure

```
benchmarks/
  videocount/
    __init__.py
    dataset.py          # HuggingFace loader + video caching  (from Phase 0)
    metrics.py          # MVC accuracy, MAE, within-k         (from Phase 0)
    eval.py             # Main eval loop
    configs/
      gpt4o_depth1.yaml
      gpt4o_depth2.yaml
      gemini_depth1.yaml
      claude_depth1.yaml
      naive_16f.yaml     # non-RVLM baseline configs
    results/             # .jsonl output files (git-ignored)
    dev_indices.json     # 50-example dev set (git-tracked)
    analysis/
      analyze.py         # Load results, print tables
      plot.py            # Generate figures
```

## Tasks

### 4.1 — Config Format

Each run is fully described by a YAML config. Example `configs/gpt4o_depth1.yaml`:

```yaml
name: gpt4o_depth1
description: "GPT-4o, RVLM depth=1, counting prompt V1"

backend: gpt4o
backend_kwargs:
  model_name: gpt-4o
  max_tokens: 1024

vlm_backend: gpt4o              # VLM used for vlm_count tool calls
vlm_backend_kwargs:
  model_name: gpt-4o
  detail: low

environment: local
max_depth: 1
max_iterations: 8
max_tokens: 200000              # stop if total tokens exceed this

prompt_variant: COUNTING_PROMPT_V1

dataset:
  name: allenai/Molmo2-VideoCountEval
  split: validation
  max_examples: null            # null = full dataset; set int for quick runs
  video_cache_dir: /tmp/rvlm_videos

output:
  results_dir: benchmarks/videocount/results
  log_trajectories: true        # save full RLMLogger JSONL per example
```

Load and validate configs with `pydantic` or plain dataclasses (no new dep needed).

### 4.2 — `benchmarks/videocount/eval.py`

```python
def run_eval(config_path: str, resume: bool = True) -> str:
    """
    Run evaluation and return path to results JSONL file.

    Args:
        config_path: Path to YAML config.
        resume: If True and a partial results file exists, skip already-evaluated examples.
    """
```

**Per-example loop**:
```python
for i, example in enumerate(dataset):
    example_id = example["id"]  # or hash of video + question
    if resume and already_done(example_id, results_file):
        continue

    video_path = cache_video(example, config.dataset.video_cache_dir)
    question = example["question"]
    expected = int(example["count"])

    try:
        result = rvlm.completion({
            "video_path": video_path,
            "question": question,
        })
        predicted = parse_count_from_response(result.response)
        error = None
    except Exception as e:
        predicted = -1
        error = str(e)

    record = {
        "example_id": example_id,
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "error": error,
        "execution_time": result.execution_time if result else None,
        "usage": result.usage_summary.to_dict() if result else None,
        "trajectory_path": save_trajectory(result, example_id) if config.output.log_trajectories else None,
    }
    append_jsonl(results_file, record)
    print(f"[{i+1}/{total}] expected={expected}, predicted={predicted}")
```

**Resume mechanism**: Check the output JSONL for existing `example_id` entries before
running. This allows interrupting and restarting without re-running completed examples.

### 4.3 — `parse_count_from_response`

The model may return:
- `"7"` — ideal
- `"The answer is 7."` — extract integer
- `"7.0"` — round to int
- `"approximately 7"` — extract 7
- `"I count 6-8 sheep"` — take the midpoint or lower bound (with logging)
- `"0"` — valid
- No integer at all — log as parse error, record predicted=-1

```python
def parse_count_from_response(response: str) -> int:
    """
    Extract integer count from model response.
    Priority: last integer in response (models tend to put final answer last).
    Raises ValueError if no integer found.
    """
    import re
    matches = re.findall(r'\b\d+\b', response)
    if not matches:
        raise ValueError(f"No integer found in response: {response!r}")
    return int(matches[-1])
```

### 4.4 — Metrics report (`benchmarks/videocount/metrics.py`)

Extended from Phase 0 with additional breakdowns:

```python
def full_report(results_path: str, threshold: int = 1) -> dict:
    """
    Load results JSONL and compute:

    Overall:
      - mvc_accuracy (|pred - gt| <= threshold)
      - mae, rmse
      - within_1, within_2, within_5 accuracy
      - parse_error_rate (predicted == -1)
      - over_count_rate, under_count_rate

    By count range:
      - [0], [1-2], [3-5], [6-10], [11-20], [21+]

    By video source:
      - youtube, mammalnet (if available in metadata)

    By example duration bucket:
      - [0-15s], [16-60s], [61-300s], [300s+]

    Cost:
      - total_input_tokens, total_output_tokens, total_cost_usd
      - avg tokens per example
      - avg execution_time per example

    Returns nested dict. Also prints formatted table to stdout.
    """
```

### 4.5 — `benchmarks/videocount/analysis/analyze.py`

CLI wrapper for `full_report`:
```bash
uv run python -m benchmarks.videocount.analysis.analyze \
  --results benchmarks/videocount/results/gpt4o_depth1.jsonl \
  --compare benchmarks/videocount/results/naive_16f.jsonl
```

Prints a comparison table and delta for each metric.

### 4.6 — Run Commands

Add to `Makefile`:
```makefile
eval-gpt4o-depth1:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/gpt4o_depth1.yaml

eval-dev:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/gpt4o_depth1.yaml \
	  --max-examples 50 \
	  --example-ids benchmarks/videocount/dev_indices.json

analyze:
	uv run python -m benchmarks.videocount.analysis.analyze \
	  --results benchmarks/videocount/results/
```

### 4.7 — Cost Guardrails

Before running full eval, add a dry-run mode that estimates cost without making API calls:

```python
def estimate_cost(config, n_examples: int = None) -> dict:
    """
    Estimate total API cost based on:
    - n_examples (default: full dataset size)
    - avg frames per example (from Phase 0 metadata)
    - avg text tokens per example
    - detail level
    Returns: {"estimated_cost_usd": float, "estimated_tokens": int}
    """
```

Require `--confirm` flag or print estimated cost and prompt for confirmation before
running any eval on > 100 examples.

## Deliverables

- [ ] `benchmarks/videocount/eval.py` with resume support
- [ ] Config files for all planned run configurations
- [ ] `parse_count_from_response` with test cases
- [ ] `full_report` with all metric breakdowns
- [ ] `analyze.py` CLI tool
- [ ] Cost estimation / dry-run mode
- [ ] Makefile targets for eval and analysis
- [ ] Full eval run completed on dev set (50 examples) for at least one config
- [ ] Results JSONL saved (not git-tracked; add `benchmarks/videocount/results/` to `.gitignore`)

## Expected Timeline

- Dev set run (50 examples, GPT-4o depth=1): ~2-3 hours, ~$5-15 API cost
- Full benchmark run (est. size TBD from Phase 0): estimate after Phase 0
