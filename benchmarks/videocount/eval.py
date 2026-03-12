"""
RVLM evaluation harness for Molmo2-VideoCountEval video counting benchmark.

Supports both naive single-shot baselines and full RVLM recursive runs.
Results are written as per-example JSONL with resume support.

Usage:
    # RVLM eval on dev set
    uv run python -m benchmarks.videocount.eval \
        --config benchmarks/videocount/configs/gpt4o_depth1.yaml

    # Override max examples for quick test
    uv run python -m benchmarks.videocount.eval \
        --config benchmarks/videocount/configs/gpt4o_depth1.yaml \
        --max-examples 5

    # Dry run (estimate cost only)
    uv run python -m benchmarks.videocount.eval \
        --config benchmarks/videocount/configs/gpt4o_depth1.yaml --dry-run
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.videocount.dataset import cache_video, load_dataset_split


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class DatasetConfig:
    name: str = "allenai/Molmo2-VideoCountEval"
    split: str = "val"
    max_examples: int | None = None
    video_cache_dir: str = "/tmp/rvlm_videos"
    dev_indices_path: str = "benchmarks/videocount/dev_indices.json"
    dev_only: bool = False


@dataclass
class OutputConfig:
    results_dir: str = "benchmarks/videocount/results"
    log_trajectories: bool = True


@dataclass
class EvalConfig:
    name: str = "unnamed"
    description: str = ""

    # Mode: "naive" for single-shot baseline, "rvlm" for recursive eval
    mode: str = "rvlm"

    # Model used for the RLM reasoning loop (text LM)
    backend: str = "openai"
    backend_kwargs: dict = field(default_factory=lambda: {"model_name": "gpt-4o", "max_tokens": 1024})

    # VLM used for vlm_count / vlm_describe tool calls
    vlm_model: str = "gpt-4o"
    vlm_kwargs: dict = field(default_factory=dict)

    # RLM parameters
    max_depth: int = 1
    max_iterations: int = 8
    max_tokens: int | None = 200000

    # Prompt
    prompt_variant: str = "v1"

    # Naive baseline parameters (mode=naive only)
    n_frames: int = 16

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        dataset_raw = raw.pop("dataset", {})
        output_raw = raw.pop("output", {})

        config = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        config.dataset = DatasetConfig(**{k: v for k, v in dataset_raw.items() if k in DatasetConfig.__dataclass_fields__})
        config.output = OutputConfig(**{k: v for k, v in output_raw.items() if k in OutputConfig.__dataclass_fields__})
        return config


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_count_from_response(response: str) -> int:
    """
    Extract integer count from model response.
    Takes the last integer found (models tend to put final answer last).
    """
    matches = re.findall(r"\b\d+\b", response)
    if not matches:
        raise ValueError(f"No integer found in response: {response!r}")
    return int(matches[-1])


# ── Cost estimation ───────────────────────────────────────────────────────────

# Approximate cost per 1M tokens (input/output) by model
_COST_TABLE = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "image_low": 85},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "image_low": 85},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "image_low": 85},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "image_low": 259},
    "gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60, "image_low": 259},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "image_low": 1600},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "image_low": 1600},
}


def estimate_cost(config: EvalConfig, n_examples: int) -> dict:
    """Estimate total API cost without making API calls."""
    vlm_cost_info = _COST_TABLE.get(config.vlm_model, _COST_TABLE["gpt-4o"])
    backend_model = config.backend_kwargs.get("model_name", "gpt-4o")
    backend_cost_info = _COST_TABLE.get(backend_model, _COST_TABLE["gpt-4o"])

    if config.mode == "naive":
        # Naive: one VLM call per example
        img_tokens = config.n_frames * vlm_cost_info["image_low"]
        text_tokens = 100  # prompt + response
        input_per_ex = img_tokens + text_tokens
        output_per_ex = 20
        total_input = input_per_ex * n_examples
        total_output = output_per_ex * n_examples
        cost = (total_input / 1e6) * vlm_cost_info["input"] + (total_output / 1e6) * vlm_cost_info["output"]
    else:
        # RVLM: ~3-5 VLM calls + ~4-6 LM iterations per example
        avg_vlm_calls = 4
        avg_frames_per_call = 8
        vlm_img_tokens = avg_vlm_calls * avg_frames_per_call * vlm_cost_info["image_low"]
        vlm_text_tokens = avg_vlm_calls * 100
        vlm_output_tokens = avg_vlm_calls * 30

        avg_iterations = 5
        lm_input_per_iter = 2000  # growing context
        lm_output_per_iter = 500

        total_vlm_input = (vlm_img_tokens + vlm_text_tokens) * n_examples
        total_vlm_output = vlm_output_tokens * n_examples
        total_lm_input = avg_iterations * lm_input_per_iter * n_examples
        total_lm_output = avg_iterations * lm_output_per_iter * n_examples

        vlm_cost = (total_vlm_input / 1e6) * vlm_cost_info["input"] + (total_vlm_output / 1e6) * vlm_cost_info["output"]
        lm_cost = (total_lm_input / 1e6) * backend_cost_info["input"] + (total_lm_output / 1e6) * backend_cost_info["output"]
        cost = vlm_cost + lm_cost

    return {
        "n_examples": n_examples,
        "estimated_cost_usd": round(cost, 2),
        "model": config.vlm_model if config.mode == "naive" else backend_model,
        "mode": config.mode,
    }


# ── Eval runners ──────────────────────────────────────────────────────────────


def _load_done_ids(results_path: Path) -> set[str]:
    """Load already-completed example IDs from a results JSONL."""
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add(r.get("example_id"))
                    except Exception:
                        pass
    return done


def _make_example_id(ex: dict) -> str:
    return f"{ex['video_id']}_{ex['clip_start']:.0f}_{ex['clip_end']:.0f}"


def run_naive_eval(config: EvalConfig, resume: bool = True) -> str:
    """Run naive single-shot baseline using the eval config."""
    from benchmarks.videocount.baselines.naive_singleshot import (
        PROVIDER_DISPATCH,
        parse_count,
        sample_frames_uniform,
    )

    model = config.vlm_model
    if model not in PROVIDER_DISPATCH:
        raise ValueError(f"Unknown model for naive baseline: {model}")
    _, call_fn = PROVIDER_DISPATCH[model]

    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{config.name}.jsonl"

    ds = load_dataset_split(config.dataset.split)
    if config.dataset.dev_only:
        with open(config.dataset.dev_indices_path) as f:
            indices = json.load(f)
        ds = ds.select(indices)
    elif config.dataset.max_examples:
        ds = ds.select(range(min(config.dataset.max_examples, len(ds))))

    done_ids = _load_done_ids(results_path) if resume else set()
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")

    total = len(ds)
    n_done = 0
    n_error = 0

    with open(results_path, "a") as out_f:
        for i, ex in enumerate(ds):
            example_id = _make_example_id(ex)
            if example_id in done_ids:
                continue

            video_path = cache_video(ex, cache_dir=config.dataset.video_cache_dir, verbose=False)
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
                    "category": ex.get("category", ""),
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_error += 1
                print(f"[{i + 1}/{total}] SKIP (unavailable): {ex['video_id']}")
                continue

            t0 = time.time()
            try:
                frames = sample_frames_uniform(video_path, n=config.n_frames)
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
                    "category": ex.get("category", ""),
                    "n_frames": len(frames),
                    "config": config.name,
                }
                n_done += 1
                status = "OK" if abs(predicted - ex["count"]) <= 1 else "MISS"
                print(f"[{i + 1}/{total}] {status} expected={ex['count']:3d} predicted={predicted:3d} | {ex['label'][:50]}")
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
                    "category": ex.get("category", ""),
                    "config": config.name,
                }
                n_error += 1
                print(f"[{i + 1}/{total}] ERROR: {e}")

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"\nDone. {n_done} evaluated, {n_error} errors. Results: {results_path}")
    return str(results_path)


def run_rvlm_eval(config: EvalConfig, resume: bool = True) -> str:
    """Run RVLM recursive eval using the eval config."""
    from rlm.core.rlm import RLM
    from rlm.logger import RLMLogger
    from rvlm.clients import get_vlm_client
    from rvlm.utils.counting_prompts import PROMPT_VARIANTS
    from rvlm.utils.counting_tools import make_counting_tools

    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{config.name}.jsonl"
    trajectory_dir = results_dir / f"{config.name}_trajectories"
    if config.output.log_trajectories:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset_split(config.dataset.split)
    if config.dataset.dev_only:
        with open(config.dataset.dev_indices_path) as f:
            indices = json.load(f)
        ds = ds.select(indices)
    elif config.dataset.max_examples:
        ds = ds.select(range(min(config.dataset.max_examples, len(ds))))

    done_ids = _load_done_ids(results_path) if resume else set()
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")

    # Setup VLM client for counting tools
    vlm_client = get_vlm_client(config.vlm_model, **config.vlm_kwargs)
    counting_tools = make_counting_tools(vlm_client)

    # Get system prompt
    system_prompt = PROMPT_VARIANTS.get(config.prompt_variant, PROMPT_VARIANTS["v1"])

    total = len(ds)
    n_done = 0
    n_error = 0

    with open(results_path, "a") as out_f:
        for i, ex in enumerate(ds):
            example_id = _make_example_id(ex)
            if example_id in done_ids:
                continue

            video_path = cache_video(ex, cache_dir=config.dataset.video_cache_dir, verbose=False)
            if video_path is None:
                record = {
                    "example_id": example_id,
                    "video_id": ex["video_id"],
                    "question": ex["question"],
                    "label": ex["label"],
                    "expected": ex["count"],
                    "predicted": -1,
                    "error": "video_unavailable",
                    "video_source": ex["video_source"],
                    "clip_duration": ex["clip_end"] - ex["clip_start"],
                    "category": ex.get("category", ""),
                    "config": config.name,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_error += 1
                print(f"[{i + 1}/{total}] SKIP (unavailable): {ex['video_id']}")
                continue

            # Inject video_path and question as REPL globals
            tools = dict(counting_tools)
            tools["video_path"] = {"tool": video_path, "description": "Path to the input video file."}
            tools["question"] = {"tool": ex["question"], "description": "The counting question to answer."}

            logger = RLMLogger() if config.output.log_trajectories else None

            rlm = RLM(
                backend=config.backend,
                backend_kwargs=config.backend_kwargs,
                environment="local",
                max_depth=config.max_depth,
                max_iterations=config.max_iterations,
                max_tokens=config.max_tokens,
                custom_system_prompt=system_prompt,
                custom_tools=tools,
                logger=logger,
                verbose=False,
            )

            t0 = time.time()
            try:
                result = rlm.completion(
                    f"Video path: {video_path}\nQuestion: {ex['question']}\n\nBegin your analysis."
                )
                predicted = parse_count_from_response(result.response)
                elapsed = time.time() - t0

                usage_dict = None
                if result.usage_summary:
                    usage_dict = {
                        "total_input_tokens": result.usage_summary.total_input_tokens,
                        "total_output_tokens": result.usage_summary.total_output_tokens,
                        "total_cost": result.usage_summary.total_cost,
                    }

                # Save trajectory
                trajectory_path = None
                if logger and config.output.log_trajectories:
                    traj = logger.get_trajectory()
                    if traj:
                        traj_file = trajectory_dir / f"{example_id}.json"
                        with open(traj_file, "w") as tf:
                            json.dump(traj, tf, indent=2, default=str)
                        trajectory_path = str(traj_file)

                record = {
                    "example_id": example_id,
                    "video_id": ex["video_id"],
                    "question": ex["question"],
                    "label": ex["label"],
                    "expected": ex["count"],
                    "predicted": predicted,
                    "raw_response": result.response,
                    "error": None,
                    "execution_time": elapsed,
                    "usage": usage_dict,
                    "video_source": ex["video_source"],
                    "clip_duration": ex["clip_end"] - ex["clip_start"],
                    "category": ex.get("category", ""),
                    "trajectory_path": trajectory_path,
                    "config": config.name,
                }
                n_done += 1
                status = "OK" if abs(predicted - ex["count"]) <= 1 else "MISS"
                print(f"[{i + 1}/{total}] {status} expected={ex['count']:3d} predicted={predicted:3d} | {ex['label'][:50]}")

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
                    "category": ex.get("category", ""),
                    "config": config.name,
                }
                n_error += 1
                print(f"[{i + 1}/{total}] ERROR: {e}")
            finally:
                rlm.close()

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"\nDone. {n_done} evaluated, {n_error} errors. Results: {results_path}")
    return str(results_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def run_eval(config: EvalConfig, resume: bool = True) -> str:
    """Run evaluation based on config mode. Returns path to results JSONL."""
    if config.mode == "naive":
        return run_naive_eval(config, resume=resume)
    elif config.mode == "rvlm":
        return run_rvlm_eval(config, resume=resume)
    else:
        raise ValueError(f"Unknown mode: {config.mode}. Use 'naive' or 'rvlm'.")


def main():
    parser = argparse.ArgumentParser(description="RVLM Video Counting Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--max-examples", type=int, default=None, help="Override max examples")
    parser.add_argument("--dev-only", action="store_true", help="Run on 50-example dev set only")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (don't resume)")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without running")
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    if args.max_examples is not None:
        config.dataset.max_examples = args.max_examples
    if args.dev_only:
        config.dataset.dev_only = True

    # Load dataset size for cost estimation
    ds = load_dataset_split(config.dataset.split)
    if config.dataset.dev_only:
        with open(config.dataset.dev_indices_path) as f:
            n_examples = len(json.load(f))
    elif config.dataset.max_examples:
        n_examples = min(config.dataset.max_examples, len(ds))
    else:
        n_examples = len(ds)

    if args.dry_run:
        est = estimate_cost(config, n_examples)
        print(f"\n--- Cost Estimate: {config.name} ---")
        print(f"Mode:       {est['mode']}")
        print(f"Model:      {est['model']}")
        print(f"Examples:   {est['n_examples']}")
        print(f"Est. cost:  ${est['estimated_cost_usd']:.2f}")
        print()
        return

    # Confirm if running > 100 examples
    if n_examples > 100:
        est = estimate_cost(config, n_examples)
        print(f"\n--- Running {n_examples} examples, estimated cost: ${est['estimated_cost_usd']:.2f} ---")
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    print(f"\n{'=' * 60}")
    print(f"Config: {config.name}")
    print(f"Mode:   {config.mode}")
    print(f"Model:  {config.vlm_model if config.mode == 'naive' else config.backend_kwargs.get('model_name', '?')}")
    print(f"N:      {n_examples}")
    print(f"{'=' * 60}\n")

    results_path = run_eval(config, resume=not args.no_resume)

    # Auto-report
    from benchmarks.videocount.metrics import report

    report(results_path)


if __name__ == "__main__":
    main()
