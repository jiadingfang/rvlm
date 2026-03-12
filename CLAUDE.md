# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **rlms** — a Python library implementing Recursive Language Models (RLMs). An RLM replaces a standard `llm.completion(prompt)` call with `rlm.completion(prompt)`, offloading context to a REPL environment where the LM can execute code, decompose tasks, and recursively spawn child RLM calls.

## Commands

```bash
# Install (base deps)
uv sync

# Install dev + test deps
uv sync --group dev --group test

# Install pre-commit hooks
uv run pre-commit install

# Lint + format + test
make check

# Individually:
uv run ruff check --fix .
uv run ruff format .
uv run pytest

# Run a single test file
uv run pytest tests/test_local_repl.py

# Run a single test
uv run pytest tests/test_local_repl.py::test_name
```

## Code Style

- **Formatter**: `ruff` (line length 100, double quotes)
- **Do NOT use** `# type: ignore` without strong justification; use `cast(...)` or `assert` for type narrowing
- **No `_` prefix** on private methods unless explicitly requested
- **Error philosophy**: Fail fast, fail loud — no silent fallbacks or defensive programming
- **Naming**: `snake_case` methods/variables, `PascalCase` classes, `UPPER_CASE` constants

## Architecture

### Core Flow

`RLM.completion(prompt)` → spawns `LMHandler` (TCP server) + `BaseEnv` (REPL) → iterates: prompt LM → parse code blocks → execute in REPL → check for `FINAL_VAR` answer → repeat.

### Module Layout

- **`rlm/core/rlm.py`** — `RLM` class: the main entry point. Manages iteration loop, compaction, subcall recursion, and resource lifecycle.
- **`rlm/core/lm_handler.py`** — `LMHandler`: a `ThreadingTCPServer` wrapping LM clients; environments call into it via TCP socket to make LM completions.
- **`rlm/core/comms_utils.py`** — Length-prefixed JSON socket protocol (`4-byte big-endian + UTF-8 JSON`); `LMRequest`/`LMResponse` dataclasses.
- **`rlm/core/types.py`** — Shared types: `RLMChatCompletion`, `RLMIteration`, `CodeBlock`, `REPLResult`, `UsageSummary`, etc.
- **`rlm/clients/`** — LM client implementations, all inheriting from `BaseLM` (`base_lm.py`). Supported: `openai`, `anthropic`, `gemini`, `litellm`, `azure_openai`, `portkey`.
- **`rlm/environments/`** — REPL environment implementations, inheriting from `NonIsolatedEnv` or `IsolatedEnv` (`base_env.py`). Supported: `local`, `docker`, `modal`, `prime`, `daytona`, `e2b`.
- **`rlm/logger/`** — `RLMLogger` (trajectory capture) and `VerbosePrinter` (rich console output).
- **`rlm/utils/`** — Parsing (code block extraction, `FINAL_VAR` detection), prompts, token utilities, exceptions.

### Environment ↔ LM Handler Communication

**Non-isolated** (local, docker): Environment calls `llm_query()` / `rlm_query()` inside executed code → TCP socket → `LMHandler`.

**Isolated** (modal, prime, e2b, daytona): Environment runs in cloud sandbox; uses an HTTP broker (Flask server) inside the sandbox with `/enqueue`, `/pending`, `/respond` endpoints. Host polls the tunnel URL and forwards requests to `LMHandler`. State serialized via `dill` to `/tmp/rlm_state.dill`.

### Adding a New Client

Inherit from `BaseLM`, implement `completion`, `acompletion`, `get_usage_summary`, `get_last_usage`. Register in `rlm/clients/__init__.py`.

### Adding a New Environment

Inherit from `NonIsolatedEnv` or `IsolatedEnv`, implement `setup`, `load_context`, `execute_code`, `cleanup`. Provide `llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched`, `FINAL_VAR`, `SHOW_VARS` in globals. Register in `rlm/environments/__init__.py`. For isolated environments, see `modal_repl.py` as the canonical reference.

### Key Globals Injected into REPL

- `context`: The loaded context payload
- `llm_query(prompt, model=None)`: Plain LM call (no REPL)
- `rlm_query(prompt, model=None)`: Recursive child RLM call (falls back to `llm_query` at max depth)
- `FINAL_VAR(variable_name)`: Signal final answer
- `SHOW_VARS()`: List available variables

### Optional Features

- **`logger=RLMLogger(log_dir="./logs")`**: Captures trajectory as JSONL for the visualizer (`cd visualizer/ && npm run dev`).
- **`persistent=True`**: Reuse REPL environment across `completion()` calls (only supported for `local`). Use as context manager or call `rlm.close()`.
- **`compaction=True`**: Auto-summarize when context hits `compaction_threshold_pct` (default 85%) of the model's limit.
- **`custom_tools`**: Dict of callables injected into REPL globals.

## RVLM Extension (Recursive Vision Language Models)

The `rvlm/` package extends RLM to multimodal (video + text) tasks, targeting the
`allenai/Molmo2-VideoCountEval` video object counting benchmark.

### RVLM Module Layout

- **`rvlm/clients/`** — VLM client wrappers (`openai_vlm.py`, `gemini_vlm.py`, `anthropic_vlm.py`) inheriting `BaseLM`, plus `vision_utils.py` for frame encoding. Registry in `rvlm/clients/__init__.py`.
- **`rvlm/utils/video_utils.py`** — Frame sampling (`sample_frames`, `sample_clip`), metadata, segmentation via `cv2`.
- **`rvlm/utils/counting_tools.py`** — `make_counting_tools(vlm_client)` builds `custom_tools` dict for RLM: `vlm_count`, `vlm_describe`, `count_in_segments`.
- **`rvlm/utils/counting_prompts.py`** — System prompt variants (`v1`, `minimal`, `aggressive_decompose`) teaching the model a recursive counting strategy.

### Benchmark & Evaluation

- **`benchmarks/videocount/dataset.py`** — HuggingFace loader + yt-dlp video caching for Molmo2-VideoCountEval (533 examples, YouTube clips).
- **`benchmarks/videocount/metrics.py`** — MVC accuracy (exact match), MAE, RMSE, within-k, breakdowns by count range/source/duration.
- **`benchmarks/videocount/baselines/naive_singleshot.py`** — Single-shot VLM baseline (N uniform frames, one API call).
- **`benchmarks/videocount/eval.py`** — Main eval harness with YAML config loading, naive + RVLM modes, resume support, cost estimation.
- **`benchmarks/videocount/configs/`** — YAML configs for all run configurations.
- **`benchmarks/videocount/analysis/analyze.py`** — Results comparison CLI.

### Eval Commands

```bash
# Cost estimate
make eval-dry-run CONFIG=benchmarks/videocount/configs/rvlm_d1_v1_gpt4o.yaml

# Run on 50-example dev set
make eval-dev CONFIG=benchmarks/videocount/configs/naive_16f_gemini_flash.yaml

# Run full benchmark
make eval CONFIG=benchmarks/videocount/configs/naive_16f_gemini_flash.yaml

# Compare all results
make analyze-results
```
