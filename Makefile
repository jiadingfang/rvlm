.PHONY: help install install-dev install-modal run-all \
        quickstart docker-repl lm-repl modal-repl \
        lint format test check \
        download-videos inspect-dataset \
        baseline-dev baseline-full analyze-results \
        eval eval-dev eval-dry-run \
        eval-naive-8f eval-naive-16f eval-naive-32f \
        eval-rvlm-d1 eval-rvlm-d2 eval-rvlm-minimal \
        eval-rvlm-gemini eval-rvlm-claude

help:
	@echo "RLM Examples Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install        - Install base dependencies with uv"
	@echo "  make install-dev    - Install dev dependencies with uv"
	@echo "  make install-modal  - Install modal dependencies with uv"
	@echo "  make run-all        - Run all examples (requires all deps and API keys)"
	@echo ""
	@echo "Examples:"
	@echo "  make quickstart     - Run quickstart.py (needs OPENAI_API_KEY)"
	@echo "  make docker-repl    - Run docker_repl_example.py (needs Docker)"
	@echo "  make lm-repl        - Run lm_in_repl.py (needs PORTKEY_API_KEY)"
	@echo "  make modal-repl     - Run modal_repl_example.py (needs Modal)"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Run ruff formatter"
	@echo "  make test           - Run tests"
	@echo "  make check          - Run lint + format + tests"

install:
	uv sync

install-dev:
	uv sync --group dev --group test

install-modal:
	uv pip install -e ".[modal]"

run-all: quickstart docker-repl lm-repl modal-repl

quickstart: install
	uv run python -m examples.quickstart

docker-repl: install
	uv run python -m examples.docker_repl_example

lm-repl: install
	uv run python -m examples.lm_in_repl

modal-repl: install-modal
	uv run python -m examples.modal_repl_example

lint: install-dev
	uv run ruff check .

format: install-dev
	uv run ruff format .

test: install-dev
	uv run pytest

check: lint format test

# ── VideoCount Benchmark (Phase 0) ──────────────────────────────────────
inspect-dataset:
	uv run python scripts/inspect_dataset.py

download-videos:
	uv run python -c "from benchmarks.videocount.dataset import download_all; download_all(num_workers=6, verbose=True)"

baseline-dev:
	uv run python -m benchmarks.videocount.baselines.naive_singleshot \
	  --model gpt-4o --n-frames 16 --dev-only \
	  --output benchmarks/videocount/results/naive_16f_gpt4o_dev.jsonl

baseline-full:
	uv run python -m benchmarks.videocount.baselines.naive_singleshot \
	  --model gpt-4o --n-frames 16 \
	  --output benchmarks/videocount/results/naive_16f_gpt4o.jsonl

analyze-results:
	uv run python -m benchmarks.videocount.analysis.analyze \
	  --results benchmarks/videocount/results/*.jsonl

# ── RVLM Evaluation Harness ───────────────────────────────────────────

# Generic eval target: pass CONFIG=path/to/config.yaml
eval:
	uv run python -m benchmarks.videocount.eval --config $(CONFIG)

# Dev set (50 examples) with any config
eval-dev:
	uv run python -m benchmarks.videocount.eval --config $(CONFIG) --dev-only

# Cost estimation
eval-dry-run:
	uv run python -m benchmarks.videocount.eval --config $(CONFIG) --dry-run

# ── Naive baselines ───────────────────────────────────────────────────

eval-naive-8f:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/naive_8f_gpt4o.yaml

eval-naive-16f:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/naive_16f_gpt4o.yaml

eval-naive-32f:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/naive_32f_gpt4o.yaml

# ── RVLM runs ────────────────────────────────────────────────────────

eval-rvlm-d1:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/rvlm_d1_v1_gpt4o.yaml

eval-rvlm-d2:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/rvlm_d2_v1_gpt4o.yaml

eval-rvlm-minimal:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/rvlm_d1_minimal_gpt4o.yaml

eval-rvlm-gemini:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/rvlm_d1_v1_gemini_flash.yaml

eval-rvlm-claude:
	uv run python -m benchmarks.videocount.eval \
	  --config benchmarks/videocount/configs/rvlm_d1_v1_claude.yaml
