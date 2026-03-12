# Phase 5: Ablation Studies & Analysis

## Goal

Systematically measure what each component of RVLM contributes to counting accuracy,
identify failure modes, and produce a clear story for the final writeup.

## Run Matrix

Each row is a separate eval run using the harness from Phase 4.

| Run ID | Model | Depth | Strategy | Prompt | Notes |
|---|---|---|---|---|---|
| `naive-8f` | GPT-4o | — | 8 uniform frames, single call | none | Absolute baseline |
| `naive-16f` | GPT-4o | — | 16 uniform frames, single call | none | Stronger baseline |
| `naive-32f` | GPT-4o | — | 32 uniform frames, single call | none | Upper bound naive |
| `rvlm-d1-v1` | GPT-4o | 1 | RVLM loop, adaptive sampling | V1 | Main Phase 4 result |
| `rvlm-d2-v1` | GPT-4o | 2 | RVLM + recursive sub-calls | V1 | Recursion ablation |
| `rvlm-d1-minimal` | GPT-4o | 1 | RVLM loop, adaptive sampling | Minimal | Prompt ablation |
| `rvlm-d1-aggressive` | GPT-4o | 1 | Always decompose | Aggressive | Decomposition ablation |
| `gemini-native` | Gemini 2.0 Flash | 1 | Native video upload | V1 | Provider ablation |
| `claude-d1` | Claude 3.7 Sonnet | 1 | RVLM loop | V1 | Provider ablation |
| `rvlm-d1-v1-burst` | GPT-4o | 1 | Same as rvlm-d1-v1 | V1 | On BURST-VideoCount |

Run all on the full benchmark (Phase 0 determines size). Run `naive-*` first (cheapest)
to understand the baseline gap before investing in full RVLM runs.

## Ablation Questions

### A. How much does the RVLM loop add over single-shot?
Compare: `naive-32f` vs `rvlm-d1-v1`
- Controls for frame count (RVLM may use ~16-32 frames per segment anyway)
- Shows value of adaptive strategy vs fixed uniform sampling

### B. Does recursion help?
Compare: `rvlm-d1-v1` vs `rvlm-d2-v1`
- Expected gain: long videos with complex sub-regions
- Expected cost: higher latency and token cost
- Hypothesis: depth=2 helps mainly for count > 20 and duration > 60s

### C. How important is the prompt?
Compare: `rvlm-d1-v1` vs `rvlm-d1-minimal`
- Shows whether the structured counting strategy in the prompt is necessary
- If `minimal` is competitive → the model can self-discover the strategy
- If `minimal` is much worse → the strategy in V1 is load-bearing

### D. Decomposition: always vs adaptive?
Compare: `rvlm-d1-v1` vs `rvlm-d1-aggressive`
- `aggressive` always segments the video, even for short clips
- Expected: `aggressive` is worse for short videos (over-splits), better for long videos
- Informs whether the duration threshold in V1 is correctly tuned

### E. Which provider is best?
Compare: `rvlm-d1-v1` vs `gemini-native` vs `claude-d1`
- Gemini's native video upload may outperform frame sampling for long videos
- Claude's image limit (20/request) forces chunking → may miss context
- Report accuracy, latency, and cost per example

### F. Does the benchmark generalize?
Compare: `rvlm-d1-v1` (Molmo2-VideoCountEval) vs `rvlm-d1-v1-burst` (BURST-VideoCount)
- BURST is derived from annotation tracks → different style of ground truth
- Tests whether the RVLM approach generalizes or overfits to Molmo2 evaluation style

## Analysis Dimensions

For each run, compute metrics from `metrics.py::full_report`, then additionally:

### Failure Mode Taxonomy

Manually inspect ~50 wrong examples (|pred - gt| > 1) from `rvlm-d1-v1`.
Categorize each failure into:

| Category | Description |
|---|---|
| `phantom` | Model counts objects that aren't there |
| `miss` | Model misses real objects (under-count) |
| `wrong_aggregation` | Used sum where max was needed, or vice versa |
| `parse_error` | Model did not return a parseable integer |
| `bad_frame_selection` | Sampled frames missed the peak density moment |
| `tool_misuse` | Wrong tool call (e.g. used vlm_describe instead of vlm_count) |
| `hallucinated_question` | Model misunderstood what to count |

Produce a bar chart of failure categories.

### Performance vs Count Range

Plot MVC accuracy as a function of ground-truth count:
- x-axis: ground truth count (0, 1, 2, ..., 50+)
- y-axis: accuracy at threshold=1
- Overlay naive-16f vs rvlm-d1-v1
- Expected: RVLM gains most for high-count examples

### Performance vs Video Duration

Plot MVC accuracy as a function of video duration:
- x-axis: duration bucket [0-15s, 15-60s, 60-300s, 300s+]
- Overlay naive-16f vs rvlm-d1-v1 vs gemini-native

### Cost-Accuracy Tradeoff

Plot accuracy vs avg cost per example for all runs:
- x-axis: avg USD cost per example
- y-axis: MVC accuracy
- Pareto frontier plot
- Expected: RVLM trades more cost for more accuracy vs naive

### Token Budget Sensitivity

For `rvlm-d1-v1`, vary `max_tokens` (50k, 100k, 200k) on the dev set:
- Shows the marginal value of additional tokens
- Helps choose the right `max_tokens` for cost-accuracy tradeoff

## Comparison with Published Numbers

From the Molmo2 tech report (arXiv 2601.10611), collect:
- Molmo2-4B MVC accuracy on Molmo2-VideoCountEval
- Molmo2-8B MVC accuracy
- Qwen3-VL (35.5 vs 29.6 mentioned in search results)
- GPT-4o baseline (if reported)

Add these as horizontal reference lines in all plots. Document the exact metric
definition used in the paper and confirm it matches our `mvc_accuracy` implementation.

## Output Artifacts

```
benchmarks/videocount/
  results/
    naive_8f.jsonl
    naive_16f.jsonl
    naive_32f.jsonl
    rvlm_d1_v1.jsonl
    rvlm_d2_v1.jsonl
    rvlm_d1_minimal.jsonl
    gemini_native.jsonl
    claude_d1.jsonl
  analysis/
    figures/
      accuracy_vs_count.png
      accuracy_vs_duration.png
      cost_accuracy_tradeoff.png
      failure_taxonomy.png
    summary_table.csv       # all runs × all metrics
    failure_analysis.csv    # per-failure-mode counts
```

## Deliverables

- [ ] All configs created for the 10-run matrix
- [ ] All runs completed (start with dev set, then full benchmark)
- [ ] `summary_table.csv` with all metrics
- [ ] Failure mode taxonomy for 50 inspected examples
- [ ] 4 figures generated
- [ ] Comparison with Molmo2 published numbers documented
- [ ] Clear conclusion: which config is best, and why
