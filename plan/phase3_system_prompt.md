# Phase 3: RVLM Counting System Prompt

## Goal

Design and iterate on the system prompt that teaches the model the recursive video counting
strategy. This is the highest-leverage single artifact in the project — a well-designed
prompt can close most of the gap between naive single-shot and optimal recursive counting.

## Core Insight: Counting Is Not Captioning

The existing RLM system prompt (`rlm/utils/prompts.py::RLM_SYSTEM_PROMPT`) teaches
general decomposition. Video counting requires specific guidance on:

1. **When to decompose** — short static videos don't need recursion
2. **How to aggregate** — `max` vs `sum` vs `union` depending on object behavior
3. **Frame efficiency** — counting is token-expensive; avoid wasteful dense sampling
4. **Integer discipline** — always return a clean integer, never a range or approximation

## File Structure

```
rvlm/
  utils/
    counting_prompts.py   # RVLM_COUNTING_SYSTEM_PROMPT + builder functions
```

## Tasks

### 3.1 — Decision Tree the Model Should Follow

Before writing the prompt, specify the algorithm:

```
1. ASSESS
   - get_video_metadata(video_path) → duration, fps
   - sample_frames(video_path, n=8) → coarse_frames
   - vlm_describe(coarse_frames, "What objects are present? Are they moving or static?")
   → Decide: static_scene | moving_objects | throughput (objects passing camera)

2. ROUGH COUNT
   - vlm_count(coarse_frames, question) → rough_count
   - If rough_count == 0: verify with 16 more frames, then FINAL_VAR("count") = 0

3. DECIDE DECOMPOSITION
   If duration <= 30s AND rough_count <= 20:
       → Single pass with 16-32 frames (no recursion needed)
   If duration > 30s OR rough_count > 20:
       → Temporal decomposition into segments

4. TEMPORAL DECOMPOSITION
   segments = split_into_segments(video_path, segment_sec=15)
   For each segment:
       frames = sample_clip(segment_start, segment_end, fps=2)
       count_i = vlm_count(frames, question)
       OR: use rlm_query(f"Count in clip {i}") for truly complex sub-problems

5. AGGREGATE
   If static_scene:   count = max(count_i)    # peak density
   If throughput:     count = sum(count_i)    # objects flowing past
   If moving_objects: count = max(count_i)    # same objects throughout

6. VERIFY (optional, for high-stakes or uncertain cases)
   - Sample 4 frames near the peak-count segment
   - vlm_count(verify_frames, question, detail="high")

7. FINAL_VAR("count")
```

### 3.2 — `rvlm/utils/counting_prompts.py`

```python
RVLM_COUNTING_SYSTEM_PROMPT = """
You are a video object counting agent. You operate in a REPL environment with access to
video utility functions and a VLM (vision-language model) that can see video frames.

Your goal is to accurately count a specific type of object or event in a video.

## Available Tools
The following functions are available in your REPL:
- get_video_metadata(video_path) → dict with duration, fps, total_frames, width, height
- sample_frames(video_path, n, strategy='uniform') → list of numpy frames (RGB)
- sample_clip(video_path, start_sec, end_sec, fps=1.0) → list of frames
- split_into_segments(video_path, segment_sec) → list of (start_sec, end_sec) tuples
- resize_frames(frames, max_side=512) → resized frames (reduces cost)
- vlm_count(frames, question) → int (asks VLM to count, returns integer)
- vlm_describe(frames, question) → str (free-form VLM description)
- count_in_segments(video_path, question, segment_sec=15, frames_per_segment=4,
                    aggregation='max') → int

The video to analyze is available as the variable: video_path
The counting question is available as: question

## Counting Strategy

Follow this strategy. Adapt based on what you observe.

### Step 1: Assess
Always start by understanding the video before counting.
```python
meta = get_video_metadata(video_path)
coarse_frames = sample_frames(video_path, n=8)
description = vlm_describe(coarse_frames,
    "Briefly describe the scene and the objects relevant to: " + question +
    ". Are the objects static or moving? Do they persist throughout or flow past the camera?")
print(description)
```

### Step 2: Quick count
Get a rough count on the coarse frames.
```python
rough_count = vlm_count(coarse_frames, question)
print(f"Rough count: {rough_count}, Duration: {meta['duration']:.1f}s")
```

### Step 3: Decide whether to decompose
- If duration <= 30s AND rough_count is small (≤ 10): proceed directly to final count
  with 16–32 frames.
- If duration > 30s OR rough_count seems uncertain: decompose into temporal segments.

### Step 4: Temporal decomposition (if needed)
```python
# Example: segment every 15 seconds, sample 4 frames per segment
counts = []
for start, end in split_into_segments(video_path, segment_sec=15):
    frames = sample_clip(video_path, start, end, fps=2)
    frames = resize_frames(frames)
    c = vlm_count(frames, question)
    counts.append(c)
    print(f"[{start:.0f}s–{end:.0f}s]: {c}")
```

### Step 5: Aggregate
Choose the aggregation method based on object behavior:
- **max**: Objects that persist across time (animals in a pen, cars in a parking lot).
  The count is the peak number visible at one time.
- **sum**: Objects that flow past the camera one by one (people walking through a door,
  fish swimming past a sensor). Count how many pass in total.
When unsure, use **max** and state your reasoning.

```python
count = max(counts)   # or sum(counts)
print(f"Final count: {count}")
```

### Step 6: Return answer
Always end with a clean integer.
```python
FINAL_VAR("count")
```

## Rules
- NEVER return a range ("about 5-7") — always commit to a single integer.
- NEVER hallucinate. If frames are blurry or the object isn't visible, say count=0
  for that segment rather than guessing.
- Be efficient. Start with low detail (n=8) and only sample more frames if needed.
- If rlm_query is available and a segment is genuinely complex, you may delegate:
  result = rlm_query(f"Count {question} in video clip from {start}s to {end}s at {video_path}")
  But prefer code-based decomposition for simple cases.
"""


def build_counting_system_prompt(custom_tools: dict | None = None) -> list[dict]:
    """
    Return message history with the counting system prompt.
    Optionally appends descriptions of additional custom tools.
    """
    system = RVLM_COUNTING_SYSTEM_PROMPT
    if custom_tools:
        tool_descs = "\n".join(
            f"- {name}: {spec['description']}"
            for name, spec in custom_tools.items()
            if isinstance(spec, dict) and "description" in spec
        )
        system += f"\n\n## Additional Tools\n{tool_descs}"
    return [{"role": "user", "content": system}, {"role": "assistant", "content": "Understood."}]


def build_counting_user_prompt(video_path: str, question: str) -> dict:
    """
    First user turn for a counting task.
    """
    return {
        "role": "user",
        "content": (
            f"Video path: {video_path}\n"
            f"Question: {question}\n\n"
            "Begin your analysis."
        ),
    }
```

### 3.3 — Prompt Variants to Test

Create multiple prompt variants in `rvlm/utils/counting_prompts.py`:

| Variant | Description |
|---|---|
| `COUNTING_PROMPT_V1` | Full strategy above (baseline) |
| `COUNTING_PROMPT_CHAIN_OF_THOUGHT` | Ask model to verbalize reasoning before each code block |
| `COUNTING_PROMPT_MINIMAL` | Minimal instructions; model figures out strategy itself |
| `COUNTING_PROMPT_AGGRESSIVE_DECOMPOSE` | Always decompose into segments regardless of duration |

Evaluate all variants in Phase 5 ablations to determine which performs best.

### 3.4 — Integration with `RLM`

The counting prompt replaces the default `RLM_SYSTEM_PROMPT` via `custom_system_prompt`:

```python
from rvlm.utils.counting_prompts import build_counting_system_prompt

rvlm = RLM(
    backend="gpt4o",
    backend_kwargs={"model_name": "gpt-4o"},
    environment="local",
    custom_system_prompt=RVLM_COUNTING_SYSTEM_PROMPT,
    custom_tools=make_counting_tools(vlm_client),
    max_depth=2,
    max_iterations=10,
)
```

Note: `build_rlm_system_prompt` in `rlm/utils/prompts.py` wraps the system prompt in a
message list with query metadata. Verify it still works correctly when `custom_system_prompt`
is set (it does — the existing code uses it as-is if provided).

### 3.5 — Prompt Evaluation Protocol

Do NOT evaluate prompts on the full benchmark. Use a small held-out dev set (50 examples)
for rapid iteration:

1. Sample 50 examples from the dataset: 10 each from count ranges [0-2], [3-5], [6-10], [11-20], [21+]
2. Run each prompt variant on the 50-example dev set (GPT-4o, depth=1)
3. Report MVC accuracy, MAE, and qualitative failure analysis
4. Iterate on the prompt based on failure modes before running full benchmark

Common failure modes to watch for:
- **Phantom counting**: VLM hallucinates objects in dark/blurry frames → add "if frames are unclear, count=0"
- **Segment double-counting**: `sum` where `max` was needed → clarify aggregation rules in prompt
- **Refusal to commit**: Model returns "approximately 10" → add rule to always return integer
- **Over-decomposition**: Model samples 64 frames for a 5s video → add duration threshold checks

## Deliverables

- [ ] `rvlm/utils/counting_prompts.py` with `RVLM_COUNTING_SYSTEM_PROMPT`
- [ ] `build_counting_system_prompt()` and `build_counting_user_prompt()` functions
- [ ] At least 2 prompt variants (V1 + one alternative)
- [ ] Dev set of 50 examples selected and fixed (save indices to `benchmarks/videocount/dev_indices.json`)
- [ ] Prompt iteration complete: final chosen variant documented with rationale
