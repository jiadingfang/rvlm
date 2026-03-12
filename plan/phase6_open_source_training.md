# Phase 6: Open-Source Model Training

## Goal

Train an open-source VLM to replicate the RVLM counting behavior without requiring
closed-model API calls, using successful trajectories from Phase 4/5 as training data.

## Prerequisite

Phase 5 must be complete. We need:
- ≥ 1000 successful RVLM trajectories (|pred - gt| ≤ 1) with full `RLMLogger` JSONL
- A clear best config (prompt variant, depth, aggregation strategy) from ablations
- The BURST-VideoCount results to confirm generalization

## Approach: Trajectory Distillation (SFT → RLVR)

Two stages:
1. **SFT**: Supervised fine-tuning on successful closed-model trajectories
2. **RLVR** (if SFT plateaus): Reinforcement learning with a verifiable reward
   (reward = 1 if |pred - gt| ≤ 1, else 0)

## Target Models

| Model | Params | Rationale |
|---|---|---|
| **Molmo2-4B** | 4B | Same family as the benchmark; strong video prior; smaller for iteration |
| **Molmo2-8B** | 8B | Upper bound for open-source; matches Molmo2's own baseline |
| **Qwen2.5-VL-7B** | 7B | Strong independent baseline; different architecture |

Start with Molmo2-4B for iteration speed. Scale to 8B once SFT pipeline is validated.

## File Structure

```
training/
  __init__.py
  trajectory_converter.py    # JSONL trajectory → SFT format
  dataset_builder.py         # build and validate training dataset
  sft_train.py               # SFT training script (trl SFTTrainer)
  rlvr_train.py              # RLVR training script (trl GRPOTrainer or TRL PPO)
  reward_fn.py               # MVC reward function
  configs/
    sft_molmo2_4b.yaml
    sft_qwen25vl_7b.yaml
    rlvr_molmo2_4b.yaml
  eval_open.py               # evaluate fine-tuned model on benchmark
```

## Tasks

### 6.1 — Trajectory Conversion (`training/trajectory_converter.py`)

The `RLMLogger` saves trajectories as JSONL with this structure (from `rlm/logger/rlm_logger.py`):
```json
{
  "metadata": {...},
  "iterations": [
    {
      "prompt": [...],         // message list
      "response": "...",       // model response text
      "code_blocks": [
        {"code": "...", "result": {"stdout": "...", "stderr": "..."}}
      ],
      "final_answer": "7"     // present on last iteration if successful
    }
  ]
}
```

Convert to a single multi-turn conversation in the model's chat format:

```python
def trajectory_to_sft_example(
    trajectory: dict,
    video_path: str,
    question: str,
    expected_count: int,
) -> dict | None:
    """
    Convert an RLMLogger trajectory to an SFT training example.

    Returns None (skip) if:
    - Trajectory did not reach a final answer
    - |parsed_final_answer - expected_count| > 1  (incorrect answer)
    - Trajectory has > max_turns iterations (too long for context)

    Returns a dict with:
    - "messages": list of dicts in OpenAI chat format (role/content)
                  content may include image content blocks for VLM models
    - "video_path": str
    - "question": str
    - "expected": int
    - "predicted": int
    """
```

The resulting message list interleaves:
```
[system]   counting system prompt
[user]     "Video: {path}\nQuestion: {question}"
[assistant] (reasoning + code block)
[user]     "Code output:\n{stdout}"
[assistant] (more reasoning + code block)
[user]     "Code output:\n{stdout}"
...
[assistant] FINAL_VAR("count")
[user]     "count = 7"   ← REPL result
[assistant] "7"          ← final answer
```

### 6.2 — Dataset Builder (`training/dataset_builder.py`)

```python
def build_sft_dataset(
    trajectory_dir: str,
    dataset_metadata: str,         # JSONL with {example_id, video_path, question, expected}
    output_path: str,
    min_examples: int = 500,
    max_turns: int = 8,
    balance_by_count_range: bool = True,
) -> None:
    """
    1. Load all trajectory JSONLs from trajectory_dir
    2. Match to metadata (video_path, question, expected)
    3. Filter: keep only successful, ≤ max_turns examples
    4. Optionally balance across count ranges [0-5, 6-20, 21+]
    5. Write to output_path as HuggingFace Dataset (arrow format)
    6. Print statistics: n_examples, n_filtered, distribution by count range
    """
```

Target dataset size: 2000-5000 examples (may need to run Phase 4 eval on more examples
from the training split of the dataset, if a training split exists).

### 6.3 — SFT Training (`training/sft_train.py`)

Use `trl.SFTTrainer` with HuggingFace `transformers`.

Key considerations for vision-language SFT:
- **Frame encoding**: Videos must be pre-processed into frame tensors and passed with the
  correct processor format for each model (Molmo2 uses its own `MolmoProcessor`)
- **Context length**: Trajectories with many frames can be long. Cap at 8192 tokens.
  Filter examples exceeding this during dataset building.
- **LoRA**: Use LoRA (rank=16, alpha=32) for efficient fine-tuning on single GPU.
  Full fine-tune only if multiple GPUs available.
- **Gradient checkpointing**: Required for 4B+ models

```yaml
# configs/sft_molmo2_4b.yaml
model:
  name: allenai/Molmo2-4B
  load_in_4bit: true       # QLoRA
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "v_proj"]

training:
  num_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  warmup_ratio: 0.1
  max_seq_length: 8192
  output_dir: checkpoints/sft_molmo2_4b

data:
  train_path: training/data/sft_train.arrow
  eval_path: training/data/sft_eval.arrow
```

### 6.4 — RLVR Training (`training/rlvr_train.py`)

If SFT plateaus (MVC accuracy stops improving), switch to reinforcement learning.

**Reward function** (`training/reward_fn.py`):
```python
def mvc_reward(predicted_response: str, expected_count: int, threshold: int = 1) -> float:
    """
    Binary reward: 1.0 if |parse_count(predicted_response) - expected_count| <= threshold,
    else 0.0. Returns 0.0 on parse error.
    """
```

Use `trl.GRPOTrainer` (Group Relative Policy Optimization) — simpler than PPO, works
well for verifiable rewards (same setup as DeepSeek-R1-Zero). Alternatively `trl.PPOTrainer`.

GRPO config:
- Generate K=8 rollouts per example, score with `mvc_reward`, update policy
- KL penalty from reference model (base Molmo2-4B) to prevent catastrophic forgetting
- Start from the SFT checkpoint, not from scratch

### 6.5 — Open-Source Evaluation (`training/eval_open.py`)

Evaluate the fine-tuned model directly (not via RVLM framework — the model has
internalized the strategy and runs it as a single multi-turn completion):

```python
def eval_finetuned_model(
    model_path: str,           # local checkpoint or HF hub
    dataset_split: str = "validation",
    max_examples: int | None = None,
    output_path: str = "results/finetuned.jsonl",
) -> None:
    """
    Run inference on the benchmark using the fine-tuned model.
    The model is prompted with system prompt + video + question.
    It generates a multi-turn response (code + REPL simulation) and gives a final count.
    """
```

Note: At inference time, the fine-tuned open model must still execute code. Options:
1. **Keep RVLM framework**: Run fine-tuned model through `RLM` with `VideoREPL`
   (cleanest; requires loading via `transformers` not OpenAI API)
2. **Self-contained inference**: The model outputs a chain-of-thought with code,
   we extract and execute code blocks, and feed back REPL output (replicate the loop
   manually without the RLM framework)

Option 1 is strongly preferred for consistency. Requires a `LocalVLMClient` wrapper
that runs inference via `transformers` instead of an API.

### 6.6 — `rvlm/clients/local_vlm.py`

```python
class LocalVLMClient(BaseLM):
    """
    VLM client backed by a locally-loaded HuggingFace model.
    Loads model with transformers and runs inference on GPU.
    """
    def __init__(
        self,
        model_name_or_path: str,   # e.g. "allenai/Molmo2-4B" or "./checkpoints/sft_molmo2_4b"
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 2048,
    ): ...

    def vlm_completion(self, frames: list[np.ndarray], prompt: str) -> str:
        """Run inference using the model's processor + generate."""
```

Add `local` to the VLM client registry so eval configs can just set `backend: local`.

## Deliverables

- [ ] `training/trajectory_converter.py` converting RLMLogger JSONL → SFT format
- [ ] `training/dataset_builder.py` with filtering and balancing
- [ ] SFT training dataset built (min 2000 successful trajectories)
- [ ] `training/sft_train.py` runnable on single A100/H100
- [ ] SFT training completed for Molmo2-4B
- [ ] `rvlm/clients/local_vlm.py` for local inference
- [ ] Full benchmark eval of fine-tuned model via `training/eval_open.py`
- [ ] Comparison table: naive Molmo2-4B vs SFT-Molmo2-4B vs closed-model RVLM
- [ ] (Stretch) RLVR training if SFT MVC accuracy < closed-model - 5%

## Compute Requirements

| Step | Hardware | Est. Time |
|---|---|---|
| SFT Molmo2-4B (QLoRA) | 1× A100 80GB | ~4-8 hours |
| SFT Molmo2-8B (QLoRA) | 2× A100 80GB | ~8-16 hours |
| RLVR Molmo2-4B | 2× A100 80GB | ~24-48 hours |
| Benchmark eval (fine-tuned) | 1× A100 80GB | ~4-6 hours |
