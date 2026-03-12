# RVLM Implementation Plan

Extending Recursive Language Models (RLM) to Vision Language Models (RVLM) for video
object counting, benchmarked on `allenai/Molmo2-VideoCountEval`.

## Phases

| Phase | File | Status | Goal |
|---|---|---|---|
| 0 | [phase0_dataset_baseline.md](phase0_dataset_baseline.md) | IN PROGRESS | Understand dataset format; naive VLM baseline |
| 1 | [phase1_vlm_clients.md](phase1_vlm_clients.md) | TODO | VLM client wrappers (GPT-4o, Gemini, Claude) |
| 2 | [phase2_video_repl_tools.md](phase2_video_repl_tools.md) | TODO | Video frame sampling tools injected into REPL |
| 3 | [phase3_system_prompt.md](phase3_system_prompt.md) | TODO | Counting strategy system prompt |
| 4 | [phase4_eval_harness.md](phase4_eval_harness.md) | TODO | Reproducible evaluation pipeline |
| 5 | [phase5_ablations.md](phase5_ablations.md) | TODO | 10-run ablation study + failure analysis |
| 6 | [phase6_open_source_training.md](phase6_open_source_training.md) | TODO | SFT/RLVR on Molmo2-4B/8B |

## Suggested Order

```
Phase 0  →  Phase 1 + Phase 2 (parallel)  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6
```

Phases 1 and 2 are independent and can be developed in parallel.
Phase 3 depends on Phase 2 (needs the tool names for the prompt).
Phase 4 depends on Phases 1–3.
Phase 5 depends on Phase 4.
Phase 6 depends on Phase 5 (needs successful trajectories).

## Key Design Decisions

- **No new base classes**: VLM clients inherit `BaseLM`; video tools injected as `custom_tools`.
  The `RLM` class is used as-is — no fork of core framework code.
- **`rvlm/` namespace**: New code lives in `rvlm/` to stay separate from upstream `rlm/`.
- **Counting strategy lives in the prompt** (Phase 3), not in framework code — this makes
  it easy to iterate without changing infrastructure.
- **Gemini native video** preferred for long videos; GPT-4o frame sampling for short clips.
- **Trajectory distillation** (Phase 6) avoids expensive closed-model calls at inference time.

## Open Questions (resolve in Phase 0)

1. Exact MVC accuracy threshold (off-by-1? exact match?) — check arXiv 2601.10611 §Evaluation
2. Video format: bytes vs YouTube URLs → determines caching strategy
3. Dataset size: number of validation examples → determines eval cost
4. Train split availability → needed for Phase 6 trajectory collection
5. Ground truth format: scalar integer vs point-set annotations
