# Phase 2: Video REPL Tools

## Goal

Give the model Python functions it can call inside REPL code blocks to manipulate videos —
sampling frames, querying a VLM on a subset of frames, and inspecting video metadata. These
are injected as `custom_tools` at `RLM` instantiation time, requiring no changes to the
core framework.

## Design Principle

The model should be able to write natural Python to decompose a counting task:

```python
meta = get_video_metadata(video_path)
frames = sample_frames(video_path, n=16)
rough_count = vlm_count(frames, "How many sheep are visible?")

if meta["duration"] > 60:
    clips = split_video_into_clips(video_path, segment_sec=15)
    counts = [vlm_count(sample_clip(c, fps=2), "How many sheep?") for c in clips]
    count = max(counts)  # persistent objects: take max, not sum
else:
    count = rough_count

FINAL_VAR("count")
```

The tools must be stateless, serializable (for isolated env compat), and safe to call
from within the sandboxed `exec` namespace.

## File Structure

```
rvlm/
  utils/
    video_utils.py       # frame sampling, metadata, encoding
    counting_tools.py    # vlm_count and higher-level counting helpers
  environments/
    video_repl.py        # VideoREPL: LocalREPL pre-loaded with video tools
```

## Tasks

### 2.1 — `rvlm/utils/video_utils.py`

Low-level video I/O, no VLM calls here.

```python
import cv2
import numpy as np

def get_video_metadata(video_path: str) -> dict:
    """
    Return:
      {
        "duration": float,        # seconds
        "fps": float,
        "total_frames": int,
        "width": int,
        "height": int,
        "format": str,            # e.g. "mp4"
      }
    Raises FileNotFoundError if path does not exist.
    """

def sample_frames(
    video_path: str,
    n: int,
    strategy: str = "uniform",
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[np.ndarray]:
    """
    Sample n frames from [start_sec, end_sec].
    strategy options:
      "uniform"      — evenly spaced frame indices
      "scene_change" — select frames with highest inter-frame difference
                       (good for highlight-style videos)
    Returns list of HWC uint8 numpy arrays (RGB).
    Raises ValueError if n < 1 or n > total frames in range.
    """

def sample_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 1.0,
) -> list[np.ndarray]:
    """
    Extract all frames at the given fps from [start_sec, end_sec].
    Capped at 64 frames — raises ValueError if fps * duration > 64.
    """

def split_into_segments(
    video_path: str,
    segment_sec: float,
) -> list[tuple[float, float]]:
    """
    Return list of (start_sec, end_sec) tuples dividing the video
    into segments of approximately segment_sec duration.
    Last segment may be shorter.
    """

def resize_frames(
    frames: list[np.ndarray],
    max_side: int = 512,
) -> list[np.ndarray]:
    """
    Resize frames so the longer side is at most max_side, preserving aspect ratio.
    Used to reduce token cost before passing to VLM.
    """
```

Use `cv2.VideoCapture` as the primary backend. Add `decord.VideoReader` as a faster
fallback for long videos (import guarded with try/except).

### 2.2 — `rvlm/utils/counting_tools.py`

Higher-level tools that combine frame sampling + VLM calls. These depend on a VLM client
that is injected at construction time (closure over the client instance).

```python
def make_counting_tools(vlm_client) -> dict[str, dict]:
    """
    Build the custom_tools dict for RLM, bound to the given VLM client.

    Returns a dict suitable for passing to RLM(custom_tools=...).
    """
    def vlm_count(frames: list[np.ndarray], question: str) -> int:
        """
        Ask the VLM to count objects visible in the given frames.
        Prompt template: "Look at these {n} video frames. {question}
        Answer with a single integer only."
        Parses the response and returns int.
        Raises ValueError if the model does not return a parseable integer.
        """

    def vlm_describe(frames: list[np.ndarray], question: str) -> str:
        """
        Free-form VLM query over frames. Returns the raw response string.
        Use this for exploration before committing to a counting strategy.
        """

    def count_in_segments(
        video_path: str,
        question: str,
        segment_sec: float = 15.0,
        frames_per_segment: int = 4,
        aggregation: str = "max",
    ) -> int:
        """
        Split video into segments, count in each, aggregate.
        aggregation options:
          "max"  — for objects that persist across time (count peak density)
          "sum"  — for passage/throughput counting (objects flowing past camera)
        Returns aggregated count as int.
        """

    return {
        "get_video_metadata": {
            "tool": get_video_metadata,
            "description": "Returns duration, fps, total_frames, width, height for a video file.",
        },
        "sample_frames": {
            "tool": sample_frames,
            "description": (
                "Sample n frames from a video file. strategy='uniform' (default) or "
                "'scene_change'. Returns list of numpy HWC uint8 arrays (RGB)."
            ),
        },
        "sample_clip": {
            "tool": sample_clip,
            "description": "Extract frames at given fps from a time range [start_sec, end_sec].",
        },
        "split_into_segments": {
            "tool": split_into_segments,
            "description": "Divide video into (start_sec, end_sec) segments of ~segment_sec duration.",
        },
        "resize_frames": {
            "tool": resize_frames,
            "description": "Resize frames to reduce token cost (max_side=512 by default).",
        },
        "vlm_count": {
            "tool": vlm_count,
            "description": (
                "Ask the VLM to count objects in a list of frames. "
                "Returns an integer. Use vlm_describe first if unsure what you're looking for."
            ),
        },
        "vlm_describe": {
            "tool": vlm_describe,
            "description": "Free-form VLM query over frames. Returns raw string response.",
        },
        "count_in_segments": {
            "tool": count_in_segments,
            "description": (
                "Count objects across video segments. "
                "aggregation='max' for persistent objects, 'sum' for throughput counting."
            ),
        },
    }
```

**Important**: `vlm_count` inside `make_counting_tools` calls the client's `vlm_completion`
directly — this is a *tool call*, separate from the RLM's `llm_query` / `rlm_query` path.
The distinction is:
- `llm_query(prompt)` — text-only LM call routed through `LMHandler` (tracked in usage)
- `vlm_count(frames, q)` — direct VLM call for vision tasks (also tracked, but separately)

We need to decide whether `vlm_count` calls go through `LMHandler` or bypass it. For Phase 1,
bypass it (simpler). Track cost in a separate counter on the client. Revisit if we need
unified cost tracking.

### 2.3 — `rvlm/environments/video_repl.py`

A thin subclass of `LocalREPL` that pre-loads the video tools and the target video path.

```python
from rlm.environments.local_repl import LocalREPL
from rvlm.utils.counting_tools import make_counting_tools

class VideoREPL(LocalREPL):
    """
    LocalREPL pre-configured with video utility tools and an initial video_path variable.
    Usage:
        repl = VideoREPL(
            vlm_client=my_vlm_client,
            video_path="/tmp/video.mp4",
            lm_handler_address=handler.address,
        )
    The model can access:
        video_path    — str, the path to the video
        get_video_metadata(video_path)
        sample_frames(video_path, n)
        sample_clip(video_path, start, end, fps)
        split_into_segments(video_path, segment_sec)
        resize_frames(frames, max_side)
        vlm_count(frames, question)
        vlm_describe(frames, question)
        count_in_segments(video_path, question)
    """
    def __init__(self, vlm_client, video_path: str, **kwargs):
        tools = make_counting_tools(vlm_client)
        # Inject video_path as a plain value tool
        tools["video_path"] = {"tool": video_path, "description": "Path to the input video file."}
        super().__init__(custom_tools=tools, **kwargs)
```

### 2.4 — Tests

`tests/repl/test_video_tools.py`:
- `test_get_video_metadata`: create a 3-second synthetic mp4 with `cv2`, verify metadata dict
- `test_sample_frames_uniform`: verify n=8 returns 8 frames of the right shape
- `test_sample_frames_scene_change`: verify it returns n frames (exact frames can vary)
- `test_sample_clip_range`: verify only frames within [start, end] are returned
- `test_split_into_segments`: verify segment count and last segment boundary
- `test_resize_frames`: verify output dimensions respect max_side
- `test_vlm_count_parses_integer`: mock VLM client returning "5\n", verify returns 5
- `test_vlm_count_raises_on_non_integer`: mock returning "several", verify ValueError
- `test_count_in_segments_max_aggregation`: mock per-segment counts [3,7,5], verify returns 7
- `test_count_in_segments_sum_aggregation`: same mock, verify returns 15
- `test_video_repl_injects_tools`: instantiate VideoREPL, execute `get_video_metadata(video_path)`,
  verify returns dict without errors

All video tests use a synthetic video generated in a pytest fixture (no real video files in repo).

## Synthetic Video Fixture

```python
# tests/conftest.py
import cv2, numpy as np, tempfile, pytest

@pytest.fixture(scope="session")
def synthetic_video_path():
    """Create a 5-second 10fps 320x240 video with colored frames."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(50):
        frame = np.full((240, 320, 3), (i * 5 % 255, 100, 200), dtype=np.uint8)
        out.write(frame)
    out.release()
    yield path
```

## Deliverables

- [ ] `rvlm/utils/video_utils.py` with all 6 functions
- [ ] `rvlm/utils/counting_tools.py` with `make_counting_tools`
- [ ] `rvlm/environments/video_repl.py`
- [ ] Synthetic video pytest fixture in `tests/conftest.py`
- [ ] Tests passing: `uv run pytest tests/repl/test_video_tools.py`
- [ ] Manual demo: run a VideoREPL session on one real video from the dataset

## Cost Considerations

| Tool call | Approx token cost (GPT-4o, low detail) |
|---|---|
| `vlm_count` with 8 frames | ~8 × 85 = 680 image tokens + ~50 text |
| `vlm_count` with 16 frames | ~1360 image tokens |
| `count_in_segments` (4 segments × 4 frames) | ~4 × 340 = 1360 image tokens |

Keep `detail="low"` for all counting passes by default. Only switch to `detail="high"`
for a final verification pass on ≤ 4 frames.
