"""
Higher-level counting tools that combine frame sampling + VLM calls.
These are injected into the REPL as custom_tools for RLM.
"""

import re

from rvlm.utils.video_utils import (
    get_video_metadata,
    resize_frames,
    sample_clip,
    sample_frames,
    split_into_segments,
)


def make_counting_tools(vlm_client) -> dict[str, dict]:
    """
    Build the custom_tools dict for RLM, bound to the given VLM client.
    Returns a dict suitable for passing to RLM(custom_tools=...).
    """

    def vlm_count(frames: list, question: str) -> int:
        """Ask the VLM to count objects visible in the given frames. Returns int."""
        prompt = (
            f"Look at these {len(frames)} video frames. {question} "
            "Answer with a single integer only. Do not explain."
        )
        response = vlm_client.vlm_completion(frames=frames, prompt=prompt)
        matches = re.findall(r"\b\d+\b", response)
        if not matches:
            raise ValueError(f"VLM did not return a parseable integer: {response!r}")
        return int(matches[-1])

    def vlm_describe(frames: list, question: str) -> str:
        """Free-form VLM query over frames. Returns the raw response string."""
        prompt = f"Look at these {len(frames)} video frames. {question}"
        return vlm_client.vlm_completion(frames=frames, prompt=prompt)

    def count_in_segments(
        video_path: str,
        question: str,
        segment_sec: float = 15.0,
        frames_per_segment: int = 4,
        aggregation: str = "max",
    ) -> int:
        """
        Split video into segments, count in each, aggregate.
        aggregation: "max" for persistent objects, "sum" for throughput counting.
        """
        segments = split_into_segments(video_path, segment_sec)
        counts = []
        for start, end in segments:
            frames = sample_clip(video_path, start, end, fps=frames_per_segment / max(end - start, 0.1))
            frames = resize_frames(frames)
            c = vlm_count(frames, question)
            counts.append(c)

        if aggregation == "max":
            return max(counts) if counts else 0
        elif aggregation == "sum":
            return sum(counts)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Use 'max' or 'sum'.")

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
