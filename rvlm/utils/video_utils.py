"""Low-level video I/O utilities. No VLM calls here."""

import cv2
import numpy as np


def get_video_metadata(video_path: str) -> dict:
    """Return duration, fps, total_frames, width, height, format for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    meta = {
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "format": video_path.rsplit(".", 1)[-1] if "." in video_path else "unknown",
    }
    cap.release()
    return meta


def sample_frames(
    video_path: str,
    n: int,
    strategy: str = "uniform",
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[np.ndarray]:
    """
    Sample n frames from [start_sec, end_sec].
    strategy: "uniform" (evenly spaced) or "scene_change" (highest inter-frame diff).
    Returns list of HWC uint8 numpy arrays (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else total_frames
    end_frame = min(end_frame, total_frames)

    if n < 1:
        raise ValueError("n must be >= 1")

    available = end_frame - start_frame
    n = min(n, max(available, 1))

    if strategy == "uniform":
        indices = np.linspace(start_frame, end_frame - 1, n, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    elif strategy == "scene_change":
        # Read candidate frames, pick n with highest inter-frame difference
        step = max(1, available // min(available, n * 4))
        candidate_indices = list(range(start_frame, end_frame, step))

        candidates = []
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidates.append((idx, frame, gray))
        cap.release()

        if len(candidates) <= n:
            return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f, _ in candidates]

        # Compute inter-frame differences
        diffs = [0.0]  # first frame always included
        for i in range(1, len(candidates)):
            diff = np.mean(np.abs(candidates[i][2].astype(float) - candidates[i - 1][2].astype(float)))
            diffs.append(diff)

        # Pick top-n by difference, keeping order
        top_indices = sorted(sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:n])
        return [cv2.cvtColor(candidates[i][1], cv2.COLOR_BGR2RGB) for i in top_indices]

    else:
        cap.release()
        raise ValueError(f"Unknown strategy: {strategy}")


def sample_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 1.0,
) -> list[np.ndarray]:
    """Extract frames at given fps from [start_sec, end_sec]. Capped at 64 frames."""
    duration = end_sec - start_sec
    n_frames = int(duration * fps)
    if n_frames > 64:
        raise ValueError(f"fps * duration = {n_frames} > 64. Reduce fps or narrow the range.")
    n_frames = max(1, n_frames)
    return sample_frames(video_path, n=n_frames, strategy="uniform", start_sec=start_sec, end_sec=end_sec)


def split_into_segments(video_path: str, segment_sec: float) -> list[tuple[float, float]]:
    """Divide video into (start_sec, end_sec) segments of ~segment_sec duration."""
    meta = get_video_metadata(video_path)
    duration = meta["duration"]
    segments = []
    start = 0.0
    while start < duration:
        end = min(start + segment_sec, duration)
        segments.append((start, end))
        start = end
    return segments


def resize_frames(frames: list[np.ndarray], max_side: int = 512) -> list[np.ndarray]:
    """Resize frames so the longer side is at most max_side, preserving aspect ratio."""
    from rvlm.clients.vision_utils import resize_frame

    return [resize_frame(f, max_side) for f in frames]
