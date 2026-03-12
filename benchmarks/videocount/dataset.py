"""
Dataset loader and video caching for Molmo2-VideoCountEval.

All 533 examples use YouTube video IDs. Videos are downloaded once and cached.
Only the relevant clip (clip_start..clip_end) is kept on disk to save space.
"""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

VIDEO_CACHE_DIR = Path(os.environ.get("RVLM_VIDEO_CACHE", "/tmp/rvlm_videos"))
HF_DATASET = "allenai/Molmo2-VideoCountEval"
HF_SPLIT = "val"


def load_dataset_split(split: str = HF_SPLIT):
    """Load Molmo2-VideoCountEval from HuggingFace."""
    from datasets import load_dataset

    return load_dataset(HF_DATASET)[split]


def _clip_cache_path(video_id: str, clip_start: float, clip_end: float, cache_dir: Path) -> Path:
    """Deterministic cache path for a specific clip."""
    key = f"{video_id}_{clip_start:.2f}_{clip_end:.2f}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return cache_dir / f"{video_id}_{h}.mp4"


UNAVAILABLE_ERRORS = (
    "Video unavailable",
    "This video has been removed",
    "This video is private",
    "Private video",
    "account associated with this video has been terminated",
    "has been removed by the uploader",
    "not available in your country",
    "copyright",
    "Sign in if you've been granted access",
)


def _is_unavailable_error(stderr: str) -> bool:
    """Return True if the yt-dlp error indicates the video is permanently unavailable."""
    lower = stderr.lower()
    return any(e.lower() in lower for e in UNAVAILABLE_ERRORS)


def _download_clip_yt(
    video_id: str,
    clip_start: float,
    clip_end: float,
    output_path: Path,
    max_height: int = 720,
) -> str:
    """
    Download a clip from YouTube using yt-dlp.
    Uses download sections to fetch only the needed portion.
    Returns: "ok" | "unavailable" | "failed"
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    section = f"*{clip_start:.2f}-{clip_end:.2f}"

    cmd = [
        "uv", "run", "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best[ext=mp4]/best",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if _is_unavailable_error(result.stderr):
            return "unavailable"
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            return "ok"
        # Try fallback: download full video and trim with ffmpeg
        status = _download_and_trim(video_id, clip_start, clip_end, output_path, max_height)
        return "ok" if status else "failed"
    except subprocess.TimeoutExpired:
        return "failed"
    except Exception:
        return "failed"


def _download_and_trim(
    video_id: str,
    clip_start: float,
    clip_end: float,
    output_path: Path,
    max_height: int = 720,
) -> bool:
    """Fallback: download full video, then trim with ffmpeg."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp = Path(f.name)

    try:
        # Download full video
        cmd_dl = [
            "uv", "run", "yt-dlp",
            "--quiet", "--no-warnings",
            "--format", f"best[height<={max_height}][ext=mp4]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(tmp),
            url,
        ]
        r = subprocess.run(cmd_dl, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not tmp.exists():
            return False

        # Trim with ffmpeg
        duration = clip_end - clip_start
        cmd_trim = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(clip_start),
            "-i", str(tmp),
            "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac",
            str(output_path),
        ]
        r2 = subprocess.run(cmd_trim, capture_output=True, text=True, timeout=120)
        return r2.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    finally:
        tmp.unlink(missing_ok=True)


def cache_video(
    example: dict,
    cache_dir: Path | str = VIDEO_CACHE_DIR,
    skip_existing: bool = True,
    verbose: bool = False,
) -> str | None:
    """
    Ensure the video clip for this example is available locally.
    Downloads from YouTube if not already cached.

    Returns the local file path, or None if download failed / video unavailable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_id = example["video_id"]
    clip_start = float(example["clip_start"])
    clip_end = float(example["clip_end"])

    out_path = _clip_cache_path(video_id, clip_start, clip_end, cache_dir)
    unavailable_marker = out_path.with_suffix(".unavailable")
    failed_marker = out_path.with_suffix(".failed")

    if skip_existing and out_path.exists() and out_path.stat().st_size > 1000:
        if verbose:
            print(f"  [cache hit] {out_path.name}")
        return str(out_path)

    if skip_existing and unavailable_marker.exists():
        if verbose:
            print(f"  [unavailable] {video_id}")
        return None

    if verbose:
        print(f"  [download] {video_id} {clip_start:.1f}-{clip_end:.1f}s → {out_path.name}")

    status = _download_clip_yt(video_id, clip_start, clip_end, out_path)

    if status == "ok":
        return str(out_path)
    elif status == "unavailable":
        unavailable_marker.touch()
        if verbose:
            print(f"  [unavailable] {video_id}")
        return None
    else:
        failed_marker.touch()
        if verbose:
            print(f"  [failed] {video_id}")
        return None


def download_all(
    split: str = HF_SPLIT,
    cache_dir: Path | str = VIDEO_CACHE_DIR,
    max_examples: int | None = None,
    num_workers: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Download all videos in the dataset split in parallel.
    Returns a summary dict with success/failure counts and a manifest.

    The manifest is saved to cache_dir/manifest.json mapping
    example index → local video path (or None if failed).
    """
    import concurrent.futures

    cache_dir = Path(cache_dir)
    ds = load_dataset_split(split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    manifest_path = cache_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    total = len(ds)
    results = {"success": 0, "failed": 0, "cached": 0, "unavailable": 0, "paths": {}}

    def _process(args):
        idx, ex = args
        key = str(idx)
        # Check if already in manifest and file exists
        if key in manifest and manifest[key] and Path(manifest[key]).exists():
            return idx, manifest[key], "cached"
        path = cache_video(ex, cache_dir=cache_dir, verbose=verbose)
        # Determine status: check marker files
        cache_dir_p = Path(cache_dir)
        vid = ex["video_id"]
        cs, ce = float(ex["clip_start"]), float(ex["clip_end"])
        out_path = _clip_cache_path(vid, cs, ce, cache_dir_p)
        if path:
            status = "success"
        elif out_path.with_suffix(".unavailable").exists():
            status = "unavailable"
        else:
            status = "failed"
        return idx, path, status

    items = list(enumerate(ds))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process, item): item for item in items}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            idx, path, status = future.result()
            results["paths"][idx] = path
            results[status] += 1
            manifest[str(idx)] = path
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total} | ok={results['success']} cached={results['cached']} unavailable={results['unavailable']} failed={results['failed']}")

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"\nManifest saved to {manifest_path}")
        print(f"Results: {results['success']} downloaded, {results['cached']} cached, {results['unavailable']} unavailable, {results['failed']} failed")

    return results


def load_manifest(cache_dir: Path | str = VIDEO_CACHE_DIR) -> dict:
    """Load the download manifest. Returns {} if not yet created."""
    p = Path(cache_dir) / "manifest.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)
