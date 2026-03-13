import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_video_path():
    """Create a 5-second 10fps 320x240 video with colored frames."""
    fps = 10
    duration = 5
    width, height = 320, 240
    total_frames = fps * duration

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]

    for i in range(total_frames):
        color = colors[i % len(colors)]
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)

    writer.release()

    yield path

    import os

    os.unlink(path)
