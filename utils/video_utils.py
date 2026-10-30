"""Streaming video I/O.

The original `read_video` decoded an entire clip into a list: 750 frames of 1920x1080 is
about 4.6 GB, and holding the annotated copy at the same time roughly doubled it. The
pipeline here makes two bounded-memory passes over the file instead -- one to detect and
measure, one to draw -- so peak memory is a handful of frames regardless of clip length.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

#: Used when a container reports a nonsensical frame rate (some do report 0).
FALLBACK_FPS = 25.0


@dataclass(frozen=True)
class VideoMeta:
    """Properties read from the file rather than assumed."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (
            f"{Path(self.path).name}: {self.width}x{self.height} @ {self.fps:.3f} fps, "
            f"{self.frame_count} frames ({self.duration_seconds:.1f}s)"
        )


def probe_video(video_path: str | Path) -> VideoMeta:
    """Read resolution, frame rate and length without decoding the whole clip."""
    path = str(video_path)
    if not Path(path).exists():
        raise FileNotFoundError(f"video not found: {path}")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise OSError(f"could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = FALLBACK_FPS
        return VideoMeta(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            # CAP_PROP_FRAME_COUNT is an estimate for some containers; treat it as a hint.
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(video_path: str | Path, limit: int | None = None) -> Iterator[np.ndarray]:
    """Yield frames one at a time. The caller must not retain them."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"could not open video: {video_path}")
        n = 0
        while limit is None or n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            n += 1
    finally:
        cap.release()


def iter_batches(
    video_path: str | Path, batch_size: int, limit: int | None = None
) -> Iterator[list[np.ndarray]]:
    """Yield lists of at most `batch_size` frames, for batched inference."""
    batch: list[np.ndarray] = []
    for frame in iter_frames(video_path, limit=limit):
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def fourcc_for(path: str | Path) -> int:
    """Pick a codec from the output extension.

    The original hardcoded XVID/.avi, which most browsers will not play.
    """
    suffix = Path(path).suffix.lower()
    codec = {".mp4": "mp4v", ".m4v": "mp4v", ".avi": "XVID", ".mkv": "mp4v"}.get(suffix)
    if codec is None:
        raise ValueError(f"unsupported output extension {suffix!r}; use .mp4 or .avi")
    return cv2.VideoWriter_fourcc(*codec)


@contextlib.contextmanager
def video_writer(output_path: str | Path, meta: VideoMeta):
    """Open a writer matching the source clip's resolution and frame rate.

    Both were previously hardcoded (24 fps, and overlay geometry assuming 1920x1080), which
    silently rescaled time on any other clip.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), fourcc_for(output_path), meta.fps, (meta.width, meta.height)
    )
    if not writer.isOpened():
        raise OSError(f"could not open video writer for {output_path}")
    try:
        yield writer
    finally:
        writer.release()


def read_video(video_path: str | Path) -> list[np.ndarray]:
    """Decode a whole clip into memory.

    Retained for notebooks and short diagnostic snippets only. The pipeline uses
    `iter_frames`; see the module docstring for why.
    """
    return list(iter_frames(video_path))


def save_video(frames: list[np.ndarray], output_path: str | Path, fps: float) -> None:
    """Write a list of frames. `fps` is required -- it used to be hardcoded to 24."""
    if not frames:
        raise ValueError("no frames to write")
    height, width = frames[0].shape[:2]
    meta = VideoMeta(str(output_path), width, height, fps, len(frames))
    with video_writer(output_path, meta) as writer:
        for frame in frames:
            writer.write(frame)
