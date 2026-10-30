"""Shared interface for camera motion models.

A camera model answers one question: **where in the reference view (frame 0) does this
pixel of frame n belong?** It answers with a 3x3 homography per frame, because that is what
a camera that pans, tilts and zooms about a fixed point actually does to the image.

The original pipeline answered with a 2D offset instead. That is a special case, and on real
broadcast footage a poor one: registering frame 0 of the sample clip against frame 200, a
homography leaves a 0.62 px residual where the best possible translation leaves 65.70 px.
Positions corrected by subtracting an offset are wrong by that much before they even reach
the perspective transform.

Both models here return homographies so they are interchangeable; `TranslationCameraModel`
simply returns ones whose rotation and perspective terms are identity.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

IDENTITY = np.eye(3, dtype=float)


def translation_homography(dx: float, dy: float) -> np.ndarray:
    """The homography that maps a frame shifted by (dx, dy) back onto the reference."""
    return np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]])


def apply_homography(homography: np.ndarray, point) -> tuple[float, float]:
    q = homography @ np.array([point[0], point[1], 1.0], dtype=float)
    if q[2] == 0:
        return float(point[0]), float(point[1])
    return float(q[0] / q[2]), float(q[1] / q[2])


class CameraModel:
    """Base class. Subclasses implement `update`."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.frames_without_estimate = 0

    # -- subclass hooks ----------------------------------------------------------------

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Consume the next frame; return its (3, 3) homography to the reference view."""
        raise NotImplementedError

    def reset(self) -> None:
        self.frames_seen = 0
        self.frames_without_estimate = 0

    # -- convenience -------------------------------------------------------------------

    def estimate_homographies(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        self.reset()
        return np.array([self.update(frame) for frame in frames], dtype=float)

    def stream(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        self.reset()
        for frame in frames:
            yield self.update(frame)

    def estimate_sequence(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """(N, 2) cumulative translation, for the simple case and for diagnostics."""
        return self.translations(self.estimate_homographies(frames))

    # -- shared statics ----------------------------------------------------------------

    @staticmethod
    def translations(homographies) -> np.ndarray:
        """The translation component of each homography, as (N, 2).

        Exact for a translation model. For a full homography it is the motion of the image
        origin, which is a summary rather than the whole story -- use the homography itself
        for anything that matters, and this only for display.
        """
        matrices = np.asarray(homographies, dtype=float)
        if matrices.ndim == 2:
            matrices = matrices[None]
        scale = matrices[:, 2, 2]
        scale = np.where(scale == 0, 1.0, scale)
        return -matrices[:, :2, 2] / scale[:, None]

    @staticmethod
    def pan_at(homography: np.ndarray, width: int, height: int) -> tuple[float, float]:
        """How far the view has moved, measured at the frame centre."""
        cx, cy = width / 2.0, height / 2.0
        mx, my = apply_homography(homography, (cx, cy))
        return cx - mx, cy - my

    def draw(self, frame: np.ndarray, homography, scale: float = 1.0, box=None) -> np.ndarray:
        """Overlay how far the view has panned from the reference, measured at the centre."""
        from utils.draw_utils import draw_text_panel

        height, width = frame.shape[:2]
        dx, dy = self.pan_at(np.asarray(homography, dtype=float), width, height)
        return draw_text_panel(
            frame,
            [f"Camera pan X: {dx:.1f} px", f"Camera pan Y: {dy:.1f} px"],
            box=box or (0.0, 0.0, 0.26, 0.093),
            scale=scale,
        )

    @staticmethod
    def add_adjust_positions_to_tracks(tracks: dict, homographies) -> None:
        """Map every position into the reference view's coordinates.

        Takes homographies, not offsets. `position_adjusted` is then in frame-0 pixel
        space, which is the space `ViewTransformer`'s quad is defined in.
        """
        matrices = np.asarray(homographies, dtype=float)
        if matrices.ndim == 2:
            matrices = matrices[None]

        for object_tracks in tracks.values():
            for frame_num, track in enumerate(object_tracks):
                if frame_num >= len(matrices):
                    continue
                homography = matrices[frame_num]
                for info in track.values():
                    if "position" not in info:
                        continue
                    info["position_adjusted"] = apply_homography(homography, info["position"])
