"""Pixel -> pitch-metre perspective transform.

Input points are expected in **frame-0 pixel space**, i.e. positions that have already had
cumulative camera drift subtracted (`CameraMovementEstimator.add_adjust_positions_to_tracks`).
The quad is fixed in frame-0 space, so feeding it raw per-frame positions -- or positions
corrected with a non-cumulative drift, as the original did -- mixes two coordinate systems
and the two drift apart exactly when the camera moves.

Points outside the quad cannot be transformed. The original returned `None` and said nothing,
which silently removed about a third of the sample clip's player-frames from every speed and
distance figure. Here the rate is counted and reported.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import PitchConfig

logger = logging.getLogger(__name__)


class ViewTransformer:
    def __init__(self, config: PitchConfig | None = None):
        self.config = config or PitchConfig()

        self.pixel_vertices = np.array(self.config.pixel_vertices, dtype=np.float32)
        self._validate(self.pixel_vertices)

        length, width = float(self.config.length_m), float(self.config.width_m)
        #: Order matches `pixel_vertices`: near-left, far-left, far-right, near-right.
        self.target_vertices = np.array(
            [[0, width], [0, 0], [length, 0], [length, width]], dtype=np.float32
        )

        self.transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
        self.inverse_transformer = cv2.getPerspectiveTransform(self.target_vertices, self.pixel_vertices)

        self.points_seen = 0
        self.points_outside = 0

        if not self.config.calibrated:
            logger.warning(
                "pitch quad is UNCALIBRATED (config.pitch.calibrated is false). The default "
                "vertices were eyeballed from one clip; speeds and distances are provisional. "
                "Recalibrate against known pitch lines for any other footage."
            )

    # -- validation --------------------------------------------------------------------

    @staticmethod
    def _validate(vertices: np.ndarray) -> None:
        """Reject a degenerate quad rather than build a silently garbage homography."""
        if vertices.shape != (4, 2):
            raise ValueError(f"pitch quad needs exactly 4 (x, y) vertices, got {vertices.shape}")
        if len({tuple(v) for v in vertices.tolist()}) != 4:
            raise ValueError("pitch quad has duplicate vertices")

        area = float(cv2.contourArea(vertices))
        if area <= 0:
            raise ValueError("pitch quad is degenerate (zero area)")
        if not cv2.isContourConvex(vertices):
            raise ValueError(
                "pitch quad is not convex -- vertices are probably out of order. Expected "
                "near-left, far-left, far-right, near-right."
            )

    # -- transform ---------------------------------------------------------------------

    def contains(self, point) -> bool:
        p = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0

    def transform_point(self, point) -> np.ndarray | None:
        """Pixel (x, y) -> pitch (x_m, y_m), or None if outside the calibrated quad."""
        self.points_seen += 1
        if not self.contains(point):
            self.points_outside += 1
            return None
        reshaped = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.transformer)
        return transformed.reshape(2)

    def inverse_transform_point(self, point) -> np.ndarray:
        """Pitch metres -> pixels. Used by the round-trip test."""
        reshaped = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(reshaped, self.inverse_transformer).reshape(2)

    def add_transformed_position_to_tracks(self, tracks: dict) -> None:
        for obj, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, info in track.items():
                    position = info.get("position_adjusted")
                    if position is None:
                        info["position_transformed"] = None
                        continue
                    transformed = self.transform_point(position)
                    info["position_transformed"] = None if transformed is None else transformed.tolist()

    # -- diagnostics -------------------------------------------------------------------

    @property
    def coverage(self) -> float:
        """Fraction of transformed points that fell inside the quad."""
        if not self.points_seen:
            return 0.0
        return 1.0 - self.points_outside / self.points_seen

    def coverage_report(self) -> str:
        return (
            f"pitch quad coverage: {self.coverage:.1%} "
            f"({self.points_seen - self.points_outside}/{self.points_seen} positions inside; "
            f"the rest have no speed or distance)"
        )
