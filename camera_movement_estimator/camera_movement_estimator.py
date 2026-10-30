"""Translation-only camera model, from sparse optical flow.

Cheap and adequate for a near-static camera. **It is not adequate for broadcast footage** --
see `HomographyCameraModel` and the residual table in FIXES.md item 10. It is kept because
it costs almost nothing and a fixed tactical camera really does move by a translation.

The original implementation of this model had three defects, all fixed here and all covered
by `tests/test_camera.py`:

1. It took the displacement of the single *largest-moving* feature as the camera motion --
   the most outlier-sensitive statistic available, so one feature landing on a running player
   became the camera's motion. Measured against a static scene with one moving object, it
   reported 22.5 px of camera motion where there was none.
2. It refreshed the reference *image* every frame but the reference *points* only when motion
   exceeded a threshold, so optical flow ran one frame's point coordinates against a
   different frame's image.
3. Because of (2) each value was drift since the last reset, not cumulative, yet the consumer
   subtracted it as though it were cumulative. On a 48 px synthetic pan it reported a final
   drift of zero.

Here: flow runs between consecutive frames on points detected from the matching image, failed
tracks are dropped using the status flag, the per-frame translation is the median of the
RANSAC inliers, and the result accumulates explicitly.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import CameraConfig

from .base import CameraModel, translation_homography

#: Below this many surviving points a frame's estimate is not trustworthy; we hold the
#: previous cumulative value and re-detect rather than invent a translation.
MIN_TRACKED_POINTS = 8


class CameraMovementEstimator(CameraModel):
    """Incremental estimator: feed it frames in order, get homographies out.

    Incremental rather than list-at-a-time so the pipeline can stream a clip of any length.
    """

    def __init__(self, config: CameraConfig | None = None, mask: np.ndarray | None = None):
        super().__init__()
        self.config = config or CameraConfig()
        self.mask = mask

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = dict(
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
        )

        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._frames_since_refresh = 0
        self._cumulative = np.zeros(2, dtype=float)

    # -- internals ---------------------------------------------------------------------

    def _detect(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(gray, mask=self.mask, **self.feature_params)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _translation(self, old_pts: np.ndarray, new_pts: np.ndarray) -> np.ndarray | None:
        """Robust global translation between two matched point sets.

        RANSAC gives the inlier set; the translation is the *median* displacement of those
        inliers. We deliberately do not read the translation out of the affine matrix: a
        partial-affine fit absorbs apparent rotation and scale from player motion into the
        translation term, and we only want the pan.
        """
        if len(old_pts) < MIN_TRACKED_POINTS:
            return None

        displacements = new_pts - old_pts
        inliers = None
        if len(old_pts) >= 3:
            _, inlier_mask = cv2.estimateAffinePartial2D(
                old_pts,
                new_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=2000,
                confidence=0.99,
            )
            if inlier_mask is not None:
                inliers = inlier_mask.ravel().astype(bool)

        chosen = displacements[inliers] if inliers is not None and inliers.sum() >= 3 else displacements
        return np.median(chosen, axis=0)

    def _should_refresh(self, n_tracked: int) -> bool:
        floor = self.config.refresh_below_fraction * self.config.max_corners
        return n_tracked < floor or self._frames_since_refresh >= self.config.refresh_every_frames

    # -- public API --------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._prev_gray = None
        self._prev_points = None
        self._frames_since_refresh = 0
        self._cumulative = np.zeros(2, dtype=float)

    @property
    def cumulative(self) -> np.ndarray:
        return self._cumulative.copy()

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        self.frames_seen += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect(gray)
            self._frames_since_refresh = 0
            return translation_homography(*self._cumulative)

        delta, tracked = None, None
        if self._prev_points is not None and len(self._prev_points) >= MIN_TRACKED_POINTS:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_points, None, **self.lk_params
            )
            if new_points is not None and status is not None:
                keep = status.ravel() == 1
                old_good = self._prev_points.reshape(-1, 2)[keep]
                new_good = new_points.reshape(-1, 2)[keep]
                delta = self._translation(old_good, new_good)
                tracked = new_good.reshape(-1, 1, 2).astype(np.float32)

        if delta is None:
            self.frames_without_estimate += 1
            delta, tracked = np.zeros(2), None
        elif np.linalg.norm(delta) < self.config.deadzone_px:
            # Sub-pixel estimation noise would otherwise random-walk into a phantom drift
            # over thousands of frames.
            delta = np.zeros(2)

        self._cumulative = self._cumulative + delta

        self._frames_since_refresh += 1
        if tracked is None or self._should_refresh(len(tracked)):
            self._prev_points = self._detect(gray)
            self._frames_since_refresh = 0
        else:
            # Carry the points forward at their *current* locations, matched to the image
            # they were measured in. This is the fix for defect (2).
            self._prev_points = tracked
        self._prev_gray = gray

        return translation_homography(*self._cumulative)

    def report(self) -> str:
        return (
            f"camera: translation model, {self.frames_seen} frames, "
            f"{self.frames_without_estimate} frame(s) without an estimate"
        )
