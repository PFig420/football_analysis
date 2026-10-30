"""Per-player speed and distance, in real units.

Changes from the original, all of which affected the numbers:

* The measurement window is defined in **seconds**, not a fixed 5 frames, and the frame rate
  comes from the file instead of a hardcoded 24 -- so a 50 fps clip no longer reports speeds
  roughly double the truth.
* Trajectories are smoothed before differencing. Summing raw per-window chords turns
  positional jitter into distance, and only ever upward.
* Samples above a plausibility ceiling are dropped and counted rather than displayed.
* Each track carries its own coverage: how many frames it was actually measurable for.
  Distances accumulate per *track ID*, and tracking identity switches (39 IDs for ~22 players
  on the sample clip) split one player across several tracks. That is a re-identification
  problem this module cannot solve, so instead of presenting a fragment as a player's total,
  it reports the span the figure covers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import SpeedConfig
from utils.draw_utils import draw_label

logger = logging.getLogger(__name__)

#: Object classes that are people and therefore have a meaningful running speed.
MEASURED_OBJECTS = ("players", "goalkeepers")

SECONDS_PER_HOUR = 3600.0
METRES_PER_KM = 1000.0


@dataclass
class TrackSummary:
    track_id: int
    object_type: str
    first_frame: int
    last_frame: int
    measured_frames: int
    distance_m: float = 0.0
    dropped_samples: int = 0
    top_speed_kmh: float = 0.0

    @property
    def span_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def coverage(self) -> float:
        return self.measured_frames / self.span_frames if self.span_frames else 0.0

    def as_row(self, fps: float) -> dict:
        seconds = self.span_frames / fps if fps else 0.0
        return {
            "track_id": int(self.track_id),
            "object": self.object_type,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "span_seconds": round(seconds, 2),
            "coverage": round(self.coverage, 3),
            "distance_m": round(self.distance_m, 1),
            "top_speed_kmh": round(self.top_speed_kmh, 1),
            "dropped_samples": self.dropped_samples,
        }


class SpeedAndDistanceEstimator:
    def __init__(self, fps: float, config: SpeedConfig | None = None):
        if not fps or fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self.fps = float(fps)
        self.config = config or SpeedConfig()
        self.summaries: dict[tuple[str, int], TrackSummary] = {}
        self.dropped_total = 0

    # -- helpers -----------------------------------------------------------------------

    @property
    def window_frames(self) -> int:
        return max(1, int(round(self.config.window_seconds * self.fps)))

    def _smooth(self, positions: np.ndarray) -> np.ndarray:
        """Savitzky-Golay smoothing over a contiguous run of valid positions."""
        window = int(round(self.config.smoothing_seconds * self.fps))
        if window % 2 == 0:
            window += 1
        if window < 5 or len(positions) < window:
            return positions
        try:
            from scipy.signal import savgol_filter

            return savgol_filter(positions, window_length=window, polyorder=2, axis=0)
        except Exception:  # scipy absent or degenerate input -- smoothing is optional
            return positions

    @staticmethod
    def _contiguous_runs(valid: np.ndarray) -> list[tuple[int, int]]:
        """Index ranges [start, end) where `valid` is True."""
        runs, start = [], None
        for i, ok in enumerate(valid):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(valid)))
        return runs

    def _collect(self, object_tracks: list[dict]) -> dict[int, list[tuple[int, list | None]]]:
        by_track: dict[int, list[tuple[int, list | None]]] = {}
        for frame_num, frame_tracks in enumerate(object_tracks):
            for track_id, info in frame_tracks.items():
                by_track.setdefault(track_id, []).append((frame_num, info.get("position_transformed")))
        return by_track

    # -- main --------------------------------------------------------------------------

    def add_speed_and_distance_to_tracks(self, tracks: dict) -> None:
        for object_type in MEASURED_OBJECTS:
            object_tracks = tracks.get(object_type)
            if not object_tracks:
                continue
            for track_id, observations in self._collect(object_tracks).items():
                self._process_track(tracks, object_type, track_id, observations)

    def _process_track(self, tracks, object_type, track_id, observations) -> None:
        frames = np.array([f for f, _ in observations])
        valid = np.array([p is not None for _, p in observations])
        summary = TrackSummary(
            track_id=track_id,
            object_type=object_type,
            first_frame=int(frames[0]),
            last_frame=int(frames[-1]),
            measured_frames=int(valid.sum()),
        )
        self.summaries[(object_type, track_id)] = summary
        if valid.sum() < 2:
            return

        window = self.window_frames
        max_speed = self.config.max_plausible_kmh

        for start, end in self._contiguous_runs(valid):
            if end - start < 2:
                continue
            run_frames = frames[start:end]
            run_positions = np.array([observations[i][1] for i in range(start, end)], dtype=float)
            smoothed = self._smooth(run_positions)

            # Non-overlapping windows, so distance is a sum of independent displacements
            # rather than an overlapping sum that would multiply-count the same movement.
            for i in range(0, len(smoothed) - 1, window):
                j = min(i + window, len(smoothed) - 1)
                frame_gap = int(run_frames[j] - run_frames[i])
                if frame_gap <= 0:
                    continue

                displacement = float(np.linalg.norm(smoothed[j] - smoothed[i]))
                elapsed = frame_gap / self.fps
                speed_kmh = (displacement / elapsed) * SECONDS_PER_HOUR / METRES_PER_KM

                if speed_kmh > max_speed:
                    summary.dropped_samples += 1
                    self.dropped_total += 1
                    continue
                if displacement < self.config.noise_floor_m:
                    displacement = 0.0

                summary.distance_m += displacement
                summary.top_speed_kmh = max(summary.top_speed_kmh, speed_kmh)

                for k in range(i, j):
                    frame_num = int(run_frames[k])
                    info = tracks[object_type][frame_num].get(track_id)
                    if info is not None:
                        info["speed"] = speed_kmh
                        info["distance"] = summary.distance_m

    # -- output ------------------------------------------------------------------------

    def summary_rows(self) -> list[dict]:
        return [s.as_row(self.fps) for s in sorted(
            self.summaries.values(), key=lambda s: (-s.distance_m, s.track_id)
        )]

    def report(self) -> str:
        measured = [s for s in self.summaries.values() if s.measured_frames >= 2]
        if not measured:
            return "speed/distance: no measurable tracks"
        distances = np.array([s.distance_m for s in measured])
        tops = np.array([s.top_speed_kmh for s in measured if s.top_speed_kmh > 0])
        return (
            f"speed/distance: {len(measured)} tracks, "
            f"median distance {np.median(distances):.1f} m, "
            f"max top speed {tops.max() if len(tops) else 0:.1f} km/h, "
            f"{self.dropped_total} samples dropped above {self.config.max_plausible_kmh:.0f} km/h"
        )

    def draw(self, frame: np.ndarray, tracks: dict, frame_num: int, scale: float = 1.0) -> np.ndarray:
        for object_type in MEASURED_OBJECTS:
            object_tracks = tracks.get(object_type)
            if not object_tracks or frame_num >= len(object_tracks):
                continue
            for info in object_tracks[frame_num].values():
                speed, distance = info.get("speed"), info.get("distance")
                if speed is None or distance is None:
                    continue
                x1, _, x2, y2 = info["bbox"]
                x = int((x1 + x2) / 2)
                y = int(y2 + 40 * scale)
                draw_label(frame, f"{speed:.1f} km/h", (x, y), scale)
                draw_label(frame, f"{distance:.0f} m", (x, int(y + 18 * scale)), scale)
        return frame


#: The original class name, kept so existing notebooks and scripts keep importing.
SpeedAndDistance_Estimator = SpeedAndDistanceEstimator
