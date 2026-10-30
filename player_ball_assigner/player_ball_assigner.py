"""Ball possession.

Two changes that matter to the headline percentage:

* **Interpolated ball positions no longer count.** The ball is detected in only 37% of the
  sample clip's frames; the original linearly filled the rest and then treated all frames as
  equally good evidence, so most of the possession figure rested on straight lines drawn
  between distant sightings. Frames whose ball position was interpolated are excluded from
  the denominator, and the coverage is reported alongside the result.
* **No crash on frame 0.** The original carried the previous frame's holder forward with
  `team_ball_control[-1]`, which raises `IndexError` if the very first frame has no player
  near the ball. It survived on the sample clip by luck.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import BallConfig
from utils.bbox_utils import get_center_of_bbox, measure_distance
from utils.draw_utils import draw_text_panel

NO_PLAYER = -1
UNKNOWN_TEAM = 0


class PlayerBallAssigner:
    def __init__(self, config: BallConfig | None = None, scale: float = 1.0):
        self.config = config or BallConfig()
        #: Distances are in pixels, so the threshold has to follow the resolution.
        self.max_distance = self.config.max_possession_distance_px * scale

    def assign_ball_to_player(self, players: dict, ball_bbox) -> int:
        """Track ID of the closest player within range, else NO_PLAYER."""
        if not players or ball_bbox is None:
            return NO_PLAYER

        ball_position = get_center_of_bbox(ball_bbox)
        best_id, best_distance = NO_PLAYER, self.max_distance

        for track_id, player in players.items():
            x1, _, x2, y2 = player["bbox"]
            # Measure from the player's feet, left and right, rather than the box centre.
            distance = min(
                measure_distance((x1, y2), ball_position),
                measure_distance((x2, y2), ball_position),
            )
            if distance < best_distance:
                best_id, best_distance = track_id, distance

        return best_id


@dataclass
class PossessionTracker:
    """Per-frame possession, with an explicit record of how much is actually measured."""

    team_per_frame: list[int] = field(default_factory=list)
    #: True where the frame contributed real evidence (ball detected and a player in range).
    measured: list[bool] = field(default_factory=list)

    def add(self, team: int, measured: bool) -> None:
        if not measured:
            # Carry the last known holder forward for display continuity, but do not let it
            # vote. UNKNOWN_TEAM until someone has held the ball.
            team = self.team_per_frame[-1] if self.team_per_frame else UNKNOWN_TEAM
        self.team_per_frame.append(team)
        self.measured.append(measured)

    def __len__(self) -> int:
        return len(self.team_per_frame)

    def shares_until(self, frame_num: int) -> tuple[float, float, float]:
        """(team 1 share, team 2 share, coverage) over frames 0..frame_num inclusive."""
        teams = np.array(self.team_per_frame[: frame_num + 1])
        measured = np.array(self.measured[: frame_num + 1], dtype=bool)
        if teams.size == 0:
            return 0.0, 0.0, 0.0

        counted = teams[measured]
        coverage = measured.mean()
        if counted.size == 0:
            return 0.0, 0.0, coverage

        one = float((counted == 1).sum())
        two = float((counted == 2).sum())
        total = one + two
        if total == 0:
            return 0.0, 0.0, coverage
        return one / total, two / total, coverage

    def summary(self) -> dict:
        one, two, coverage = self.shares_until(len(self) - 1)
        # Cast off numpy scalars so the report is plain JSON.
        return {
            "team_1_possession": round(float(one), 4),
            "team_2_possession": round(float(two), 4),
            "measured_frames": int(sum(self.measured)),
            "total_frames": len(self),
            "coverage": round(float(coverage), 4),
        }

    def draw(self, frame: np.ndarray, frame_num: int, box, scale: float = 1.0) -> np.ndarray:
        one, two, coverage = self.shares_until(frame_num)
        return draw_text_panel(
            frame,
            [
                f"Team 1 possession: {one * 100:.1f}%",
                f"Team 2 possession: {two * 100:.1f}%",
                f"(measured on {coverage * 100:.0f}% of frames)",
            ],
            box=box,
            scale=scale,
        )
