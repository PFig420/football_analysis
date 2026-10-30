"""Assign players to teams from kit colour.

Fixes to the original:

* **No hardcoded track IDs.** The original contained `if player_id == 91: team_id = 1`, a
  patch for one misclassified goalkeeper in one clip. The cause was goalkeepers being
  relabelled as outfield players and then clustered by shirt colour, which they never match.
  Goalkeepers are now their own class, excluded from the kit fit they were distorting, and
  given a team by a separate positional heuristic.
* **Team colours are fitted over a sample of frames**, not frame 0 alone, with a stable
  `n_init`. One unlucky first frame no longer defines both teams for the whole clip.
* **A track's team is a running majority vote**, not whatever the first crop happened to
  look like. The original cached by track ID on first sight and never revisited.
* The corner-pixel background heuristic has a fallback for crops where it degenerates
  (tight boxes, overlapping players) instead of silently returning the grass colour.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

from config import TeamConfig

logger = logging.getLogger(__name__)

TEAM_A, TEAM_B = 1, 2
UNKNOWN_TEAM = 0


def kit_feature(bgr) -> np.ndarray:
    """Project a kit colour onto hue and saturation, discarding brightness.

    Clustering raw BGR splits players by *illumination* rather than by kit. Measured on the
    sample clip -- white shirts against green ones, both light -- raw BGR put 35 of 39 tracks
    on one team and 4 on the other, because the dominant axis in BGR space was sunlit versus
    shadowed. The same crops in this space split 21/18, which is what two teams look like.

    Hue is encoded as a vector rather than an angle so that red does not sit at both ends of
    the scale, and it is scaled by saturation so that greys and whites, whose hue is
    meaningless, collapse towards the origin instead of landing at an arbitrary angle.
    """
    pixel = np.uint8([[[np.clip(bgr[0], 0, 255), np.clip(bgr[1], 0, 255), np.clip(bgr[2], 0, 255)]]])
    hue, saturation, _ = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    angle = np.deg2rad(hue * 2.0)  # OpenCV packs hue into 0-179
    return np.array([np.cos(angle) * saturation, np.sin(angle) * saturation])


class TeamAssigner:
    def __init__(self, config: TeamConfig | None = None):
        self.config = config or TeamConfig()
        self.kmeans: KMeans | None = None
        self.team_colors: dict[int, np.ndarray] = {}
        self._votes: dict[int, Counter] = defaultdict(Counter)
        self._decided: dict[int, int] = {}

    # -- colour extraction -------------------------------------------------------------

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 4:
            return None
        return frame[y1:y2, x1:x2]

    def get_player_color(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """Dominant shirt colour for a detection in a frame."""
        return self.color_from_crop(self._crop(frame, bbox))

    def color_from_crop(self, image: np.ndarray | None) -> np.ndarray | None:
        """Dominant shirt colour: cluster the top half of the crop into kit and background.

        Takes a crop rather than a frame so the pipeline can keep a few small crops per
        track from its single streaming pass instead of holding whole frames in memory.
        """
        if image is None or image.size == 0:
            return None
        top_half = image[: max(1, image.shape[0] // 2), :]
        if top_half.size == 0:
            return None

        pixels = top_half.reshape(-1, 3).astype(np.float32)
        if len(np.unique(pixels, axis=0)) < 2:
            return pixels.mean(axis=0)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=0)
        labels = kmeans.fit_predict(pixels).reshape(top_half.shape[:2])

        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background = Counter(corners).most_common(1)[0][0]

        centre = labels[labels.shape[0] // 2, labels.shape[1] // 2]
        if centre == background:
            # The corner heuristic has failed: the box is tight enough, or crowded enough,
            # that the centre and the border are the same cluster. Fall back to the cluster
            # covering less of the crop, which is the kit rather than the grass.
            counts = np.bincount(labels.ravel(), minlength=2)
            player_cluster = int(np.argmin(counts))
        else:
            player_cluster = int(centre)

        return kmeans.cluster_centers_[player_cluster]

    # -- fitting -----------------------------------------------------------------------

    def crop_player(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """Public wrapper so callers can harvest crops during a streaming pass."""
        crop = self._crop(frame, bbox)
        return None if crop is None else crop.copy()

    def fit(self, samples) -> None:
        """Fit the two team centroids.

        `samples` is an iterable of (frame, {track_id: {"bbox": ...}}) covering several
        frames spread across the clip. Pass outfield players only -- goalkeepers wear a
        different kit and pull the centroids apart.
        """
        colors = []
        for frame, detections in samples:
            for info in detections.values():
                color = self.get_player_color(frame, info["bbox"])
                if color is not None:
                    colors.append(color)
        self.fit_colors(colors)

    def fit_colors(self, colors: list) -> None:
        """Fit the two team centroids from already-extracted kit colours.

        Clustering happens in hue-saturation space (see `kit_feature`); the BGR colours are
        kept only so the annotations can be drawn in something recognisable.
        """
        if len(colors) < 2:
            raise ValueError(
                f"not enough player crops to fit team colours (got {len(colors)}); "
                "check the detector is finding players"
            )

        bgr = np.array(colors, dtype=float)
        features = np.array([kit_feature(c) for c in colors])

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        labels = self.kmeans.fit_predict(features)

        # A cluster centroid lives in hue-saturation space and is not a colour, so take the
        # mean BGR of each cluster's members for drawing.
        self.team_colors = {
            team: bgr[labels == index].mean(axis=0) if (labels == index).any() else np.array([0, 0, 255.0])
            for team, index in ((TEAM_A, 0), (TEAM_B, 1))
        }
        logger.info(
            "fitted team colours from %d crops: team 1 BGR %s (n=%d), team 2 BGR %s (n=%d)",
            len(colors),
            np.round(self.team_colors[TEAM_A]).astype(int).tolist(),
            int((labels == 0).sum()),
            np.round(self.team_colors[TEAM_B]).astype(int).tolist(),
            int((labels == 1).sum()),
        )

    # -- assignment --------------------------------------------------------------------

    def vote_from_colors(self, track_colors: dict[int, list]) -> dict[int, int]:
        """Decide each track's team by majority vote over its sampled colours.

        The original locked a track's team in on first sight, so a single bad crop -- a
        player occluded at the moment they first appeared -- mislabelled them for the whole
        clip.
        """
        if self.kmeans is None:
            raise RuntimeError("call fit_colors() before vote_from_colors()")

        decided: dict[int, int] = {}
        for track_id, colors in track_colors.items():
            if not colors:
                continue
            predictions = self.kmeans.predict(np.array([kit_feature(c) for c in colors])) + 1
            self._votes[track_id].update(int(p) for p in predictions)
            decided[track_id] = self._majority(track_id)
        self._decided.update(decided)
        return decided

    def observe(self, frame: np.ndarray, bbox, track_id: int) -> int:
        """Record one observation of a track and return its current best team."""
        if self.kmeans is None:
            raise RuntimeError("call fit() before observe()")

        if len(self._votes[track_id]) and sum(self._votes[track_id].values()) >= self.config.vote_observations:
            # Enough evidence; stop paying for the clustering on every later frame.
            return self._decided.setdefault(track_id, self._majority(track_id))

        color = self.get_player_color(frame, bbox)
        if color is not None:
            team = int(self.kmeans.predict(kit_feature(color).reshape(1, -1))[0]) + 1
            self._votes[track_id][team] += 1

        return self._majority(track_id)

    def _majority(self, track_id: int) -> int:
        votes = self._votes.get(track_id)
        if not votes:
            return UNKNOWN_TEAM
        return votes.most_common(1)[0][0]

    def team_of(self, track_id: int) -> int:
        return self._decided.get(track_id) or self._majority(track_id)

    def color_of(self, team: int):
        return self.team_colors.get(team, (0, 0, 255))

    # -- goalkeepers -------------------------------------------------------------------

    def assign_goalkeeper_teams(self, tracks: dict) -> dict[int, int]:
        """Give each goalkeeper track a team from where they stand.

        A keeper defends the end their own team is attacking away from, so a keeper belongs
        to whichever team's outfield players are, on average, *further* up the pitch from
        them. This needs `position_transformed`, so run it after the view transform.

        This is a heuristic, not a kit match -- keepers wear colours that match neither team
        by design. Tracks it cannot place are left as UNKNOWN_TEAM rather than guessed.
        """
        team_x: dict[int, list[float]] = defaultdict(list)
        for frame_tracks in tracks.get("players", []):
            for info in frame_tracks.values():
                position, team = info.get("position_transformed"), info.get("team")
                if position is not None and team in (TEAM_A, TEAM_B):
                    team_x[team].append(position[0])

        if len(team_x) < 2:
            logger.warning("cannot place goalkeepers: outfield teams have no transformed positions")
            return {}

        team_centre = {team: float(np.median(xs)) for team, xs in team_x.items()}

        keeper_x: dict[int, list[float]] = defaultdict(list)
        for frame_tracks in tracks.get("goalkeepers", []):
            for track_id, info in frame_tracks.items():
                position = info.get("position_transformed")
                if position is not None:
                    keeper_x[track_id].append(position[0])

        assignment: dict[int, int] = {}
        for track_id, xs in keeper_x.items():
            if len(xs) < 3:
                continue
            x = float(np.median(xs))
            # Furthest team centre wins: that is the end this keeper's side attacks.
            assignment[track_id] = max(team_centre, key=lambda t: abs(team_centre[t] - x))

        for frame_tracks in tracks.get("goalkeepers", []):
            for track_id, info in frame_tracks.items():
                team = assignment.get(track_id, UNKNOWN_TEAM)
                info["team"] = team
                info["team_color"] = self.color_of(team) if team else (255, 255, 255)

        logger.info("placed %d goalkeeper track(s) by pitch position", len(assignment))
        return assignment
