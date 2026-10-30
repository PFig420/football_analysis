"""Team assignment.

The decisive property is that kit matching must not depend on how brightly a player is lit.
On the sample clip -- white shirts against green ones, both light -- clustering raw BGR put
35 of 39 tracks on one team and 4 on the other, because the dominant axis in BGR space was
sunlit versus shadowed rather than white versus green.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import TeamConfig
from team_assigner import TEAM_A, TEAM_B, UNKNOWN_TEAM, TeamAssigner
from team_assigner.team_assigner import kit_feature

WHITE = np.array([235.0, 235.0, 235.0])
GREEN = np.array([60.0, 200.0, 70.0])


def dim(color, factor):
    """The same kit, in shade."""
    return np.asarray(color, dtype=float) * factor


# -- the feature -------------------------------------------------------------------------


def test_feature_is_invariant_to_brightness():
    bright, shaded = kit_feature(GREEN), kit_feature(dim(GREEN, 0.45))
    assert np.linalg.norm(bright - shaded) < 25.0


def test_feature_separates_kits_that_brightness_confuses():
    """White and green are far apart in hue but close in brightness."""
    white_to_green = np.linalg.norm(kit_feature(WHITE) - kit_feature(GREEN))
    green_to_shaded_green = np.linalg.norm(kit_feature(GREEN) - kit_feature(dim(GREEN, 0.45)))
    assert white_to_green > 3 * green_to_shaded_green


def test_greys_collapse_towards_the_origin():
    """Hue is meaningless without saturation, so it must not dominate for white or grey."""
    for grey in (np.array([40.0, 40.0, 40.0]), WHITE, np.array([128.0, 128.0, 128.0])):
        assert np.linalg.norm(kit_feature(grey)) < 20.0


def test_hue_wraparound_does_not_split_red():
    """Encoded as an angle instead of a vector, reds would sit at both ends of the scale."""
    just_below = kit_feature(np.array([5.0, 5.0, 200.0]))
    just_above = kit_feature(np.array([12.0, 5.0, 200.0]))
    assert np.linalg.norm(just_below - just_above) < 30.0


# -- fitting and voting --------------------------------------------------------------------


def mixed_lighting(color, n=12):
    return [dim(color, f) for f in np.linspace(0.4, 1.0, n)]


def test_fit_splits_two_kits_across_lighting():
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))

    teams = assigner.vote_from_colors({
        1: mixed_lighting(WHITE, 6),
        2: mixed_lighting(GREEN, 6),
        3: mixed_lighting(WHITE, 6),
        4: mixed_lighting(GREEN, 6),
    })
    assert teams[1] == teams[3]
    assert teams[2] == teams[4]
    assert teams[1] != teams[2]


def test_shaded_player_joins_the_right_team():
    """The failure mode that produced a 35/4 split on real footage."""
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))

    teams = assigner.vote_from_colors({
        1: [WHITE] * 5,
        2: [GREEN] * 5,
        3: [dim(GREEN, 0.4)] * 5,   # same kit, deep shade
    })
    assert teams[3] == teams[2], "a shaded player was assigned by brightness, not kit"


def test_team_colors_are_bgr_for_drawing():
    """Cluster centroids live in hue-saturation space and are not colours."""
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))

    for team in (TEAM_A, TEAM_B):
        color = assigner.color_of(team)
        assert len(color) == 3
        assert all(0 <= float(c) <= 255 for c in color)


def test_vote_survives_a_single_bad_crop():
    """The original locked a track's team in on first sight."""
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))

    teams = assigner.vote_from_colors({1: [GREEN] + [WHITE] * 7})
    clean = assigner.vote_from_colors({2: [WHITE] * 8})
    assert teams[1] == clean[2]


def test_fit_rejects_too_few_crops():
    with pytest.raises(ValueError, match="not enough player crops"):
        TeamAssigner().fit_colors([WHITE])


def test_voting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="fit_colors"):
        TeamAssigner().vote_from_colors({1: [WHITE]})


def test_unseen_track_is_unknown_not_guessed():
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))
    assert assigner.team_of(999) == UNKNOWN_TEAM


# -- goalkeepers ----------------------------------------------------------------------------


def test_goalkeepers_are_placed_by_pitch_position():
    """Keepers wear a kit that matches neither team, so they are placed, not matched."""
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))

    n = 10
    tracks = {
        "players": [
            {
                1: {"position_transformed": [5.0, 30.0], "team": TEAM_A},
                2: {"position_transformed": [95.0, 30.0], "team": TEAM_B},
            }
            for _ in range(n)
        ],
        "goalkeepers": [
            {
                7: {"bbox": [0, 0, 10, 10], "position_transformed": [1.0, 34.0]},
                8: {"bbox": [0, 0, 10, 10], "position_transformed": [99.0, 34.0]},
            }
            for _ in range(n)
        ],
    }

    placed = assigner.assign_goalkeeper_teams(tracks)
    # A keeper defends the end their own side attacks away from, so the keeper at x=1 belongs
    # to the team whose outfielders sit at x=95.
    assert placed[7] == TEAM_B
    assert placed[8] == TEAM_A
    assert tracks["goalkeepers"][0][7]["team"] == TEAM_B


def test_goalkeeper_without_enough_positions_is_left_unknown():
    assigner = TeamAssigner()
    assigner.fit_colors(mixed_lighting(WHITE) + mixed_lighting(GREEN))

    tracks = {
        "players": [
            {1: {"position_transformed": [5.0, 30.0], "team": TEAM_A},
             2: {"position_transformed": [95.0, 30.0], "team": TEAM_B}}
            for _ in range(5)
        ],
        "goalkeepers": [{7: {"bbox": [0, 0, 10, 10], "position_transformed": [1.0, 34.0]}}],
    }

    placed = assigner.assign_goalkeeper_teams(tracks)
    assert 7 not in placed
    assert tracks["goalkeepers"][0][7]["team"] == UNKNOWN_TEAM
