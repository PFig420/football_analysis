import numpy as np
import pytest

from config import PitchConfig
from view_transformer import ViewTransformer


def test_corners_map_to_target_rectangle():
    view = ViewTransformer()
    for pixel, expected in zip(view.pixel_vertices, view.target_vertices):
        assert view.transform_point(pixel) == pytest.approx(expected, abs=1e-3)


def test_round_trip_through_inverse():
    view = ViewTransformer()
    centre = view.pixel_vertices.mean(axis=0)
    metres = view.transform_point(centre)
    assert metres is not None
    assert view.inverse_transform_point(metres) == pytest.approx(centre, abs=1e-2)


def test_point_outside_quad_returns_none_and_is_counted():
    view = ViewTransformer()
    assert view.transform_point((-500.0, -500.0)) is None
    assert view.points_outside == 1
    assert view.coverage == 0.0


def test_coverage_tracks_the_fraction_inside():
    view = ViewTransformer()
    view.transform_point(view.pixel_vertices.mean(axis=0))
    view.transform_point((-500.0, -500.0))
    assert view.coverage == pytest.approx(0.5)


def test_rejects_non_convex_quad():
    bad = PitchConfig(pixel_vertices=[[0, 0], [100, 0], [10, 10], [0, 100]])
    with pytest.raises(ValueError, match="convex"):
        ViewTransformer(bad)


def test_rejects_duplicate_vertices():
    bad = PitchConfig(pixel_vertices=[[0, 0], [0, 0], [100, 100], [0, 100]])
    with pytest.raises(ValueError, match="duplicate"):
        ViewTransformer(bad)


def test_rejects_collinear_quad():
    bad = PitchConfig(pixel_vertices=[[0, 0], [10, 0], [20, 0], [30, 0]])
    with pytest.raises(ValueError):
        ViewTransformer(bad)


def test_known_scale_on_a_square_quad():
    """A 100x100 px quad declared as 10x20 m must scale each axis independently."""
    config = PitchConfig(
        pixel_vertices=[[0, 100], [0, 0], [100, 0], [100, 100]],
        length_m=10.0,
        width_m=20.0,
        calibrated=True,
    )
    view = ViewTransformer(config)
    assert view.transform_point((50.0, 50.0)) == pytest.approx([5.0, 10.0], abs=1e-3)


def test_stub_coverage_is_reported(track_stub):
    """The sample clip's real coverage -- a third of player-frames fall outside the quad."""
    view = ViewTransformer()
    tracks = {"players": track_stub["players"]}
    for frame_tracks in tracks["players"]:
        for info in frame_tracks.values():
            x1, _, x2, y2 = info["bbox"]
            info["position_adjusted"] = ((x1 + x2) / 2, y2)

    view.add_transformed_position_to_tracks(tracks)
    assert view.points_seen > 10_000
    assert 0.5 < view.coverage < 0.8
    assert "coverage" in view.coverage_report()
