"""Bounding-box geometry.

Drawing helpers return ints because OpenCV wants pixel indices. The positions that feed the
measurement chain stay float: they are about to be camera-corrected, projected and
differenced, and rounding each one to a whole pixel adds avoidable noise to every speed.
"""
from __future__ import annotations


def get_center_of_bbox(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox) -> float:
    return bbox[2] - bbox[0]


def get_bbox_height(bbox) -> float:
    return bbox[3] - bbox[1]


def measure_distance(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2) -> tuple[float, float]:
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox) -> tuple[float, float]:
    """Midpoint of the bottom edge, in sub-pixel precision."""
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)
