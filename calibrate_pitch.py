"""Derive a pitch homography from the markings visible in a frame.

    python calibrate_pitch.py --input input_videos/clip.mp4 --out config.yaml

Writes a `pitch:` config block and an overlay image to check the fit by eye. Speed and
distance are only as good as this calibration, so look at the overlay before trusting them.

**Calibrate on the reference frame.** The homography camera model maps every other frame
into frame 0, so `position_adjusted` is in frame-0 pixel space and that is the space the
quad must be defined in. `--frame` exists for the case where frame 0 is unusable (a cut, a
replay wipe), but then the camera model's reference has to move with it.

What has to be visible in the frame:

* **the halfway line**, end to end,
* **both touchlines**,
* **the centre circle**, at least in part.

Those carry enough known geometry on their own: the circle is 9.15 m in radius, the halfway
line is perpendicular to the touchlines, and the touchlines are a pitch width apart. No
penalty area is needed -- which matters, because a camera covering the middle third never
sees one.

Pitch width defaults to 68 m, the IFAB recommendation and the required size for UEFA
competition. Measure yours and pass `--width` if it differs; everything scales with it.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from utils import iter_frames, probe_video

CIRCLE_RADIUS_M = 9.15
DEFAULT_WIDTH_M = 68.0


# -- detection ---------------------------------------------------------------------------


def pitch_mask(frame: np.ndarray) -> np.ndarray:
    """The playing surface: the largest connected region of grass."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue > 30) & (hue < 90) & (sat > 60) & (val > 40)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise RuntimeError("no grass found -- is this a football pitch?")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def marking_mask(frame: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """White paint on the grass: bright and desaturated, inside the pitch."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 70)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, pitch)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def line_segments(markings: np.ndarray) -> np.ndarray:
    segments = cv2.HoughLinesP(
        markings, 1, np.pi / 360, threshold=70, minLineLength=120, maxLineGap=30
    )
    if segments is None:
        raise RuntimeError("no straight markings found")
    return np.asarray(segments).reshape(-1, 4)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through points, as (origin, unit direction)."""
    origin = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - origin)
    return origin, vt[0]


def _distance_to_line(points: np.ndarray, origin, direction) -> np.ndarray:
    v = points - np.asarray(origin, float)
    return np.abs(v[:, 0] * direction[1] - v[:, 1] * direction[0])


def boundary_line(pitch: np.ndarray, top: bool, step: int = 20):
    """Fit a line to the top or bottom edge of the pitch mask.

    Touchlines are better found this way than with Hough. The painted line is thin, often
    occluded by players and, at the far side, blends into the advertising boards -- so the
    longest segment Hough finds may span only a quarter of the frame, and extrapolating its
    angle across the full width magnifies a small error into metres. The grass boundary runs
    the whole width and sits within a few centimetres of the paint.
    """
    height, width = pitch.shape[:2]
    points = []
    for column in range(0, width, step):
        rows = np.where(pitch[:, column] > 0)[0]
        if len(rows):
            points.append((column, rows.min() if top else rows.max()))
    points = np.array(points, float)
    if len(points) < 10:
        raise RuntimeError("could not trace the pitch boundary")

    # Two rounds of trimming: stands and dugouts intrude on the boundary here and there.
    for _ in range(2):
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        residual = points[:, 1] - np.polyval((slope, intercept), points[:, 0])
        keep = np.abs(residual) < max(3.0 * residual.std(), 6.0)
        if keep.sum() < 10:
            break
        points = points[keep]

    slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
    if abs(slope) > 0.5:
        # A touchline is never this steep in a broadcast view. Usually it means the grass
        # region ends inside the frame, so the column-wise scan traced its slanted side
        # rather than the touchline -- and a seed this bad lets the refinement settle on
        # some other marking entirely.
        raise RuntimeError(
            f"pitch boundary came out at slope {slope:.2f}, which is not a touchline; "
            f"the grass may not span the frame width"
        )
    origin = np.array([0.0, intercept])
    direction = np.array([1.0, slope]) / np.hypot(1.0, slope)
    return origin, direction


def refine_line(pixels: np.ndarray, origin, direction, tolerance: float, rounds: int = 3):
    """Re-fit a line to the marking pixels that lie near it, a few times over."""
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    for _ in range(rounds):
        near = _distance_to_line(pixels, origin, direction) < tolerance
        if near.sum() < 50:
            break
        origin, direction = _fit_line(pixels[near])
    return origin, direction


def longest_by_orientation(segments: np.ndarray, vertical: bool):
    angle = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1],
                                  segments[:, 2] - segments[:, 0])) % 180
    length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    pick = (angle > 50) & (angle < 130) if vertical else ((angle < 30) | (angle > 150))
    return segments[pick], length[pick]


def fit_circle(markings: np.ndarray, straight: np.ndarray, halfway=None, seed: int = 0):
    """RANSAC ellipse over marking pixels, after erasing the straight lines.

    Players occlude the circle, so it arrives as several disconnected arcs and a
    contour-by-contour fit latches onto one of them. Fitting the whole point set instead,
    with RANSAC to reject the fragments that belong to other markings, recovers it.
    """
    rng = np.random.default_rng(seed)
    residue = markings.copy()
    length = np.hypot(straight[:, 2] - straight[:, 0], straight[:, 3] - straight[:, 1])
    for x1, y1, x2, y2 in straight[length > 200]:
        cv2.line(residue, (x1, y1), (x2, y2), 0, 27)

    ys, xs = np.where(residue > 0)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if halfway is not None:
        # The circle straddles the halfway line, reaching 9.15 m either side of it. Paint far
        # from that band belongs to some other marking, and leaving it in lets RANSAC settle
        # on a larger ellipse that fits an arc of the penalty area instead.
        points = points[_distance_to_line(points, *halfway) < 0.35 * markings.shape[1]]
    if len(points) < 200:
        raise RuntimeError("not enough curved markings to find the centre circle")

    def residual(ellipse, p):
        (cx, cy), (major, minor), angle = ellipse
        a, b = major / 2.0, minor / 2.0
        if a < 5 or b < 5:
            return None
        t = np.deg2rad(angle)
        rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = (p - np.array([cx, cy])) @ rot.T
        r = np.sqrt((q[:, 0] / a) ** 2 + (q[:, 1] / b) ** 2)
        return np.abs(r - 1.0) * ((a + b) / 2.0)

    best, best_inliers = None, 0
    for _ in range(4000):
        sample = points[rng.choice(len(points), 6, replace=False)]
        try:
            ellipse = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        (cx, cy), (major, minor), _ = ellipse
        if max(major, minor) > 1400 or min(major, minor) < 60:
            continue
        if halfway is not None and _distance_to_line(np.array([[cx, cy]]), *halfway)[0] > 250:
            continue
        r = residual(ellipse, points)
        if r is None:
            continue
        inliers = int((r < 4.0).sum())
        if inliers > best_inliers:
            best, best_inliers = ellipse, inliers

    if best is None:
        raise RuntimeError("could not fit the centre circle")
    r = residual(best, points)
    return cv2.fitEllipse(points[r < 4.0]), points[r < 4.0], best_inliers


# -- solving -----------------------------------------------------------------------------


def to_world(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    q = np.column_stack([points, np.ones(len(points))]) @ homography.T
    return q[:, :2] / q[:, 2:3]


def solve(halfway, far_touchline, near_touchline, circle, width_m, initial):
    """Least squares over the 8 homography parameters, against known pitch geometry.

    No point correspondences: each group of pixels is constrained by what it *is* -- the
    halfway line lies at x = 0, the touchlines at y = 0 and y = width, the circle at
    9.15 m from the centre spot. This uses every detected pixel rather than four
    hand-picked corners, and avoids having to identify individual landmarks, which
    perspective makes unreliable (the centre of a circle does not project to the centre of
    its ellipse).
    """
    from scipy.optimize import least_squares

    def residuals(p):
        h = np.append(p, 1.0).reshape(3, 3)
        circle_world = to_world(h, circle)
        return np.concatenate([
            to_world(h, halfway)[:, 0],
            to_world(h, far_touchline)[:, 1],
            to_world(h, near_touchline)[:, 1] - width_m,
            np.hypot(circle_world[:, 0], circle_world[:, 1] - width_m / 2.0) - CIRCLE_RADIUS_M,
        ])

    out = least_squares(residuals, (initial / initial[2, 2]).ravel()[:8],
                        method="lm", max_nfev=20000)
    homography = np.append(out.x, 1.0).reshape(3, 3)

    sizes = [len(halfway), len(far_touchline), len(near_touchline), len(circle)]
    bounds = np.cumsum([0] + sizes)
    r = residuals(out.x)
    rms = {
        name: float(np.sqrt((r[bounds[i]:bounds[i + 1]] ** 2).mean()))
        for i, name in enumerate(("halfway line", "far touchline", "near touchline", "centre circle"))
    }
    return homography, rms


def widest_quad(homography: np.ndarray, width_m: float, shape, margin: int = 8):
    """The largest world rectangle spanning the pitch width that stays inside the frame."""
    inverse = np.linalg.inv(homography)
    height, image_width = shape[:2]

    def projects_inside(x0, x1):
        corners = np.array([[x0, width_m], [x0, 0], [x1, 0], [x1, width_m]], float)
        p = to_world(inverse, corners)
        return (
            (p[:, 0] >= margin).all() and (p[:, 0] <= image_width - margin).all()
            and (p[:, 1] >= margin).all() and (p[:, 1] <= height - margin).all()
        )

    best = None
    for x0 in np.arange(-70, 20, 0.5):
        for x1 in np.arange(x0 + 10, 70, 0.5):
            if projects_inside(x0, x1) and (best is None or x1 - x0 > best[1] - best[0]):
                best = (float(x0), float(x1))
    if best is None:
        raise RuntimeError("no usable rectangle projects inside the frame")

    x0, x1 = best
    corners = np.array([[x0, width_m], [x0, 0], [x1, 0], [x1, width_m]], float)
    return to_world(inverse, corners), x1 - x0


# -- driver ------------------------------------------------------------------------------


def calibrate(frame: np.ndarray, width_m: float = DEFAULT_WIDTH_M, verbose: bool = True):
    pitch = pitch_mask(frame)
    markings = marking_mask(frame, pitch)
    segments = line_segments(markings)

    vertical, v_len = longest_by_orientation(segments, vertical=True)
    horizontal, h_len = longest_by_orientation(segments, vertical=False)
    if not len(vertical):
        raise RuntimeError("no near-vertical marking found -- is the halfway line visible?")

    ys, xs = np.where(markings > 0)
    pixels = np.column_stack([xs, ys]).astype(float)
    midline = frame.shape[0] / 2

    def line_of(seg):
        return _fit_line(np.array([[seg[0], seg[1]], [seg[2], seg[3]]], float))

    # Halfway line: the longest near-vertical marking. With the centre circle in view -- a
    # precondition for this script -- nothing else vertical comes close in length.
    halfway_line = refine_line(pixels, *line_of(vertical[int(np.argmax(v_len))]), 6.0)

    ellipse, circle_points, inliers = fit_circle(markings, segments, halfway=halfway_line)

    # Far touchline: from the grass boundary. The paint there is thin, often occluded, and
    # blends into the advertising boards, but the grass stops exactly at it.
    far_line = refine_line(pixels[pixels[:, 1] < midline], *boundary_line(pitch, top=True), 8.0)

    # Near touchline: from the paint. The grass boundary is NOT the touchline on this side --
    # run-off grass continues past it toward the camera, which on the sample clip puts the
    # boundary some 37 px beyond the line, worth several metres once projected.
    lower = np.where((horizontal[:, 1] + horizontal[:, 3]) / 2 > midline, h_len, -1)
    if lower.max() <= 0:
        raise RuntimeError("near touchline not found -- is it inside the frame?")
    near_line = refine_line(pixels[pixels[:, 1] > midline],
                            *line_of(horizontal[int(np.argmax(lower))]), 8.0)

    halfway_m = _distance_to_line(pixels, *halfway_line) < 6.0
    far_m = (_distance_to_line(pixels, *far_line) < 7.0) & (pixels[:, 1] < frame.shape[0] / 2)
    near_m = (_distance_to_line(pixels, *near_line) < 7.0) & (pixels[:, 1] > frame.shape[0] / 2)

    rng = np.random.default_rng(0)

    def sample(mask, n):
        idx = np.where(mask)[0]
        if len(idx) < 20:
            raise RuntimeError("a pitch marking was detected but has too few pixels")
        return pixels[rng.choice(idx, min(n, len(idx)), replace=False)]

    halfway = sample(halfway_m, 600)
    far = sample(far_m, 600)
    near = sample(near_m, 600)
    circle = circle_points[rng.choice(len(circle_points), min(1200, len(circle_points)), replace=False)]

    # Initial guess: halfway line's two ends, and the ellipse's extremes. Approximate -- the
    # optimiser does the real work.
    def intersect(line_a, line_b):
        (o1, d1), (o2, d2) = line_a, line_b
        t = np.linalg.solve(np.array([d1, -d2]).T, np.asarray(o2, float) - np.asarray(o1, float))
        return np.asarray(o1, float) + t[0] * np.asarray(d1, float)

    (ellipse_cx, ellipse_cy), (major, minor), angle = ellipse
    axis = np.array([np.cos(np.deg2rad(angle + 90)), np.sin(np.deg2rad(angle + 90))])
    semi = max(major, minor) / 2.0
    centre = np.array([ellipse_cx, ellipse_cy])

    source = np.float32([intersect(halfway_line, far_line), intersect(halfway_line, near_line),
                         centre - axis * semi, centre + axis * semi])
    target = np.float32([[0, 0], [0, width_m],
                         [-CIRCLE_RADIUS_M, width_m / 2], [CIRCLE_RADIUS_M, width_m / 2]])
    initial = cv2.getPerspectiveTransform(source, target)

    homography, rms = solve(halfway, far, near, circle, width_m, initial)

    # Every constraint above is symmetric about the halfway line -- it sits at x = 0, the
    # touchlines run parallel to it, and the centre circle straddles it -- so the sign of
    # world x is genuinely undetermined, and which of the two solutions the optimiser lands
    # on depends on the arbitrary sign of the ellipse's reported axis. Distances are
    # identical either way, but the output should not be a coin flip: pin world +x to the
    # direction image +x points.
    height, width = frame.shape[:2]
    probe = to_world(homography, np.array([[0.25 * width, 0.5 * height],
                                           [0.75 * width, 0.5 * height]]))
    if probe[1, 0] < probe[0, 0]:
        homography = np.diag([-1.0, 1.0, 1.0]) @ homography

    if verbose:
        print(f"detected: halfway {halfway_m.sum()} px, far touchline {far_m.sum()} px, "
              f"near touchline {near_m.sum()} px, circle {inliers} px")
        print("fit residuals (metres):")
        for name, value in rms.items():
            print(f"   {name:16s} {value:.3f}")
    return homography, rms, ellipse, (halfway_line, far_line, near_line)


def write_overlay(frame, homography, quad, ellipse, width_m, path):
    canvas = frame.copy()
    inverse = np.linalg.inv(homography)
    # Re-draw the pitch from world coordinates: if the calibration is right, these land on
    # the real markings.
    for y in (0.0, width_m):
        pts = to_world(inverse, np.array([[x, y] for x in np.arange(-60, 60, 0.5)]))
        cv2.polylines(canvas, [pts.astype(np.int32)], False, (0, 255, 255), 2)
    pts = to_world(inverse, np.array([[0.0, y] for y in np.arange(0, width_m + 0.5, 0.5)]))
    cv2.polylines(canvas, [pts.astype(np.int32)], False, (255, 0, 0), 2)
    theta = np.linspace(0, 2 * np.pi, 361)
    pts = to_world(inverse, np.column_stack([CIRCLE_RADIUS_M * np.cos(theta),
                                             width_m / 2 + CIRCLE_RADIUS_M * np.sin(theta)]))
    cv2.polylines(canvas, [pts.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(canvas, [quad.astype(np.int32)], True, (255, 255, 255), 2)
    cv2.imwrite(str(path), canvas)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--frame", type=int, default=0,
                        help="reference frame to calibrate on (default 0, which is what the "
                             "camera model uses as its reference view)")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH_M,
                        help="pitch width in metres (default 68)")
    parser.add_argument("--out", default="config.yaml")
    parser.add_argument("--overlay", default="calibration_check.png")
    args = parser.parse_args(argv)

    meta = probe_video(args.input)
    print(meta)
    frame = None
    for i, f in enumerate(iter_frames(args.input, limit=args.frame + 1)):
        if i == args.frame:
            frame = f
    if frame is None:
        raise SystemExit(f"frame {args.frame} not found")

    homography, rms, ellipse, _ = calibrate(frame, args.width)
    quad, length_m = widest_quad(homography, args.width, frame.shape)

    # The far touchline is routinely the worst of the four: it is seeded from the grass
    # boundary, which sits a little outside the paint. Under a metre is fine.
    worst_name, worst = max(rms.items(), key=lambda kv: kv[1])
    verdict = "good" if worst < 1.0 else "QUESTIONABLE -- check the overlay"
    print(f"\nworst residual {worst:.2f} m ({worst_name}): {verdict}")
    print(f"quad covers {length_m:.1f} x {args.width:.1f} m")

    lines = ["pitch:", "  pixel_vertices:"]
    labels = ["near touchline, left", "far touchline, left",
              "far touchline, right", "near touchline, right"]
    for (x, y), label in zip(quad, labels):
        lines.append(f"  - [{x:.1f}, {y:.1f}]   # {label}")
    lines += [f"  length_m: {length_m:.1f}", f"  width_m: {args.width:.1f}", "  calibrated: true"]
    text = "\n".join(lines) + "\n"

    with open(args.out, "w") as handle:
        handle.write(f"# Calibrated from {meta.path} frame {args.frame}.\n")
        handle.write("# Residuals (m): " + ", ".join(f"{k} {v:.2f}" for k, v in rms.items()) + "\n")
        handle.write(text)
    write_overlay(frame, homography, quad, ellipse, args.width, args.overlay)

    print(f"\nwrote {args.out} and {args.overlay}\n")
    print(text)
    print("Look at the overlay before using this: the drawn touchlines, halfway line and "
          "centre circle should sit on the real ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
