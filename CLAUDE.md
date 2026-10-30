# Fix plan — football_analysis

Task spec for an agent. Read all of it before touching code. Work the phases in order;
within a phase the items are independent.

This repo is a clone of Abdullah Tarek's football-analysis tutorial (3 commits, original
author in `git log`). It is a YOLO + ByteTrack pipeline that annotates one broadcast clip
with player tracks, team colours, possession %, camera motion and per-player speed/distance.
657 lines, 7 modules. The structure is sound; several of the numbers it produces are wrong.

Your job is to make the metric outputs correct and the pipeline runnable on clips other than
the one it was written for. Do not restructure modules that are not listed below.

---

## 1. Current state — what you can and cannot run

**Missing assets.** `models/` and `input_videos/` contain only placeholder `.txt` files. The
README links the weights (`best.pt`) and the sample clip (`08fd33_4.mp4`) on Google Drive.
`main.py` will die at `read_video` on a fresh checkout. **Do not download them yourself —
ask the user.** Everything in Phase 0 and most of Phase 1 can be done without them.

**What you do have:** `stubs/track_stubs.pkl` and `stubs/camera_movement_stub.pkl`, covering
**750 frames** (31 s at 24 fps) of the sample clip. These are your fixture. Everything
downstream of YOLO — camera compensation, the homography, speed/distance, possession, team
assignment — is testable from the stub alone, with no model and no video. Build on that.

Stub contents, verified:

| Fact | Value |
| --- | --- |
| Frames | 750 (players / referees / ball lists all 750 long) |
| Frame size (from bbox extents) | x 2–1922, y 204–933 → the clip is 1920×1080 |
| Mean players detected per frame | 20.07 |
| Distinct player track IDs over the clip | **39** (for ~22 players on pitch) |
| Frames with a ball detection | **278 / 750 (37 %)** |
| Frames with non-zero camera movement | 329 / 750 |
| Max abs camera movement value | 67.8 px |

Python: there is no venv in this repo. `/home/pedro/football-tests/.venv/bin/python` has
numpy/pandas/matplotlib/sklearn but **no cv2 and no ultralytics**. Create a proper venv for
this repo (see Phase 0.1); `uv` is vendored at `/home/pedro/football-tests/uvbin/uv`.

---

## 2. Verified baseline — reproduce this before you change anything

I replicated the shipped pipeline's metric path (foot position → subtract camera movement →
homography → 5-frame speed window) directly from the stubs. **Reproduce these exact numbers
first.** They are your regression baseline; if your reproduction disagrees, stop and find out
why before fixing anything.

| Metric (750 frames) | With camera correction (shipped) | With camera correction **disabled** |
| --- | --- | --- |
| Speed samples | 1972 | 1979 |
| Median speed | 6.3 km/h | **5.6 km/h** |
| p95 speed | 19.8 km/h | **15.7 km/h** |
| Max speed | 59.5 km/h | **37.4 km/h** |
| Samples > 36 km/h (physically impossible) | **0.7 %** | **0.1 %** |
| Median per-track distance over 31 s | 52.9 m | **38.7 m** |

**Read that table carefully. The camera compensation makes every statistic worse.** A correct
camera compensation removes camera-induced apparent motion and should *tighten* the speed
distribution. This one widens it, inflates median distance by 37 %, and triples the rate of
physically impossible speeds. That is empirical proof the estimator is injecting noise rather
than removing it, and it is why Phase 1.1 comes first.

Note also what is *not* broken: a median of 6.3 km/h and ~9 km/90min extrapolated distance
look plausible. The bugs here produce believable-looking numbers, which is exactly what makes
them worth fixing properly rather than eyeballing the output video.

Reproduction script — save as `tests/baseline_metrics.py` and keep it working throughout:

```python
import pickle, numpy as np
from matplotlib.path import Path

t   = pickle.load(open("stubs/track_stubs.pkl", "rb"))
cam = pickle.load(open("stubs/camera_movement_stub.pkl", "rb"))

src = np.array([[110,1035],[265,275],[910,260],[1640,915]], float)
dst = np.array([[0,68],[0,0],[23.32,0],[23.32,68]], float)
A, b = [], []
for (x, y), (u, v) in zip(src, dst):
    A.append([x,y,1,0,0,0,-u*x,-u*y]); b.append(u)
    A.append([0,0,0,x,y,1,-v*x,-v*y]); b.append(v)
H = np.append(np.linalg.solve(np.array(A), np.array(b)), 1).reshape(3, 3)
P = Path(src)

def run(use_cam):
    pos = {}
    for fn, f in enumerate(t["players"]):
        for tid, i in f.items():
            x1, y1, x2, y2 = i["bbox"]
            foot = ((x1 + x2) / 2, y2)
            p = (foot[0] - cam[fn][0], foot[1] - cam[fn][1]) if use_cam else foot
            if not P.contains_point(p):
                pos.setdefault(fn, {})[tid] = None; continue
            q = H @ np.array([p[0], p[1], 1.0])
            pos.setdefault(fn, {})[tid] = q[:2] / q[2]
    sp, tot = [], {}
    for fn in range(0, 750, 5):
        last = min(fn + 5, 749)
        for tid in pos.get(fn, {}):
            if tid not in pos.get(last, {}): continue
            a, c = pos[fn][tid], pos[last][tid]
            if a is None or c is None: continue
            d = float(np.hypot(*(a - c)))
            sp.append(d / ((last - fn) / 24) * 3.6)
            tot[tid] = tot.get(tid, 0) + d
    return np.array(sp), np.array(sorted(tot.values()))
```

---

## 3. Phase 0 — make it runnable and testable

### 0.1 Dependency pinning
Create `requirements.txt` with pinned versions (`ultralytics`, `supervision`, `opencv-python`,
`numpy`, `pandas`, `scikit-learn`, `matplotlib`) and a venv. The README lists deps in prose
only. Pin `supervision` explicitly — see 1.7.

### 0.2 Test harness
Create `tests/` with pytest. Minimum coverage, all stub- or synthetic-driven, no video needed:

- `test_camera.py` — **synthetic pure-translation test**: take any image, generate N frames
  each shifted by a known `(dx, dy)`, assert the estimator recovers cumulative drift to within
  1 px and that drift is monotonic for a monotonic pan. This is the single most valuable test
  in the repo; the current estimator fails it.
- `test_view_transformer.py` — each `pixel_vertices` corner maps to its `target_vertices`
  corner within 1e-6; a point outside the quad returns `None`; report coverage over stub
  positions (currently 67.1 %, see 1.2).
- `test_speed.py` — no speed exceeds 36 km/h; no per-track distance implies > 14 km/90 min.
- `test_possession.py` — percentages sum to 100; no `IndexError` when frame 0 has no
  assignment (see 1.6).
- `test_regression.py` — the Section 2 baseline numbers, so you can see every change you make.

### 0.3 Configuration and CLI
Everything is a literal in `main.py`. Add `argparse` (`--input`, `--output`, `--model`,
`--stub-dir`, `--no-stub`) and move the magic numbers (homography vertices, possession-overlay
box, feature mask, distance thresholds) into a config dataclass or YAML. Do not leave
resolution- or clip-specific constants in module bodies.

---

## 4. Phase 1 — correctness

Ordered by dependency: 1.1 feeds 1.2 feeds 1.3. Do not reorder.

### 1.1 Camera movement estimator — rewrite
`camera_movement_estimator/camera_movement_estimator.py:54-74`

Three distinct defects:

**(a) The estimate is the single largest-moving feature.**
```python
if distance > max_distance:
    max_distance = distance
    camera_movement_x, camera_movement_y = measure_xy_distance(...)
```
This is the most outlier-sensitive statistic available. One feature on a moving player or a
flapping hoarding becomes the camera motion. Evidence from the stub, frames 100–130:
`-17.4, -17.9, -16.6, -16.7, +19.0, -17.8, -19.4, +25.4, -20.5, -11.7, ...` — the sign flips
on consecutive frames. No real camera pans 40 px one way and back in 1/24 s.

**(b) Stale feature points.** `old_features` is refreshed only when motion exceeds the
threshold, but `old_gray` is refreshed every frame. From the second frame of any static
stretch onward, optical flow runs point coordinates belonging to one frame against an image
from a different frame.

**(c) The output is not cumulative, but is consumed as if it were.** Because of (b), each
value is drift since the last refresh. `add_adjust_positions_to_tracks` subtracts it from
positions as though it were cumulative drift from frame 0. The stub series is sawtooth: 421 of
750 frames are exactly `[0, 0]` and the last frame is `[0, 0]`, while the naive cumulative sum
would be 1617 px.

**Also:** the status flag from `calcOpticalFlowPyrLK` is discarded (`new_features, _, _ =`),
so failed tracks count as real motion.

Rewrite as:
- Track features frame n−1 → n, every frame, with freshly consistent `old_gray` / `old_features`.
- Keep only points where `status == 1`.
- Estimate the per-frame global motion robustly: `cv2.estimateAffinePartial2D(old_pts, new_pts,
  method=cv2.RANSAC, ransacReprojThreshold=3)` and take the translation component, or failing
  that the component-wise median of inlier displacements. Never a max.
- Re-detect features when the surviving count drops below ~50 % of `maxCorners`, or every
  ~30 frames, and reset the reference image at the same instant.
- Accumulate: `cumulative[n] = cumulative[n-1] + delta[n]`. Document in a docstring that the
  returned series is **cumulative displacement from frame 0, in pixels**.
- Drop the `minimum_distance` gate, or apply it to the robust delta, not to a max.
- Reconsider the mask. `mask_features[:, 0:20] = 1` and `[:, 900:1050] = 1` restrict features
  to two vertical strips, one of which sits mid-frame where players run. With RANSAC you no
  longer need a hand-tuned mask; prefer using the whole frame and letting RANSAC reject the
  players. If you keep a mask, mask *out* the pitch, not *in* two arbitrary columns.
- `camera_movement = [[0,0]]*len(frames)` builds a list of references to one object. Elements
  are replaced rather than mutated so there is no live bug, but replace it anyway.

**Acceptance:** passes the synthetic translation test in 0.2, and the camera model's
reprojection residual on static scene features stays under ~1 px.

> **Retracted criterion.** This section used to require that the *with-correction* speed
> column come out **tighter** than the without-correction column. That is wrong, and it was
> measured to be wrong once the sample clip was available. It holds only when the camera is
> a nuisance to be removed; this camera deliberately tracks the play, so uncorrected
> measurement is relative to a moving frame and reads too low, and correcting it legitimately
> *widens* the distribution. Judge the camera model by its reprojection residual, which is
> directly measurable, not by what it does to the speed histogram.
>
> The same measurement showed that a **translation is the wrong model entirely** for this
> footage — 29–65 px residual against a homography's 0.62 px. See `FIXES.md` item 10.

**Invalidates `stubs/camera_movement_stub.pkl`.** Delete it and regenerate, or version it (1.9).
The user must supply the video to regenerate it, so flag this and continue with the rest.

### 1.2 View transformer — coordinate space and coverage
`view_transformer/view_transformer.py:9-19`

Two problems. The first is fixed for free by 1.1: the quad is defined in raw frame-0 pixel
space, but `transform_point` is fed `position_adjusted`. Once camera movement is genuinely
cumulative-from-frame-0, `position_adjusted` is in frame-0 space and the two agree. Add a
docstring saying so, and a comment at the call site, because the coupling is invisible.

The second you cannot fully fix without the video: the four `pixel_vertices` were eyeballed
from one clip with no record of which pitch lines they correspond to, and `court_length =
23.32` is suspiciously precise for a hand-picked number. **67.1 % of player foot positions in
the stub fall inside the quad** — the other third get `position_transformed = None` and so no
speed and no distance, silently.

Do:
- Move the vertices into config (0.3), with the current values as the documented default for
  the sample clip, labelled as uncalibrated.
- Validate at construction: quad is convex, in consistent winding order, and non-degenerate.
  Raise on failure rather than producing a silent garbage homography.
- Make the `None` return visible — count and log the out-of-quad fraction rather than letting
  a third of the data vanish.
- Add an inverse-homography round-trip test.

**Do not invent new vertex values.** Re-deriving them needs the video and identifiable pitch
lines; say so in your report and leave the default in place. One piece of evidence the metre
scale is at least roughly right: the resulting median speed of 6.3 km/h and ~9 km/90 min
extrapolated distance are in the expected range for real football, so the scale is not off by
a large factor.

### 1.3 Frame rate and resolution — stop hardcoding
`utils/video_utils.py:15` writes at a fixed 24 fps; `speed_and_distance_estimator.py:9` sets
`frame_rate = 24`. The real value is available from `cap.get(cv2.CAP_PROP_FPS)` and is ignored,
so **every speed is wrong by the ratio of true fps to 24** on any other clip.

Same class of bug, resolution: the possession overlay at `(1350,850)-(1900,970)`
(`tracker.py:170`), the camera feature mask columns, and the overlay at `(0,0)-(500,100)`
(`camera_movement_estimator.py:89`) all assume 1920×1080.

Do: read fps and `(width, height)` once in `read_video`, return them alongside the frames,
thread them to `save_video` and `SpeedAndDistance_Estimator`, and express every overlay
rectangle as a fraction of frame size.

### 1.4 Speed and distance
`speed_and_distance_estimator/speed_and_distance_estimator.py`

- Distance accumulates per **track ID**, and the stub has **39 IDs for ~22 players** — roughly
  17 identity switches. Every switch starts a fresh 0 m counter, so distances are systematic
  undercounts and are not comparable between players. You cannot fully solve re-identification
  here; you *can* stop presenting the numbers as if it were solved. Report per-track distance
  with the track's frame span and a visible caveat, and consider merging tracks whose spans are
  disjoint and whose endpoints are close in transformed space.
- Distance is the sum of 5-frame chord lengths, so positional jitter inflates it monotonically.
  Smooth the transformed trajectory (small moving average or Savitzky–Golay) before
  differencing, and/or ignore segments below a noise floor.
- Segments where either endpoint is outside the quad are skipped entirely, which silently
  undercounts rather than marking the gap. Track and expose coverage per player.
- `draw_speed_and_distance` mutates `frames` in place and its return value is discarded at
  `main.py:81`. Correct today, broken the moment someone adds a `frame.copy()`. Make it
  consistent with the other `draw_*` methods and use its return value.

**Acceptance:** no sample above 36 km/h; no per-track distance implying > 14 km/90 min.

### 1.5 Ball detections and possession
`trackers/tracker.py:93-98`, `:28-38`, `main.py:58-70`

The ball is detected in **278 of 750 frames (37 %)**. `interpolate_ball_positions` linearly
fills the other 63 % and `assign_ball_to_player` then treats all 750 frames as equally good.
Nearly two-thirds of the headline possession percentage rests on straight-line guesses between
detections that can be dozens of frames apart.

Also: ball detections bypass ByteTrack and are hardcoded to `track_id = 1`; when a frame has
several ball detections the last one silently wins; `conf=0.1` is low enough to invite false
positives; and if the ball is never detected at all, every row is NaN and flows downstream
unflagged.

Do:
- Pick the **highest-confidence** ball detection per frame, not the last, and keep the
  confidence on the track.
- Mark interpolated frames with an `interpolated: True` flag and cap interpolation with a
  `limit` (suggest ~0.5 s of frames); beyond that, leave the gap empty.
- Exclude interpolated or gapped frames from the possession denominator, and render the
  coverage figure on the overlay so the possession % is read with its own reliability.
- Guard the all-NaN case explicitly.

### 1.6 First-frame crash
`main.py:69`
```python
else:
    team_ball_control.append(team_ball_control[-1])
```
`IndexError` if no player is within 70 px of the ball in frame 0. It survives on the sample
clip by luck. Seed with a sentinel (e.g. `0` for "unknown") and handle that value in
`draw_team_ball_control`, which must also not divide by zero when no frame has been assigned yet.

### 1.7 Team assignment, goalkeepers, and the hardcoded fudge
`team_assigner/team_assigner.py`

Delete this:
```python
if player_id == 91:
    team_id = 1
```
It is a patch for one misclassified goalkeeper in one clip (ID 91 is in the stub). It is the
symptom; the cause is `tracker.py:72` relabelling goalkeepers as players, after which they are
clustered by shirt colour, which they never match.

Do:
- Keep `goalkeeper` as its own class through tracking and annotation. Exclude goalkeepers from
  the 2-cluster kit fit, where they currently distort the centroids, and assign their team by
  pitch half or proximity to the nearer goal.
- `get_player_team` caches team by track ID forever, so one bad first crop locks a player's
  team for the whole clip. Accumulate colour samples over the first N appearances and take a
  running majority vote instead.
- `assign_team_color` fits team colours from **frame 0 only**, with `n_init=1` in
  `get_clustering_model` making the per-player clustering unstable. Fit over a sample of frames
  and raise `n_init`.
- The corner-pixel background heuristic assumes bbox corners are not the player. It fails on
  tight crops and overlapping players; at minimum, detect the degenerate case where all four
  corners agree with the centre and fall back.
- Replace positional tuple indexing into supervision detections (`frame_detection[0]`, `[3]`,
  `[4]` at `tracker.py:83-85`) with named attributes (`.xyxy`, `.class_id`, `.tracker_id`).
  The tuple gained a sixth element in later supervision releases; indices 0/3/4 happen to be
  stable today but this is version-coupled for no reason.

### 1.8 Memory
`utils/video_utils.py:3-11` decodes the whole clip into a list: 750 × 1920×1080×3 ≈ **4.6 GB**,
and `main.py` holds the input and output lists simultaneously — roughly **9 GB for 31 seconds**.
Convert to a streaming read → process → write loop, or a generator with a bounded buffer.
This is the one change that may require touching every `draw_*` signature; do it last within
the phase, and keep the annotation functions frame-at-a-time rather than list-at-a-time.

### 1.9 Stub integrity
`get_object_tracks` writes the stub whenever `stub_path` is set, and `main.py` passes
`read_from_stub=True` by default — so edits to detection or tracking silently do nothing until
someone notices. The stub also records nothing about which video or model produced it.

Do: store a small header alongside the pickle (video path, file hash or size, frame count,
model path, code version) and refuse to load a stub whose header does not match, with a clear
message. Default `read_from_stub` to `False` in the CLI. Note that `pickle.load` on a
repo-committed file is an arbitrary-code-execution path; prefer `.npz` or JSON for the camera
series, which is plain numbers.

---

## 5. Phase 2 — hygiene

- `.gitignore` reads `*__pycahce__*` — typo, so `__pycache__` is not actually ignored. Fix to
  `__pycache__/`.
- Remove `sys.path.append('../')` from `trackers/tracker.py:9`,
  `player_ball_assigner/player_ball_assigner.py:2`,
  `camera_movement_estimator/camera_movement_estimator.py:6` and
  `speed_and_distance_estimator/speed_and_distance_estimator.py:3`. It resolves against the cwd,
  not the file, so it does nothing useful and only works because `main.py` runs from the root.
- Rename typos in names that cross module boundaries: `draw_traingle` → `draw_triangle`,
  `persepctive_trasnformer` → `perspective_transformer`, `ouput_video_frames` → `output_...`,
  and the `sekf` parameter at `tracker.py:17`. Keep these in a commit of their own so they do
  not obscure the behavioural diffs.
- Output is XVID `.avi`, which most browsers will not play. Offer `mp4v`/H.264 `.mp4`.
- The Roboflow API key in `training/football_training_yolo_v5.ipynb` is marked revoked in a
  later commit but remains in git history. Leave the history alone; do not attempt a rewrite.
  Just confirm the notebook reads the key from an env var going forward.

---

## 6. Deliverables

1. Working `tests/` suite, green, runnable without the model or the video.
2. `requirements.txt` with pins.
3. The fixes above, as a series of focused commits (one concern per commit, behavioural changes
   separate from renames).
4. `FIXES.md` reporting: the Section 2 table re-measured after each of 1.1–1.4, what moved and
   why; the new out-of-quad coverage figure; the new ball-coverage figure; and an explicit list
   of anything you could not verify without the video or weights.

## 7. Acceptance checklist

- [ ] Synthetic pure-translation test recovers known camera drift within 1 px, monotonic.
- [ ] With-correction speed stats are **tighter** than without-correction on median, p95, max
      and the >36 km/h rate — the reversal of the Section 2 table.
- [ ] No speed sample above 36 km/h; no per-track distance implying > 14 km/90 min.
- [ ] Pipeline runs on a clip that is not 1920×1080 at 24 fps without any code edit.
- [ ] Possession never raises on frame 0 and reports its own ball-detection coverage.
- [ ] No hardcoded track IDs anywhere.
- [ ] Peak RSS for a 750-frame clip under 1 GB.
- [ ] Stub with a mismatched header is refused, not silently used.

## 8. Do not

- **Do not download the weights or the sample video yourself.** Ask the user; the README has
  the Drive links.
- **Do not invent replacement homography vertices.** Recalibration needs the video and
  identifiable pitch lines. Flag it, keep the documented default.
- Do not rewrite git history or attempt to purge the revoked API key.
- Do not swap the detector or tracker for something else, add a re-identification model, or
  introduce a new ML dependency. Fix what is here.
- Do not delete `stubs/track_stubs.pkl` — it is the only fixture. `camera_movement_stub.pkl`
  becomes invalid under 1.1 and may be removed once you have said so in `FIXES.md`.
- Do not report a fix as verified on the strength of the output video looking fine. The bugs
  here produce plausible-looking numbers; use the tests and the baseline table.
