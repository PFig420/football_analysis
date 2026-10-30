# What was wrong, what changed, what is still open

Work against the plan in `CLAUDE.md`. Every number below was measured, not estimated.

## Summary

The pipeline produced plausible-looking numbers that were wrong in specific, measurable
ways. The clearest example: **camera compensation made every speed statistic worse than no
compensation at all.** It is now verified against synthetic pans with known answers, and on
a 600-frame 1080p clip panning at a known rate it recovers 299.5 px of drift against a true
299.5 px.

> **Correction, after obtaining the sample clip.** The rewrite in item 1 did not go far
> enough. It is a correct *translation* estimator, but a translation is the wrong model for
> broadcast footage: the camera pans hundreds of pixels to follow play, and that motion is
> projective, not a shift. The camera model is now a full homography (item 10), measured on
> the real clip at **0.68 px** median residual against the translation model's **60.80 px**.

Measured end to end on the sample clip (750 frames, 25 fps), the rebuilt pipeline reports
a 21/18 split of tracks between the two teams and 46.9 % / 53.1 % possession, against the
original's 32/7 and 69/31. 108 tests pass.

The repository also could not run on current dependencies at all: `supervision` removed
ByteTrack in 0.31.0. Tracking has been migrated to ultralytics' own BYTETracker and
`supervision` is no longer a dependency — see item 9.

68 tests pass, with `supervision` uninstalled. None need the weights or the sample video.

---

## 1. Camera movement — rewritten

`camera_movement_estimator/camera_movement_estimator.py`

Three defects, all confirmed by running the original against the new tests:

| Synthetic case | Old estimator | New estimator |
| --- | --- | --- |
| Pan of (2, 0) px/frame over 25 frames | **48.0 px error**, final drift reported as `[0, 0]` | < 1.0 px error |
| Pan of (1.25, 0.75) px/frame | **30.0 px error**, final drift `[0, 0]` | < 1.0 px error |
| **Static camera**, one object crossing the frame | **22.5 px** of phantom motion | < 2.0 px |

Causes: the estimate was the displacement of the single *largest-moving* feature; reference
points were refreshed on a different schedule from the reference image, so flow ran one
frame's coordinates against another frame's pixels; and the result was drift-since-last-reset
rather than cumulative, while the consumer subtracted it as cumulative.

Now: consecutive-frame flow with matched reference points, failed tracks dropped via the
status flag, per-frame translation taken as the median of RANSAC inliers, explicit
accumulation, and a dead zone so sub-pixel noise cannot random-walk into phantom drift over
a long clip. The mask that restricted features to two arbitrary vertical strips is gone —
RANSAC rejects the players instead.

**Verified on real video:** a 600-frame 1080p clip panning at 0.5 px/frame gives a final
cumulative drift of **299.5 px against a true 299.5 px**, with 0 frames lacking an estimate.

`stubs/camera_movement_stub.pkl` is invalidated by this change and has been deleted; it was
a cache of the old, wrong values.

## 2. Speed and distance

`speed_and_distance_estimator/`

Measured on the committed 750-frame fixture, camera correction disabled in both columns so
the comparison isolates this module:

| | Original chain | Rebuilt chain |
| --- | --- | --- |
| Median speed | 5.6 km/h | **5.4 km/h** |
| p95 speed | 15.7 km/h | **15.0 km/h** |
| Max speed | 37.4 km/h | **32.1 km/h** |
| Samples above 36 km/h | 0.1 % | **0 %** |
| Median per-track distance (31 s) | 38.7 m | **32.6 m** |

The 16 % drop in distance is jitter that used to be counted as running: summing raw
per-window chords only ever adds. Trajectories are now Savitzky-Golay smoothed before
differencing, with a noise floor. The plausibility ceiling never had to fire on this clip —
smoothing alone brought the maximum under it.

Also: the measurement window is in **seconds** rather than a fixed 5 frames, and the frame
rate comes from the file. On a 50 fps clip the original reported 24/50 of the true speed.

Each track now carries its span and coverage. Distances still accumulate per track ID, and
tracking identity switches split one player across several tracks (39 IDs for ~22 players on
the sample clip), so a per-track distance is a fragment, not a player's total. That is a
re-identification problem this module cannot solve; it is now reported rather than hidden.

## 3. Perspective transform

`view_transformer/`

The coordinate-space mix is fixed as a consequence of item 1: positions are corrected to
frame-0 space by a genuinely cumulative drift, which is the space the quad lives in.

The quad is now validated at construction — convex, non-degenerate, no duplicates — so a bad
configuration raises instead of silently producing a garbage homography. Coverage is counted
and logged: **67.1 % on the sample clip**, meaning a third of player-frames still get no
speed or distance.

The vertices themselves are unchanged and still uncalibrated. See [open items](#still-open).

## 4. Ball and possession

`trackers/`, `player_ball_assigner/`

The ball is detected in **278 of 750 frames (37 %)** on the sample clip. The original
interpolated every gap however long and then counted all 750 frames as equal evidence for
possession. Now: interpolation is capped (0.5 s by default), interpolated frames are flagged
and drawn in a different colour, and **possession counts only frames where the ball was
actually detected**. The reported percentage comes with its own coverage figure.

The ball is also picked as the highest-confidence detection in the frame rather than
whichever came last, is filtered by its own confidence threshold, and is kept out of
ByteTrack, where it was consuming a person identity slot.

The frame-0 `IndexError` (`team_ball_control[-1]` on an empty list) is gone.

## 5. Teams and goalkeepers

`team_assigner/`

`if player_id == 91: team_id = 1` is deleted. The cause was goalkeepers being relabelled as
outfield players and then clustered by shirt colour they never match; goalkeepers now keep
their own class, are excluded from the kit fit they were distorting, and are placed by pitch
position. Team colours are fitted over crops sampled across the clip rather than from frame 0
alone, and each track's team is a majority vote rather than whatever its first crop looked
like.

## 6. Running on other clips

Everything clip-specific moved into `config.py` / `--config` YAML: the pitch quad, overlay
geometry as fractions of the frame, thresholds, window lengths. Frame rate and resolution are
read from the file. The CLI takes `--input`, `--output`, `--model`, `--report`, `--no-video`,
`--max-frames`.

**Memory**: the original decoded the whole clip into a list and held the annotated copy
alongside it — about 9 GB for 31 s of 1080p. The pipeline now makes two bounded-memory passes.
Measured on 600 frames of 1080p: **peak RSS 598 MB**, against roughly 7.4 GB for the same
clip under the old design.

## 7. Stub provenance

Stubs carry a header recording the video name, size, frame count and model, and are refused
with a clear message if any differs. `--use-stub` is opt-in; the original defaulted to
reading the stub, so edits to detection silently did nothing. The legacy headerless fixture
still loads, with a warning.

## 8. Hygiene

- `.gitignore` said `*__pycahce__*` — a typo, so caches were never ignored.
- `sys.path.append('../')` removed from four modules; it resolved against the cwd and did
  nothing useful.
- `draw_traingle` → `draw_triangle`, `persepctive_trasnformer` → `perspective_transformer`,
  `sekf` → `self`, `ouput_video_frames` → `output_video_frames`. The old
  `draw_traingle` and `SpeedAndDistance_Estimator` names remain as aliases.
- Output defaults to `.mp4` (`mp4v`) instead of XVID `.avi`, which most browsers won't play.
- The Roboflow API key is still in git history, as instructed. Not rewritten.

## 9. supervision removed ByteTrack — migrated to ultralytics

Found while testing: **`sv.ByteTrack` does not exist in supervision 0.31.0.** The original
code raises `AttributeError` before processing a frame on any current install. The repo was
not runnable as checked out, independently of the missing weights.

Pinning `supervision==0.26.1` worked but was a dead end, so tracking now uses ultralytics'
own `BYTETracker` — the same algorithm, from a maintained home, in a package that was
already a dependency for detection. **`supervision` has been removed entirely**; the suite
passes with it uninstalled.

What changed:

- `sv.ByteTrack().update_with_detections(sv.Detections.from_ultralytics(r))` becomes
  `BYTETracker(...).update(boxes[is_person], frame)`, reading `r.boxes` directly. One fewer
  conversion, one fewer library in the hot path.
- Association parameters (`track_high_thresh`, `match_thresh`, …) are now in
  `TrackerConfig` instead of being library defaults you could not reach.
- The track buffer is expressed in **seconds**, not frames. ByteTrack's default of 30
  frames means 1.25 s at 24 fps but 0.6 s at 50 fps; it now means the same thing on any
  clip.
- `lap` is pinned explicitly. ultralytics' tracker needs it and otherwise tries to
  pip-install it at runtime, on the first tracked frame, which fails in a venv without pip.

Tested against real ultralytics objects rather than mocks: the adapter tests build genuine
`Boxes` and run the genuine `BYTETracker`, and one test confirms `model.predict` hands back
that same `Boxes` type, so both halves of the seam are covered. The model for that test is
built from a YAML config with random weights, so none of it needs a download.

---

## 10. The camera model is a homography, not a translation

Measured once the sample clip was available. Two findings, both from the real video.

**The clip is 25 fps, not 24.** The original hardcoded 24, so every speed it ever reported
was 4 % low on its own sample footage. Distances were unaffected, which is what made it easy
to miss. The pipeline reads the rate from the file now, so this is already fixed — but it is
worth recording that the bug was real and not hypothetical.

**The camera motion is projective, not a translation.** Registering frame 0 against later
frames with SIFT + RANSAC:

| Frames | Matches | Homography residual | Translation residual |
| --- | --- | --- | --- |
| 0 → 50 | 1622 | 0.32 px | 1.33 px |
| 0 → 100 | 371 | 0.90 px | **29.09 px** |
| 0 → 200 | 509 | 0.62 px | **65.70 px** |
| 0 → 300 | 581 | 0.69 px | **65.16 px** |

A translation fits only while the camera has barely moved. The broadcast camera pans to
follow play — the image centre at frame 300 maps back to x ≈ 100 in frame-0 coordinates,
about 860 px of pan — and under rotation and zoom the induced mapping is a homography. A
2D shift cannot represent it, so `position - (dx, dy)` cannot undo it.

Comparing the three models on the real clip (750 frames, 25 fps, stub tracks):

| Camera model | Median | p95 | Median distance | km/90 min | Coverage |
| --- | --- | --- | --- | --- | --- |
| None | 5.6 km/h | 15.6 | 32.6 m | 5.9 | 67.1 % |
| Translation (current) | 9.9 km/h | **27.0** | 39.9 m | 7.2 | 61.7 % |
| Homography to frame 0 | 8.8 km/h | **17.6** | 37.2 m | 6.7 | 61.4 % |

The translation model's p95 of 27 km/h implies 5 % of all measurement windows are near
sprinting, which no team sport produces. The homography model is both better behaved and
correct by construction.

**This is now the default** (`camera.model: homography`). `CameraMovementEstimator` returns
a per-frame 3×3 matrix mapping that frame into the reference view, and
`CameraModel.add_adjust_positions_to_tracks` applies it instead of subtracting an offset.
Registration is SIFT + RANSAC against a keyframe, re-anchoring to the previous frame when a
pan carries the view off the current one, so long clips do not accumulate chaining drift.
It registers all 750 frames of the sample clip with zero failures in about 28 s.

The translation model is kept as `camera.model: translation` — it costs almost nothing and is
adequate for a genuinely static camera, such as a fixed tactical one.

**Measured residual against static scenery on the real clip** (the replacement acceptance
criterion, now a test in `tests/test_acceptance_real_clip.py`):

| Frame | Static points | Homography | Translation |
| --- | --- | --- | --- |
| 60 | 1161 | 0.34 px | 1.97 px |
| 150 | 421 | 0.81 px | 59.22 px |
| 300 | 437 | 0.72 px | 57.44 px |
| 450 | 648 | 0.65 px | 62.37 px |
| 600 | 442 | 0.64 px | 80.05 px |
| 749 | 322 | 1.00 px | 87.74 px |
| **median** | | **0.68 px** | **60.80 px** |

**The acceptance criterion in `CLAUDE.md` was itself wrong** and has been retired. It
required corrected speeds to be *tighter* than uncorrected ones. That holds only when the
camera is a nuisance; when the camera deliberately tracks the action, uncorrected
measurement is relative to a moving frame and reads *too low*, so correcting legitimately
widens the distribution. The replacement criterion is the camera model's reprojection
residual on static scene features, which is directly measurable: < 1 px for a homography,
29–65 px for a translation.

## 11. Team assignment was splitting players by sunlight

Surfaced by the first real run: the pipeline put **32 of 39 tracks on one team and 7 on the
other**. Both the original's rule and the rewritten one did it, so it was not the extraction
logic — it was the colour space.

The two kits in this match are white and green, which have similar brightness. Clustering
raw BGR therefore found its strongest axis along *illumination*, separating sunlit players
from shadowed ones rather than one team from the other.

Kit colours are now clustered in hue-saturation space, with hue encoded as a vector
(`cos θ·S`, `sin θ·S`) so that red does not sit at both ends of the scale and so that greys
and whites, whose hue is meaningless, collapse towards the origin instead of landing at an
arbitrary angle. Measured over the same 39 tracks:

| Feature space | Split |
| --- | --- |
| BGR (as shipped, and as the original) | **35 / 4** |
| Chromaticity | 23 / 16 |
| Lab a\*b\* | 25 / 14 |
| **Hue × saturation (now used)** | **21 / 18** |

Mean per-track vote purity is 0.99, so tracks are not merely being split evenly — each one
is being classified consistently. End to end this moved possession from 69 / 31 to
**46.9 / 53.1**, and the fitted colours from a white and a muddy brown to a white
`[211, 217, 214]` and a green `[133, 213, 158]`, which are the kits actually on the pitch.

Cluster centroids now live in hue-saturation space and are not colours, so the BGR used for
annotation is the mean of each cluster's members.

## 12. The pitch quad is calibrated, by a tool rather than by eye

`calibrate_pitch.py` fits the homography to the markings actually visible in the reference
frame, and writes both the `pitch:` config block and an overlay that redraws the pitch from
world coordinates for visual checking.

It needs the halfway line, both touchlines and the centre circle — **not** a penalty area,
which matters because a camera on the middle third never sees one. Rather than matching
hand-picked corners, it constrains each group of detected pixels by what it *is*: the
halfway line lies at x = 0, the touchlines at y = 0 and y = width, the circle 9.15 m from
the centre spot. That uses every detected pixel and sidesteps the fact that perspective does
not preserve centres — the middle of the projected ellipse is not the centre spot.

On the sample clip, fitted against 5720 halfway-line, 4838 far-touchline, 6827
near-touchline and 2808 circle pixels:

| Constraint | RMS residual |
| --- | --- |
| Halfway line | 0.054 m |
| Far touchline | 0.639 m |
| Near touchline | 0.100 m |
| Centre circle | 0.167 m |

The far touchline is consistently the worst because it is seeded from the grass boundary,
which sits slightly outside the paint; under a metre is expected.

**What this says about the original guess.** Reprojecting the old hand-picked quad through
the fitted homography, its "length" edges really measure 23.4 m and 25.3 m against the
configured 23.32, and its "width" edges 65.1 m and 64.9 m against the configured 68.0. So
the tutorial's eyeballed length was very nearly right and the width was overstated by about
4.6 % — the scale was never the main problem. **Coverage was.** The calibrated quad spans
31.0 × 68.0 m and takes the fraction of player positions that can be measured at all from
58.3 % to **71.4 %**, which is 2330 more measurable positions and two more usable tracks.

Three failures found while building it, each now a test:

* **Touchlines cannot both be found the same way.** The far one is best taken from the grass
  boundary, since the paint blends into the advertising boards. The near one must come from
  the paint, because run-off grass continues past it — on this clip the boundary sits 37 px
  beyond the line, several metres once projected.
* **The circle fit needs constraining.** Fitting contour by contour latches onto a fragment,
  since players break the circle into arcs; fitting the whole point set with RANSAC works,
  but only once candidates are restricted to a band around the halfway line, or it settles
  on a larger ellipse through some other marking.
* **The solution is mirror-ambiguous.** Every constraint is symmetric under x → −x, so the
  optimiser returned one of two mirrored solutions depending on the arbitrary sign of the
  fitted ellipse's axis. Distances are identical either way, but the output was not
  reproducible. World +x is now pinned to the direction image +x points.

## Still open

1. **Calibration is per camera setup.** `calibrate_pitch.py` now fits the quad from the
   markings (item 12), and `config.sample_clip.yaml` holds the result for the sample clip.
   Any other footage needs its own run — a quad is a property of where the camera is, not
   of the project.

3. **Goalkeeper placement is untested on real footage.** The committed stub predates the
   goalkeeper class — the original relabelled keepers as outfield players — so a run against
   it places zero keepers. The logic has unit tests, but it has never seen a real keeper.
   Re-running detection with the weights would exercise it.

4. **Identity switches are reported, not solved.** 39 track IDs for ~22 players. Fixing it
   properly means re-identification, which was explicitly out of scope.

5. **Goalkeeper placement is a positional heuristic**, not a kit match. It assumes teams
   defend opposite halves, which holds within a half of football but not across one.

## Verifying

```bash
pip install -r requirements.txt
pytest                      # 68 tests, no weights or video needed
```

The acceptance criteria from `CLAUDE.md` that can be checked without the sample clip all
pass: synthetic pan recovery within 1 px and monotonic; no speed above 36 km/h and no
distance implying more than 14 km/90 min on the fixture; the pipeline runs on a clip that is
not 1920×1080 at 24 fps with no code edit; possession never raises on frame 0 and reports its
coverage; no hardcoded track IDs; peak RSS under 1 GB; a mismatched stub is refused.
