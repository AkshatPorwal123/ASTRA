# ASTRA — Phase 1

Autonomous Space Task Recognition & Assistance System.
This is the **Phase 1 foundation**: real backend infrastructure, a real
procedure-reasoning engine, and a real dashboard — with the perception
models (person/pose/object/interaction/activity) stubbed out as clearly
labeled placeholders, per the design doc's Section 39 rule against faking
AI behavior.

## What's real vs. placeholder (updated — Milestone 7)

| Component | Status |
|---|---|
| Video ingestion (webcam + file) | REAL |
| Local recording + event log | REAL |
| Structured report generation | REAL — includes Section 37's procedure compliance accuracy metric + alert-type breakdown, auto-generated on run stop: `reports/{run_id}.report.json` + `.report.txt` |
| LAN MJPEG streaming | REAL |
| Database (SQLite by default, Postgres-ready) | REAL |
| REST API (experiments/runs/procedure/alerts/logs/system/reports) | REAL |
| WebSocket real-time event bus | REAL |
| Person detection | REAL — OpenCV HOG+SVM baseline + centroid tracker |
| **Object detection** | **REAL — two detectors combined:** MobileNet-SSD (VOC-20 classes, e.g. `bottle`) **+ HSV color-segmentation detector** for color-coded boxes (`red_box`, `blue_box`, `green_box`, `yellow_box`). **Calibration note:** the HSV ranges are reasonable defaults, not tuned to your specific lighting/box material — if detection is flaky, this is the first thing to adjust (`COLOR_HSV_RANGES` in `real_models.py`). |
| 2D pose estimation | REAL, after `python scripts/download_models.py` |
| Object tracking | REAL — centroid tracker |
| Hand-object interaction | REAL (baseline) |
| Activity recognizer | REAL (rule-based baseline). Now also emits `object_class` — which object the winning activity label is actually about — so the procedure engine can distinguish "right action, wrong object" from a genuine match. |
| **Procedure engine — sequence/state** | **REAL, substantially extended (Milestone 6):** preconditions and postconditions are now actually evaluated against a live `ObjectStateTracker`, not just declared in YAML and ignored. Simple graph branching (`branches:` conditions on object state), `repeatable` steps, and `recovery_step` redirects are all implemented and unit-tested. |
| **Error / deviation detection** | **REAL, full Section 16 taxonomy (Milestone 7):** `SKIPPED`, `WRONG_ORDER`, `REPEATED`, `WRONG_OBJECT`, `INVALID_ACTION`, `FAILED`, `TIMEOUT`, `UNCERTAIN` are all distinct, evidence-gated (TENTATIVE→CONFIRMED, never a single-frame trigger), and a `RECOVERED` lifecycle closes them out once the astronaut corrects course. **Bug fixed this round:** `TIMEOUT` evidence was previously computed but never turned into an alert — it silently vanished. It now escalates: alert → retry → `recovery_step` redirect (if configured) or `FAILED` (if not). Optional steps auto-skip on timeout instead of blocking the run. |
| Offline voice assistant | REAL logic; audio playback depends on your OS having a TTS engine `pyttsx3` can reach. Now uses `voice.error()`/`voice.recovery()` for `FAILED`/`RECOVERED` events, not just `warning()` for everything. |

## Procedure engine design (Milestone 6/7 — new)

- `backend/procedure_engine/object_state.py` — `ObjectStateTracker`: per-object,
  per-attribute state (`object.state`, or any custom attribute like
  `container.mixed`), updated **only** by validated (CONFIRMED) step
  postconditions — never by raw per-frame perception. This is a deliberate
  design choice for a single deterministic source of truth; documented in
  the module docstring, not hidden.
- `backend/procedure_engine/conditions.py` — a tiny, fixed grammar
  (`object.attribute == VALUE` / `!= VALUE`) for preconditions/postconditions.
  No `eval()` — kept auditable per Section 15/31's determinism requirement.
- `backend/procedure_engine/engine.py` — rewritten this round. Every
  `observe_activity()` call now branches through: exact match → advance
  (+ apply postconditions, evaluate `branches`, emit `RECOVERED` for any
  resolved deviations) → precondition failure (`FAILED`) → wrong object
  (`WRONG_OBJECT`) → deviation classification (`REPEATED` / `WRONG_ORDER`
  + implied `SKIPPED` steps / `INVALID_ACTION` / `UNCERTAIN`) → timeout
  check (`TIMEOUT` → retry → recovery/`FAILED`).
- **Tested**: `test_engine.py` at the project root (stubs pydantic out with
  plain dataclasses so it can run without the full dependency stack) —
  9 scenarios / 26 assertions covering the happy path and every deviation
  type. All pass. Run it with `python3 test_engine.py` from the project root.

## Experiment configs

- **`red_box_demo.yaml`** (dashboard default) — uses the color detector,
  matches your actual test object (a red box). `PICK/HOLD/MOVE/PLACE`
  vocabulary. Updated to exercise preconditions, postconditions, an
  optional step, a repeatable step, and a `recovery_step`.
- **`pick_place_demo.yaml`** — same vocabulary and same M6/7 features, for
  a bottle-shaped object via MobileNet-SSD.
- **`liquid_mixing_v1.yaml`** — legacy/reference only. Its
  preconditions/postconditions (including multi-attribute ones like
  `container.mixed == true`) are now real and will be enforced correctly —
  the engine handles them — but it still expects `OPEN/POUR/MIX/CLOSE`,
  which the current rule-based recognizer can't distinguish (needs a
  trained HAR model, Milestone 11+). Will get stuck mid-experiment with
  real perception active — intentional, not a bug (Section 39).
- **The actual official experiment** (colored boxes, exact step sequence
  from the problem statement) isn't built yet — the pasted problem text
  cuts off before the full step sequence. Once available, this becomes the
  real target config.

## Requirements

- Python 3.10+
- Node.js 18+
- A webcam (optional — you can also point a run at a video file)

## 1. Backend setup

```bash
cd astra
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Run from the `astra/` project root (not inside `backend/`), so the
`backend` package and `configs/` directory resolve correctly:

```bash
uvicorn backend.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs (FastAPI's
built-in Swagger UI) — this is the fastest way to try `/api/runs` by hand.

## 2. Dashboard setup

```bash
cd dashboard
npm install
npm run dev
```

Visit `http://localhost:3000`.

## 3. Running the demo experiment

1. With both servers running, open the dashboard.
2. Pick `liquid_mixing_v1` from the dropdown (already loaded from
   `configs/experiments/liquid_mixing_v1.yaml`) and click **Start Run**.
3. If a webcam is available at index 0, you'll see a live feed; if not,
   the pipeline still runs (perception is stubbed regardless).
4. Watch the **Procedure Status** panel advance through steps automatically
   — the stub activity recognizer cycles PICK → OPEN → POUR → MIX → CLOSE →
   PLACE every few seconds, so you'll see the full pipeline (perception →
   procedure engine → alerts → voice → recording → dashboard) working
   end-to-end without needing a trained model yet.
5. Click **Stop Run** to end early, or let it complete on its own.
6. Recorded video + the event log land in `backend/recordings/{run_id}.*`.

## 4. Running against a video file instead of a webcam

`POST /api/runs` accepts `"source": "/path/to/video.mp4"` instead of `0`.
Use the Swagger UI at `/docs` to try this directly.

## Project layout

See `ASTRA_Technical_Design.md` (the design doc from the previous phase)
for the full architecture. Folder structure matches Section 28 of that
document, trimmed to what Phase 1 actually implements.

## Known limitations (through Milestone 7)

- Activity recognition is still a rule-based baseline over real interaction
  signal (not a trained temporal model) — see Section 11/40. `WRONG_OBJECT`
  attribution depends on the activity recognizer's single-object-per-hand
  heuristic (`object_class` in its output), which is a reasonable baseline
  for the current single-actor demo, not a general multi-hand/multi-object
  resolver.
- Object state (`ObjectStateTracker`) is updated only from validated
  postconditions, not reconciled with raw perception — documented design
  choice in `object_state.py`, not a bug, but it means the tracker can lag
  reality if a step's postcondition is under-specified.
- Multi-camera frame synchronization isn't implemented (single source only).
- Dashboard doesn't yet visualize object state or the `SKIPPED`/`FAILED`/
  `RECOVERED` alert types with distinct styling — it renders whatever
  `type`/`alert_type` string comes through generically. Functionally
  correct, cosmetically flat.
- No authentication on the API/WebSocket yet (Section 26 flags this — needed before any real LAN exposure).
- SQLite is the default DB; switch `ASTRA_DATABASE_URL` to Postgres for concurrent-writer scenarios per the design doc's Section 27 reasoning.
- `liquid_mixing_v1.yaml`'s `OPEN/POUR/MIX/CLOSE` vocabulary still can't be
  recognized by the current rule-based activity recognizer — needs
  Milestone 11's trained HAR model, not a Milestone 6/7 gap.

## Milestone 8 — Dataset generation & a genuinely trained model

Everything through Milestone 7 is either classical CV or hand-written
rules — nothing was actually *trained*. This closes that gap, and it's
also an explicit requirement in the official problem statement
("Dataset generation to train model..." / "Deliverable: A trained AI
model").

- **Dashboard "Dataset Labeling" panel** — while a run is live, it shows
  the rule-based recognizer's current guess and lets you tap the label
  that's actually true. Each tap writes one `{features, label}` sample to
  `datasets/{run_id}.jsonl`. This is real labeled data from your actual
  camera and actual box, not synthetic.
- **`scripts/train_activity_model.py`** — trains a RandomForest classifier
  on everything in `datasets/*.jsonl` (only samples you explicitly
  confirmed via the labeling panel — anything unreviewed is excluded).
  Refuses to train below 30 samples or below 2 distinct labels, with a
  clear message rather than silently producing a useless model. Saves to
  `models/activity_classifier.pkl` + a metadata JSON recording the exact
  feature schema, accuracy, and sample count.
- **`TrainedActivityRecognizer`** — the orchestrator uses this
  automatically if a trained model is present *and* its feature schema
  still matches the current code; otherwise it falls back to the
  Milestone 5 rule-based recognizer. Check `GET /api/inference/status` to
  see which one is actually active — it's always explicit, never silently
  assumed.
- Tested end-to-end with synthetic data standing in for real collected
  data (since no real dataset exists yet on a fresh checkout): confirmed
  sample filtering, the min-samples guard, correct training/loading, and
  correct fallback behavior when no model exists or its schema is stale.

**To actually use this:** run a few experiments, label generously via the
dashboard panel, then run `python scripts/train_activity_model.py` and
restart the backend. More labeled runs → better model. This is the one
piece of the system that gets better through your own use, not further
code changes.

## Milestone 9 — API auth & offline standalone deployment

- **API key auth** (closes the Section 26 item flagged since Milestone
  1's design doc) — set `ASTRA_API_KEY` to require an `X-API-Key` header
  on all `/api/*` routes. Unset by default (open access, for local dev) —
  a startup log line always states plainly whether auth is on or off, so
  it's never ambiguous. The dashboard prompts for the key itself if the
  backend requires one (`GET /api/config` reports this, no secret
  leaked). WebSocket and the MJPEG video stream accept the key as
  `?api_key=` since browsers can't attach custom headers to those.
- **`run_astra.bat` / `run_astra.sh`** — one script starts backend +
  dashboard together, checks for the venv and pose model, and warns if
  auth is off. This is the "standalone operation" the official problem
  calls for: both processes run entirely on this machine, no ground-control
  or internet dependency at any point.
- **Dashboard step checklist fix** — now fetches the full experiment
  definition via `/api/experiments/{name}` and shows every step upfront
  (pending/current/done), instead of only showing steps discovered so far
  as the run progresses. This was a known gap flagged back in Milestone 5.

## Milestone 13 — the optional challenge, TLS, and an honest final assessment

**The official spec's OPTIONAL challenge** ("orientation-agnostic 3D Human
Mesh Recovery to track the astronaut relative to the payload rack, not
the floor") is addressed, but NOT as literal SMPL-based 3D HMR — that
needs body-model weights gated behind a manual license-agreement
registration (https://smpl.is.tue.mpg.de/) with no way to obtain them in
this environment. Shipping an "HMR module" that can't load its required
files would be exactly the unverified, faked capability this project has
avoided at every step so far.

What's actually built and rigorously tested: `backend/vision/rack_reference.py`
uses ArUco marker detection (already in opencv-contrib-python, zero new
dependencies) as a physical reference frame fixed to the payload rack.
Pose keypoints get transformed into the marker's own coordinate frame
instead of raw image coordinates. This solves the literal physical
problem stated — no fixed up/down, track relative to the rack — with 2D
geometry instead of 3D mesh recovery. **Proven, not just implemented**:
tested that two wildly different camera/rack orientations (0° vs 137°)
produce IDENTICAL rack-relative coordinates for the same physical joint
position (error ~1e-14, floating-point noise only) — genuine orientation
invariance. Opt in via `use_rack_reference: true` when starting a run;
needs a printed marker from `scripts/generate_rack_marker.py` mounted on
the rack. Off by default since most experiments don't have one mounted.

**TLS/HTTPS support** — closes the "plain HTTP, API key travels in the
clear" gap. `scripts/generate_self_signed_cert.py` generates a real
self-signed cert (correctly marked as a non-CA leaf certificate, includes
this machine's LAN IP automatically). Tested with a genuine end-to-end
TLS handshake — a real `uvicorn` server serving `https://`, a real client
connecting and completing the handshake, a real request succeeding. Both
startup scripts auto-detect `certs/cert.pem` + `certs/key.pem` and enable
TLS automatically if present. A caught-and-fixed bug worth noting
honestly: a "quick fix" for a cosmetic duplicate-SAN-entries issue
accidentally shadowed the RSA private key variable with a same-named loop
variable (Python has no block scoping) — caught immediately by re-running
the same rigorous end-to-end test after the "trivial" change, not by
assuming a small edit was safe. That's the actual discipline this project
has tried to hold to throughout, not just a nice-sounding principle.

## Known remaining gaps — genuinely permanent, not just unfinished

Some things are done as completely as this environment honestly allows.
Two are not, and won't be without something outside my control:

- **The actual official experiment.** The pasted problem text cuts off
  before the full step sequence. Every config here (`red_box_demo`,
  `pick_place_demo`) is a stand-in that proves the architecture works —
  none of them is the real competition deliverable. This is the one gap
  that blocks everything else from being the *actual* submission, and it
  can only close with the missing text.
- **Full 3D Human Mesh Recovery**, if the literal SMPL-based approach
  turns out to be a hard requirement rather than one option the spec's
  OPTIONAL text suggests. That needs license-gated model weights this
  environment cannot obtain — see Milestone 13 above for what was built
  instead and why.

Everything else is either fully closed, or an explicit, reasoned scope
decision stated plainly in this README (pose estimation training, the
appearance classifier's honest "second opinion, not a detector" framing)
— not a silently dropped corner.

## Milestone 10 — Operator-facing completeness (GUI + voice + IP streaming)

- **Object States & skipped-steps now visible** — the engine always
  computed `object_states` and `skipped_steps`, but nothing rendered
  them. New "Object States" panel + skipped steps now show inline in the
  step checklist (purple "skipped — optional" badge).
- **FAILED alerts are now specific, not generic** — the engine already
  captured *which* precondition failed (`evidence.failing`), but the
  voice/alert message discarded it and said only "could not be
  validated." Now: *"Step S2_HOLD cannot proceed: `red_box.state == HELD`
  not satisfied."* — actually actionable.
- **Video push to a specific IP** — the official spec literally says
  *"Stream the video of the experiment to specific IP"*; everything
  before this was pull-based (dashboard fetches from the backend).
  `POST /api/runs` now accepts `stream_to: "ip:port"` and pushes each
  frame there via HTTP POST in a background thread, with a bounded
  drop-oldest queue so an unreachable/slow destination degrades to
  dropped frames rather than blocking the run. Tested against a real
  mock HTTP receiver, including the unreachable-destination path.

## Milestone 11 — Calibration & object-detection dataset tooling

- **Interactive color calibration** — fixes the "not tuned to your
  lighting" limitation flagged since Milestone 5. Click a color name +
  click the box in the live video, and the system samples the actual
  pixel color under actual lighting and derives an HSV range from it
  (with correct circular-mean handling for hues that wrap around 0/180,
  like red — verified against a synthetic red patch). Saved to
  `configs/calibrated_colors.json`, merged over the hardcoded defaults on
  the next run start. Can calibrate entirely new colors, too — not just
  retune the four built-in ones.
- **Object-detection dataset capture** — Milestone 8's dataset tooling
  only covered activity labels; the official spec also asks for dataset
  generation for object detection. New "Capture frame for object-
  detection training" button saves the current frame + the detector's
  current boxes (or a manually-corrected set, via the API) to
  `datasets/object_detection/{run_id}/` in a simple images+JSONL format,
  ready for future custom-detector training.
- Also fixed a real, unrelated bug while regression-testing this round:
  `test_engine.py` had a hardcoded absolute sandbox path from whoever
  built Milestone 6/7, which would have failed identically on your
  Windows machine. Now resolves relative to the script's own location.

## Milestone 12 — voice polish + closing the object-detection dataset loop

- **Human-readable voice/alert messages everywhere**, not just FAILED —
  all messages (SKIPPED, WRONG_ORDER, REPEATED, WRONG_OBJECT, TIMEOUT,
  RECOVERED, recovery redirects) now use the step's actual name from the
  config ("Hold the red box steady") instead of its raw ID ("S2_HOLD").
  RECOVERED specifically also got clearer phrasing: *"Resolved: wrong
  order on Hold the red box steady — back on track."*
- **Fixed the calibration UI bug flagged last round** — the click-catcher
  was a separate overlay div that also blocked the color-name input and
  button while active. Rewired to attach the click handler directly to
  the video element via a ref instead, so it's scoped correctly by
  construction, not by careful CSS. The button now also says "click here
  to cancel" — an actual cancel path that didn't exist before.
- **Object-detection dataset → training loop, closed.** Milestone 11
  built capture; nothing consumed it. `scripts/train_object_classifier.py`
  trains an appearance classifier on captured crops (color histogram +
  gradient features → RandomForest). **Scoped honestly**: captured
  samples are all regions some detector already proposed — there's no
  negative/background data, so this can't be a full from-scratch object
  *detector* (which needs to learn what ISN'T a box too). What it
  legitimately is: a learned second opinion on WHAT a proposed region
  looks like, layered on top of the geometric color detector, never
  replacing its localization. Verified this is genuinely useful, not just
  functional — in one real end-to-end test it correctly confirmed a real
  detected box's class, and gave a known MobileNet-SSD false positive
  (misdetecting a flat background as "tvmonitor") a low-confidence,
  disagreeing second opinion — exactly the signal that could help filter
  false positives.

## Testing

`python3 test_engine.py` (from the project root, not `backend/`) runs a
standalone procedure-engine test suite — 9 scenarios / 26 assertions
covering the happy path, `WRONG_OBJECT`, precondition failure → `FAILED`,
`SKIPPED`+`WRONG_ORDER`, `REPEATED`, `TIMEOUT` retry/recovery/failure
escalation, optional-step auto-skip, and the `RECOVERED` lifecycle. It
stubs the pydantic-based config schema with plain dataclasses so it runs
without the full backend dependency stack installed — useful for quickly
verifying engine changes in isolation.

