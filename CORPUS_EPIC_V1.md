# EPIC — Corpus v1.0: Scientific Validation, Productization & Open-Source Readiness

## Epic Goal

Transform **Corpus** from a functional TEM nanoparticle analysis prototype into a **scientifically credible, reproducible, installable, and portfolio-ready open-source application**.

The objective is **not** to add many new features. The objective is to consolidate the existing functionality so that a researcher can:

1. Understand what Corpus does in less than one minute.
2. Install it without reproducing the development environment manually.
3. Run it on a small demo dataset.
4. Trust that the core measurement pipeline is tested.
5. Compare Corpus measurements against a manual/reference workflow.
6. Reproduce dataset preparation and training exports.
7. Verify that the project is actively maintained through CI and releases.

---

# Success Definition

Corpus v1.0 is considered complete when:

- [ ] The GitHub repository has correct description, topics, URLs, authorship, and package metadata.
- [ ] README installation commands refer only to `RxWhizz/Corpus`.
- [ ] README includes screenshots or a short demo GIF.
- [ ] A small redistributable TEM demo dataset is included or downloadable.
- [ ] Core measurement functions have automated tests.
- [ ] GitHub Actions validates Python, Node/Electron syntax, tests, and dataset smoke checks.
- [ ] A Windows executable/installer and Linux AppImage are attached to a GitHub Release.
- [ ] Corpus can reproduce the same measurement output from the same image/configuration.
- [ ] A scientific validation workflow exists for comparing Corpus against manual/ImageJ measurements.
- [ ] At least one pilot validation dataset has been evaluated and summarized.
- [ ] The repository clearly distinguishes classical CV measurement, AI dataset preparation, and future AI segmentation.
- [ ] Large legacy modules have begun migration into clearer internal modules without breaking the current GUI.

---

# Non-Goals

The following are explicitly outside this Epic unless required to complete v1.0:

- Replacing ImageJ/Fiji.
- Building a full annotation platform.
- Training a large production AI model.
- Supporting every nanoparticle morphology.
- Rewriting the Electron frontend.
- Converting the entire project into a web application.
- Adding cloud infrastructure.
- Building a new segmentation architecture.
- Supporting autonomous measurement without human review.

---

# Workstream A — Repository & Branding Cleanup

## Story A1 — Canonical repository metadata

### Tasks

- Replace all references to:
  - `Corpus-New`
  - old clone URLs
  - obsolete local development paths
- Update `package.json`:
  - `homepage`
  - `repository.url`
  - package description
  - maintainer metadata
- Preserve upstream attribution where legally/technically required.
- Clearly distinguish:
  - original/upstream authors
  - current Corpus maintainer
  - derivative work attribution
- Add GitHub repository description.
- Add GitHub topics.

### Recommended description

> TEM nanoparticle metrology, dataset curation and instance-segmentation toolkit built with Electron, OpenCV and Python.

### Recommended topics

- `electron-microscopy`
- `tem`
- `nanoparticles`
- `computer-vision`
- `image-analysis`
- `materials-science`
- `opencv`
- `instance-segmentation`
- `scientific-software`
- `yolo`

### Acceptance Criteria

- [ ] `grep -R "Corpus-New" .` produces no unintended references.
- [ ] README clone instructions work from a fresh machine.
- [ ] GitHub metadata describes the actual project.
- [ ] Attribution is preserved and understandable.

---

## Story A2 — README redesign

### Required README structure

1. Hero section
2. One-sentence purpose
3. Screenshot/GIF
4. Key capabilities
5. Installation
6. Quickstart
7. Scientific workflow
8. Demo dataset
9. Outputs
10. Validation
11. AI dataset workflow
12. Limitations
13. Citation
14. License / attribution

### Hero example

```text
Corpus

TEM nanoparticle metrology and dataset curation.

Load → Calibrate → Measure → Review → Export
```

### Acceptance Criteria

- [ ] A new visitor can understand Corpus in under 60 seconds.
- [ ] At least 3 real UI screenshots are included.
- [ ] Classical CV and AI functionality are clearly separated.
- [ ] Quickstart requires no knowledge of internal project structure.

---

# Workstream B — Internal Architecture Cleanup

## Story B1 — Separate scientific logic from GUI glue

Move gradually from large root-level scripts into a clear internal package.

### Target structure

```text
Corpus/
├── app/
│   ├── main/
│   └── renderer/
│
├── corpus/
│   ├── calibration/
│   ├── segmentation/
│   ├── measurement/
│   ├── metrology/
│   ├── review/
│   └── io/
│
├── dataset/
├── training/
├── tests/
├── docs/
└── examples/
```

### Migration Rule

Do **not** rewrite working modules all at once.

For each migrated function:

1. Move pure logic.
2. Preserve current interface through a compatibility wrapper.
3. Add tests.
4. Remove wrapper only after stable use.

### Priority migration targets

1. scale calibration
2. particle filtering
3. watershed logic
4. particle measurements
5. core-shell calculations
6. metadata/export
7. `measurement_modes.py`
8. portions of `render.js` that only orchestrate backend calls

### Acceptance Criteria

- [ ] Core numerical/image functions can run without Electron.
- [ ] GUI code does not contain scientific formulas that should live in Python modules.
- [ ] At least three core domains are importable and independently testable.

---

# Workstream C — Automated Testing

## Story C1 — Build a real test suite

Create:

```text
tests/
├── fixtures/
├── test_scale_calibration.py
├── test_particle_measurement.py
├── test_core_shell.py
├── test_watershed.py
├── test_filters.py
├── test_metadata_schema.py
├── test_coco_export.py
└── test_reproducibility.py
```

### Required fixture types

- synthetic single particle
- two touching particles
- circular core-shell object
- elongated particle
- image with scale bar
- edge-cut particle
- low-contrast image
- small COCO annotation fixture

### Key invariants to test

#### Calibration

Given a known pixel length and scale-bar value:

```text
nm_per_pixel = scale_nm / scale_pixels
```

must be stable and deterministic.

#### Core-shell metrology

```text
t_shell = (D_total - D_core) / 2
```

must hold within numeric tolerance.

#### Reproducibility

Same:

- image
- calibration
- thresholds
- filters
- preset

must generate the same measurements.

### Acceptance Criteria

- [ ] `pytest` completes successfully.
- [ ] At least 25 meaningful tests.
- [ ] Core metrology is covered by deterministic tests.
- [ ] Tests do not depend on private TEM images.

---

# Workstream D — Continuous Integration

## Story D1 — GitHub Actions

Create a CI workflow for pushes and pull requests.

### Linux job

- checkout
- Python 3.11
- install dependencies
- `pytest`
- `ruff` or lightweight linting
- Node install
- `node --check main.js`
- `node --check render.js`
- dataset synthetic smoke test
- optional Electron build smoke test

### Windows job

- Python 3.11
- `npm ci`
- Python tests
- Node checks
- optional NSIS/ZIP build smoke test

### Dataset smoke test

Run the current synthetic training workflow:

```bash
python training/generate_synthetic_core_shell.py --count 5
python training/prepare_yolo_seg.py ...
python training/audit_training_dataset.py ...
```

or expose one canonical command:

```bash
python -m corpus.dev.smoke
```

### Acceptance Criteria

- [ ] Pull requests show CI status.
- [ ] A broken measurement test blocks CI.
- [ ] A broken COCO/YOLO conversion blocks CI.
- [ ] README displays build/test badge.

---

# Workstream E — Demo Dataset & Reproducibility

## Story E1 — Public demo dataset

Create a redistributable subset:

```text
examples/demo_dataset/
├── images/
├── metadata.csv
├── annotations/
└── README.md
```

Target:

- 5–10 images
- legally redistributable
- clear provenance
- scale metadata
- at least 2 morphology/contrast conditions if possible

### Required functionality

A user should be able to:

1. clone Corpus
2. install dependencies
3. load a demo image
4. calibrate or use provided scale metadata
5. run measurement
6. compare their output against expected example results

### Acceptance Criteria

- [ ] Every demo asset has redistribution status documented.
- [ ] Demo output can be regenerated.
- [ ] Expected output is versioned for regression testing.

---

# Workstream F — Scientific Validation

## Story F1 — Manual-reference benchmark framework

Build a validation tool that compares Corpus measurements against a reference set.

Recommended reference:

- expert manual measurement
- ImageJ/Fiji
- or consensus of 2 manual raters for a smaller validation set

### Validation table schema

```text
image_id
particle_id
reference_diameter_nm
corpus_diameter_nm
reference_core_nm
corpus_core_nm
reference_outer_nm
corpus_outer_nm
reference_shell_nm
corpus_shell_nm
```

### Metrics

For each relevant quantity:

- MAE
- RMSE
- mean bias
- relative error
- R²
- count agreement
- optional Bland–Altman limits of agreement

### Required plots

- Corpus vs Reference scatter
- residual/error distribution
- Bland–Altman plot
- error vs particle size
- optional error vs magnification/source

### Acceptance Criteria

- [ ] Validation can run from one command.
- [ ] Outputs CSV + JSON + publication-ready figures.
- [ ] Metrics are reported independently for each morphology/mode.
- [ ] Failed/ambiguous detections are not silently removed.

---

## Story F2 — Pilot validation dataset

Minimum pilot target:

- 10–20 TEM images
- 100+ manually reviewed particles if feasible
- at least one Au@SiO2/core-shell case
- multiple image conditions

### Initial target thresholds

These are engineering targets, not scientific claims:

- mean diameter MAE ≤ 10% of reference mean
- no severe systematic bias
- count recall reported explicitly
- ambiguous cases manually flagged

### Acceptance Criteria

- [ ] Pilot report committed under `docs/validation/`.
- [ ] README links to the validation report.
- [ ] Corpus limitations are updated based on observed failure modes.

---

# Workstream G — Packaging & Releases

## Story G1 — Release automation

Use existing Electron Builder configuration to create release artifacts.

### Required v1.0 artifacts

Windows:

```text
Corpus-1.0.0-Setup.exe
Corpus-1.0.0-win-x64.zip
```

Linux:

```text
Corpus-1.0.0.AppImage
```

Optional:

```text
Corpus-1.0.0.dmg
```

### Release contents

- binaries
- changelog
- known limitations
- demo instructions
- checksums

### Acceptance Criteria

- [ ] Fresh user can run Corpus without manually cloning source.
- [ ] Release corresponds to a Git tag.
- [ ] Version shown in app matches Git tag.

---

# Workstream H — AI Dataset Pipeline Hardening

## Story H1 — Make the dataset pipeline reproducible

The current COCO → YOLO-seg → audit → Colab flow should be treated as a first-class subsystem.

### Required guarantees

- COCO remains master annotation format.
- YOLO export is derived.
- source-group leakage is prevented.
- tiles from one micrograph cannot cross train/val/test.
- dataset manifest stores:
  - source
  - checksum
  - license
  - split
  - class counts
  - calibration state
  - annotation review state

### Acceptance Criteria

- [ ] Re-running the same export with the same seed produces identical splits.
- [ ] Leakage audit fails when a source appears across incompatible splits.
- [ ] Missing license/provenance is surfaced explicitly.
- [ ] Training bundle contains dataset manifest and audit report.

---

# Workstream I — Hybrid AI Readiness

## Story I1 — Define the future model interface without requiring production AI

Do not make AI mandatory for v1.0.

Create only the stable interface:

```python
class SegmentationBackend:
    def predict(image) -> SegmentationResult:
        ...
```

Possible implementations later:

```text
ClassicalBackend
YOLOSegBackend
ManualBackend
HybridBackend
```

### Hybrid philosophy

```text
AI proposes
↓
Corpus displays
↓
Researcher reviews
↓
Final accepted measurement
```

### Acceptance Criteria

- [ ] Classical measurement continues to work with no ML dependencies.
- [ ] Future AI model can be integrated without modifying the core measurement contract.
- [ ] Every segmentation output records its method/backend.

---

# Recommended Implementation Order

## Sprint 1 — Presentation & cleanup

- A1 repository metadata
- A2 README
- screenshots
- fix old URLs
- attribution cleanup

Estimated output:

> Corpus looks professional even before deeper refactoring.

---

## Sprint 2 — Tests & reproducibility

- C1 test suite
- E1 demo dataset
- D1 CI

Estimated output:

> Corpus becomes trustworthy software instead of only a working application.

---

## Sprint 3 — Packaging

- G1 Windows/Linux releases
- versioning
- changelog

Estimated output:

> Researchers can actually install and try it.

---

## Sprint 4 — Scientific validation

- F1 benchmark framework
- F2 pilot benchmark

Estimated output:

> Corpus becomes defensible as scientific software.

---

## Sprint 5 — Internal cleanup

- B1 modular migration
- H1 dataset pipeline hardening

Estimated output:

> Long-term maintainability without breaking current functionality.

---

## Sprint 6 — Future AI

- I1 segmentation backend contract
- real dataset growth
- optional trained model integration

Estimated output:

> Corpus becomes hybrid classical-CV + AI while preserving human review.

---

# Definition of Done — Corpus v1.0

Corpus v1.0 is DONE when the following experience works:

```text
Researcher discovers GitHub
        ↓
Understands Corpus from README/GIF
        ↓
Downloads Windows installer or AppImage
        ↓
Opens included demo TEM image
        ↓
Calibrates / loads scale
        ↓
Runs particle measurement
        ↓
Reviews detections
        ↓
Exports traceable measurements
        ↓
Reads validation report against manual/ImageJ reference
```

And for developers:

```text
git clone
↓
install
↓
pytest
↓
CI green
↓
demo smoke test
↓
build
```

---

# Priority Summary

| Priority | Item | Impact |
|---|---|---|
| P0 | Repository metadata + README cleanup | Very High |
| P0 | Real automated test suite | Very High |
| P0 | Demo dataset | Very High |
| P0 | CI | Very High |
| P1 | Scientific validation framework | Critical scientifically |
| P1 | Pilot validation | Critical scientifically |
| P1 | Windows/Linux release artifacts | High adoption impact |
| P1 | Deterministic dataset pipeline | High |
| P2 | Modular internal package | Medium/High |
| P2 | Hybrid segmentation interface | Future-facing |
| P3 | Production AI segmentation model | Defer until data are ready |

---

# Final Product Positioning

Corpus should position itself as:

> **A reproducible TEM nanoparticle metrology and dataset-curation workstation that combines classical image analysis, explicit human review, traceable metadata, and AI-ready annotation workflows.**

The competitive advantage is not “more automation”.

It is:

> **measurement + review + provenance + reproducibility + dataset readiness in one focused TEM workflow.**
