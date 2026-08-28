# Changelog

All notable changes to Corpus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Corpus uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Still open before tagging v1.0.0

- Capture the three README screenshots (see `docs/screenshots.md`).
- Run the pilot validation against expert manual/ImageJ measurements on real
  TEM images (Story F2, see `docs/validation/README.md`).
- Apply the GitHub repository description and topics:
  `bash scripts/apply_github_metadata.sh`.
- Replace the placeholder maintainer email in `package.json` before publishing
  `.deb`/`.rpm` packages (see `AUTHORS.md`).

## [1.0.0] - unreleased

First release aimed at reproducibility and reviewability rather than new
measurement features.

### Added

- **`corpus` package** — the scientific core, importable and testable without
  Electron: `calibration`, `segmentation`, `measurement`, `metrology`,
  `review`, `io`, `validation`, `dev`.
- **Test suite** — 364 tests over synthetic fixtures, with no dependency on
  private TEM images. Covers calibration, particle measurement, core-shell
  metrology, watershed, filters, metadata schema, COCO export, the dataset
  pipeline, the segmentation backend contract, and end-to-end reproducibility.
- **Run provenance** — `measurements.json` now records `corpus_version`,
  `segmentation_backend`, and a `run_fingerprint` binding the image checksum to
  the measurement settings. Two runs with the same fingerprint are asserted to
  produce identical measurements.
- **Demo dataset** — `Examples/demo_dataset/`: six CC0 synthetic TEM phantoms
  with exact ground truth, COCO masks, provenance metadata, and versioned
  expected results for regression testing. Regenerates byte-for-byte from a
  fixed seed.
- **Validation harness** — `python -m corpus.validation` compares Corpus
  against a manual/ImageJ reference in one command, emitting CSV, JSON,
  Markdown and publication-ready figures with MAE, RMSE, bias, relative error,
  R² about the identity line, Pearson r, Bland–Altman limits, and explicit
  recall/precision. Uncomparable rows are counted and listed, never dropped.
- **Synthetic self-check** — a validation run against the demo phantoms'
  ground truth, committed under `docs/validation/synthetic_selfcheck/`.
- **Segmentation backend contract** — `SegmentationBackend` /
  `SegmentationResult` with `ClassicalBackend` and `ManualBackend`. Every
  measurement records which backend produced it. No ML runtime is imported by
  the classical path, and the tests assert it.
- **Continuous integration** — GitHub Actions running tests, lint, Python and
  Node syntax checks, demo-dataset integrity, and the smoke test on Linux and
  Windows, plus an Electron build smoke.
- **Release automation** — tag-triggered workflow building the Windows
  installer/ZIP and the Linux AppImage, with checksums, gated on the test
  suite and on the tag matching `package.json`.
- **Canonical smoke test** — `python -m corpus.dev.smoke` exercises
  measurement, demo integrity, the demo baseline, the dataset pipeline, split
  determinism, validation, and the backend contract.
- **Attribution** — `AUTHORS.md` distinguishing original author, current
  maintainer, third-party components, the derivative-work position, and data
  provenance.

### Changed

- **Repository metadata** — `package.json` description, homepage, repository
  URL, bug tracker, keywords and maintainer now point at `RxWhizz/Corpus`. All
  `Corpus-New` references and hard-coded local development paths are gone.
- **README** — rewritten around what a new visitor needs in the first minute,
  with the classical and AI subsystems clearly separated and limitations stated
  plainly.
- **`measurement_modes.py`** — reduced from ~1070 to ~520 lines by moving the
  scientific logic into `corpus/`. It keeps the same CLI contract and
  re-exports every name it previously defined, so existing imports still work.
  Verified byte-identical output across all four presets before and after.
- **Dataset manifest** — now records `file_sha256`, `annotation_review`,
  `skipped_review_labels` and `calibration_state` per image.
- **Dataset audit** — missing licence, source and checksum are surfaced for
  every image rather than only for the public demo layer; a missing manifest is
  now an error; a public-demo row without an accepted licence now fails instead
  of warning.
- **`measurements.json`** — written with sorted keys, so repeated runs produce
  a byte-identical file.

### Fixed

- **Scale-bar detection crashed on OpenCV 5.** `cv2.HoughLinesP` changed its
  return shape from `(N, 1, 4)` to `(N, 4)`; Corpus now handles both. Before
  this fix, automatic calibration raised an unhandled `IndexError` on any image
  where the Hough detector found a line.
- **Synthetic dataset generation crashed on small canvases.** A fine nm/px on a
  small frame produced a particle larger than the image and raised an opaque
  `low >= high` from the placement draw. Particle size is now clamped to the
  frame, and a genuinely unusable canvas raises a clear message.

[Unreleased]: https://github.com/RxWhizz/Corpus/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/RxWhizz/Corpus/releases/tag/v1.0.0
