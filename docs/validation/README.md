# Scientific validation

Corpus makes measurements that people will put in papers, so the agreement
between Corpus and a human reference has to be measurable, reproducible, and
stated with its limits. This directory holds the harness, the current results,
and the honest gaps.

## Status

| Study | Reference | Status |
|---|---|---|
| [Synthetic self-check](synthetic_selfcheck/report.md) | exact phantom geometry | **done** — runs in CI |
| Pilot validation (Story F2) | expert manual / ImageJ measurements | **not started — needs real TEM data** |

> The synthetic self-check tells you how faithfully the classical pipeline
> recovers a *known* object. It does **not** tell you how well Corpus agrees
> with a human expert on a real micrograph. Only the pilot can do that, and it
> requires real images with manual measurements that this repository does not
> contain. Do not cite the self-check as accuracy evidence.

## Running a validation

One command, from a validation table:

```bash
python -m corpus.validation --table my_table.csv --out docs/validation/pilot_2026_09
```

Outputs, all in `--out`:

| File | Contents |
|---|---|
| `report.md` | human-readable report with every table below |
| `report.json` | the same data, machine-readable |
| `metrics.csv` | one row per quantity per stratum, for spreadsheets |
| `paired_rows.csv` | the joined per-particle rows actually compared |
| `scatter_*.png` | Corpus vs reference, with the identity line |
| `residuals_*.png` | error distribution |
| `bland_altman_*.png` | agreement with limits of agreement |
| `error_vs_size_*.png` | error against particle size |

Useful flags:

```bash
python -m corpus.validation --print-schema        # the exact column names
python -m corpus.validation --fail-on-target ...  # non-zero exit if a target fails
python -m corpus.validation --reference ref.csv --corpus corpus.csv --out ...
python -m corpus.validation --stratify-by morphology,mode --out ...
```

## Table schema

One row per particle. Print the authoritative list with `--print-schema`.

| Column | Required | Meaning |
|---|---|---|
| `image_id` | yes | image the particle belongs to |
| `particle_id` | yes | stable identifier within the image |
| `reference_diameter_nm` / `corpus_diameter_nm` | one pair required | overall particle diameter |
| `reference_core_nm` / `corpus_core_nm` | optional | Au core diameter |
| `reference_outer_nm` / `corpus_outer_nm` | optional | SiO2 outer diameter |
| `reference_shell_nm` / `corpus_shell_nm` | optional | shell thickness |
| `morphology`, `mode` | optional | stratify metrics by these |
| `magnification`, `source` | optional | recorded, not analysed |
| `status` | optional | `ambiguous` / `excluded` / `unresolved` excludes the row **and reports it** |
| `notes` | optional | free text |

Blank cells mean "not measured". They are never read as zero.

## Metrics and what they are for

| Metric | Answers |
|---|---|
| MAE, MAE % | typical error size |
| RMSE | whether a few large errors dominate |
| Mean bias, bias % | whether Corpus systematically over- or under-reports |
| Mean relative error | error as a fraction of each particle's size |
| R² about the identity line | how much of the reference variance Corpus reproduces **without a free slope** |
| Pearson r | correlation, reported *alongside* R², never instead of it |
| Bland–Altman limits | the interval containing ~95% of differences |
| Recall / precision | particles missed, and particles invented |

R² is computed about `corpus = reference`, not about a fitted line. A constant
offset must show up as a poor R²; a free-slope fit would hide it. This is why
a perfectly correlated but biased result reports `r = 1.000` with `R² < 1`.

Rows that cannot be compared are **counted and listed by reason**, never
dropped. A validation that quietly discards Corpus's failures is not a
validation.

## Engineering targets

`corpus/validation/report.py` defines `PILOT_TARGETS`:

| Quantity | Max MAE | Max abs. bias |
|---|---|---|
| `diameter_nm` | 10% of reference mean | 5% |
| `core_nm` | 15% | 7.5% |
| `outer_nm` | 10% | 5% |
| `shell_nm` | 20% | 10% |

These are release gates for the engineering pipeline. They are not claims
about the accuracy of the underlying method, and they are not thresholds for
publication-quality metrology.

## Running the pilot (Story F2)

Still to do; it needs data this repository does not have.

1. Collect 10–20 TEM images, at least one Au@SiO2 core-shell case, spanning
   more than one magnification and contrast condition.
2. Have an expert measure ≥100 particles manually (ImageJ/Fiji, or the
   consensus of two raters on a smaller set). Record the protocol.
3. Measure the same images in Corpus. Record the exact preset, thresholds and
   calibration — `measurements.json` already stores all of it, including a
   `run_fingerprint`.
4. Match particles between the two sets, building the table above. Mark
   genuinely ambiguous particles `status=ambiguous`; do not delete them.
5. Run `python -m corpus.validation --table ... --out docs/validation/pilot_<date>`.
6. Commit the report directory, link it from the top-level README, and update
   the README's Limitations section with any failure mode the pilot exposes.

## Current findings from the synthetic self-check

Regenerate with `python scripts/build_synthetic_validation.py`.

- Overall diameter agreement is within ~1.3% MAE with full recall.
- Outer (carrier) diameters agree within ~1.4% MAE with full recall.
- Core recall is 0.75: the cores in the low-contrast phantom are **not
  recovered at all**. Corpus reports the carriers, flags every object for
  review, and emits one false positive on that image.
- Corpus reads slightly *small* on core and outer diameters (bias −6.4% and
  −1.4%), consistent with a threshold cutting inside a soft, blurred edge.

The low-contrast failure is a real limitation and is recorded in the top-level
README.
