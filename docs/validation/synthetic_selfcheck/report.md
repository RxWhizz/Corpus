# Corpus validation report

- Generated: `2026-08-28T16:16:23+00:00`
- Corpus version: `1.0.0`
- Reference: **synthetic ground truth (phantom generator)**
- Images: 6 | rows: 21

These are engineering agreement targets for the release gate, not a
scientific accuracy claim about the method.

## Overall agreement

| Quantity | n | Ref mean | Corpus mean | MAE | MAE % | RMSE | Bias | Bias % | R² (identity) | r | Recall | Target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `diameter_nm` | 20 | 140.70 | 139.48 | 1.88 | 1.3% | 2.44 | -1.22 | -0.9% | 0.999 | 1.000 | 1.000 | PASS |
| `core_nm` | 9 | 54.78 | 51.27 | 3.51 | 6.4% | 3.58 | -3.51 | -6.4% | 0.975 | 1.000 | 0.750 | PASS |
| `outer_nm` | 12 | 128.33 | 126.50 | 1.83 | 1.4% | 2.48 | -1.83 | -1.4% | 0.997 | 0.999 | 1.000 | PASS |
| `shell_nm` | 9 | 38.83 | 40.03 | 1.20 | 3.1% | 1.43 | 1.20 | 3.1% | 0.993 | 0.999 | 0.750 | PASS |

## Detection counts

| Quantity | Reference | Corpus | Matched | Missed | Extra | Recall | Precision |
|---|---|---|---|---|---|---|---|
| `diameter_nm` | 20 | 21 | 20 | 0 | 1 | 1.000 | 0.952 |
| `core_nm` | 12 | 9 | 9 | 3 | 0 | 0.750 | 1.000 |
| `outer_nm` | 12 | 12 | 12 | 0 | 0 | 1.000 | 1.000 |
| `shell_nm` | 12 | 9 | 9 | 3 | 0 | 0.750 | 1.000 |

## Excluded and unmatched rows

Rows that could not be compared. They are listed, never dropped silently.

- `diameter_nm`: no_reference_value × 1
- `core_nm`: corpus_missed_particle × 3
- `shell_nm`: corpus_missed_particle × 3

## Breakdown by `morphology`

| Value | Quantity | n | MAE | MAE % | Bias | Recall |
|---|---|---|---|---|---|---|
| core_shell_sphere | `diameter_nm` | 12 | 1.83 | 1.4% | -1.83 | 1.000 |
| core_shell_sphere | `core_nm` | 9 | 3.51 | 6.4% | -3.51 | 0.750 |
| core_shell_sphere | `outer_nm` | 12 | 1.83 | 1.4% | -1.83 | 1.000 |
| core_shell_sphere | `shell_nm` | 9 | 1.20 | 3.1% | 1.20 | 0.750 |
| decorated | `diameter_nm` | 2 | 0.00 | 0.0% | 0.00 | 1.000 |
| decorated | `core_nm` | 0 | n/a | n/a | n/a | n/a |
| decorated | `outer_nm` | 0 | n/a | n/a | n/a | n/a |
| decorated | `shell_nm` | 0 | n/a | n/a | n/a | n/a |
| rod | `diameter_nm` | 3 | 2.21 | 0.8% | 2.21 | 1.000 |
| rod | `core_nm` | 0 | n/a | n/a | n/a | n/a |
| rod | `outer_nm` | 0 | n/a | n/a | n/a | n/a |
| rod | `shell_nm` | 0 | n/a | n/a | n/a | n/a |
| touching_spheres | `diameter_nm` | 3 | 3.00 | 6.9% | -3.00 | 1.000 |
| touching_spheres | `core_nm` | 0 | n/a | n/a | n/a | n/a |
| touching_spheres | `outer_nm` | 0 | n/a | n/a | n/a | n/a |
| touching_spheres | `shell_nm` | 0 | n/a | n/a | n/a | n/a |

## Figures

![scatter_diameter_nm.png](scatter_diameter_nm.png)
![residuals_diameter_nm.png](residuals_diameter_nm.png)
![bland_altman_diameter_nm.png](bland_altman_diameter_nm.png)
![error_vs_size_diameter_nm.png](error_vs_size_diameter_nm.png)
![scatter_core_nm.png](scatter_core_nm.png)
![residuals_core_nm.png](residuals_core_nm.png)
![bland_altman_core_nm.png](bland_altman_core_nm.png)
![error_vs_size_core_nm.png](error_vs_size_core_nm.png)
![scatter_outer_nm.png](scatter_outer_nm.png)
![residuals_outer_nm.png](residuals_outer_nm.png)
![bland_altman_outer_nm.png](bland_altman_outer_nm.png)
![error_vs_size_outer_nm.png](error_vs_size_outer_nm.png)
![scatter_shell_nm.png](scatter_shell_nm.png)
![residuals_shell_nm.png](residuals_shell_nm.png)
![bland_altman_shell_nm.png](bland_altman_shell_nm.png)
![error_vs_size_shell_nm.png](error_vs_size_shell_nm.png)

## How to reproduce

```bash
python -m corpus.validation --table <validation_table.csv> --out <output_dir>
```
