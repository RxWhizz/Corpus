# Screenshots to capture

The README shows three screenshots. They cannot be produced in CI — Corpus is
an Electron app and needs a display — so they have to be captured by hand once.

## How to capture

```bash
PYTHON=python3 npm run start
```

Load `Examples/demo_dataset/images/demo_001_core_shell_spheres.png`, scale
length `100` nm, preset **Core-shell spheres**, mark the scale line across the
white bar at the bottom right, then **Process Image**.

Save each capture into `docs/images/` with the exact filename below, then
replace the placeholder row in the README table with the image links.

| Filename | What to show |
|---|---|
| `docs/images/measurement-workspace.png` | The main workspace after **Process Image**: the loaded micrograph with the detection overlay, the preset and scale controls visible on the left. |
| `docs/images/review-basket.png` | The Measurement Basket with several rows, at least one flagged `needs_review`, showing that a human accepts or rejects each detection. |
| `docs/images/distribution-summary.png` | The histogram and summary panel with counts, mean, standard deviation and the Gaussian reference curve. |

## Guidelines

- Capture the window only, not the whole desktop.
- Use the demo dataset, never private TEM data — these images are published.
- Target roughly 1400 px wide; PNG.
- Keep the light theme so the screenshots stay readable on GitHub in both
  light and dark mode.
- A short GIF of `Load → Calibrate → Measure → Review` is a good addition to the
  hero section, but the three stills come first.

## After capturing

Replace this block in `README.md`:

```markdown
| Measurement workspace | Review basket | Distribution summary |
|---|---|---|
| _screenshot pending_ | _screenshot pending_ | _screenshot pending_ |
```

with:

```markdown
| Measurement workspace | Review basket | Distribution summary |
|---|---|---|
| ![Measurement workspace](docs/images/measurement-workspace.png) | ![Review basket](docs/images/review-basket.png) | ![Distribution summary](docs/images/distribution-summary.png) |
```

and delete the "Screenshots are still to be captured" note above it.
