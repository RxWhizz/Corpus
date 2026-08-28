# EPIC — Corpus AI v1: Dataset Discovery, Acquisition, Training & Integration

## Epic Goal

Build the first reproducible AI segmentation model for **Corpus** by:

1. discovering suitable electron-microscopy datasets,
2. auditing licenses and provenance before download/use,
3. downloading and normalizing approved datasets,
4. pretraining a general nanoparticle segmentation model,
5. adapting it toward gold nanoparticle TEM,
6. fine-tuning on exact Au@SiO2 core-shell annotations when available,
7. evaluating it against held-out data and classical Corpus measurements,
8. exposing the trained model behind a stable Corpus segmentation backend.

The goal is **not** to claim universal TEM segmentation.

The goal is to produce a documented and reproducible model that improves particle/core-shell proposal quality while preserving **human review as the final authority**.

---

# Scientific Strategy

Corpus should not wait for a perfect public Au@SiO2 dataset.

Use staged transfer learning:

```text
General EM nanoparticles
        ↓
EMPS pretraining
        ↓
Gold nanoparticle TEM adaptation
        ↓
Au@SiO2 exact fine-tuning
        ↓
Corpus hybrid segmentation
        ↓
Human review
        ↓
Accepted metrology
```

This separates three different learning problems:

### Stage 1 — Particle morphology

Learn:

- nanoparticle/background separation,
- agglomeration,
- EM texture,
- particle boundaries,
- microscopy noise.

### Stage 2 — Gold TEM domain

Learn:

- Au contrast,
- TEM-specific edge characteristics,
- low-contrast conditions,
- realistic gold particle morphology.

### Stage 3 — Core-shell semantics

Learn the exact Corpus ontology:

```text
0: Au_core
1: SiO2_outer
```

where `SiO2_outer` means the full outer particle contour used to calculate total diameter.

---

# Candidate Dataset Stack

## Dataset A — EMPS

**Role:** General nanoparticle segmentation pretraining.

Known properties:

- electron microscopy nanoparticle images,
- 465 images,
- pixel-level instance segmentation,
- one general `particle` class,
- suitable for learning generic EM particle boundaries.

Expected Corpus layer:

```text
real_near_emps
```

### Use

Do not treat EMPS labels as Au@SiO2 truth.

Use only for:

```text
particle/background pretraining
        ↓
feature initialization
        ↓
fine-tune later
```

### Gate

- [ ] Verify repository/dataset license and citation.
- [ ] Save source URL, DOI, commit/tag if applicable.
- [ ] Store checksum for downloaded archive/repository snapshot.
- [ ] Record all terms in dataset manifest.

---

## Dataset B — PSDI Gold Nanoparticle TEM Dataset (2026)

**Role:** Domain adaptation toward gold nanoparticle TEM.

The public record describes:

- synthetic TEM images of gold nanoparticles,
- training and test splits,
- COCO segmentation annotations,
- experimental TEM validation images,
- manual annotations for the experimental validation data.

Approximate size:

```text
~1.3 GB
```

Potential mapping:

```text
synthetic_gold
experimental_gold_validation
```

This is particularly useful because it creates a bridge:

```text
EMPS generic EM particles
        ↓
gold TEM
        ↓
Au@SiO2
```

### Critical License Gate

The dataset record is marked open, but the visible metadata currently does not show an explicit license value.

Therefore:

> **DO NOT automatically redistribute or commit this dataset into Corpus until explicit reuse rights are confirmed.**

Allowed workflow before confirmation:

- metadata discovery,
- manifest entry,
- checksum planning,
- downloader disabled with `license_status=needs_review`.

After confirmation:

- download into ignored `data/external/`,
- use according to the confirmed terms,
- document attribution.

---

## Dataset C — Agglomerated / Non-Spherical EM Particle Dataset

**Role:** Robustness augmentation.

Candidate source:

- electron microscopy images,
- segmentation masks,
- TiO2/non-spherical/agglomerated particles,
- associated with a published segmentation workflow.

Use only if license/provenance gate passes.

Purpose:

```text
robustness to:
- touching particles
- irregular morphology
- agglomeration
- partial visibility
```

Do not mix this dataset into the final Au@SiO2 test set.

---

## Dataset D — Corpus Exact Au@SiO2

**Role:** Final domain truth.

This is the most important dataset.

Target ontology:

```text
Au_core
SiO2_outer
```

Recommended initial target:

```text
50–100 images/tiles      minimum useful fine-tune
150–250 images/tiles     pilot
300–600 images/tiles     strong v0
```

The exact number of images matters less than:

- diversity,
- independent source groups,
- multiple magnifications,
- multiple shell thicknesses,
- low/high contrast,
- agglomerated and isolated cases,
- careful masks.

---

# Workstream A — Dataset Discovery Engine

## Story A1 — Build a dataset registry

Create:

```text
training/datasets/
├── registry.yaml
├── licenses/
└── README.md
```

Recommended registry schema:

```yaml
datasets:
  - id: emps
    name: Electron Microscopy Particle Segmentation
    source_url: ...
    doi: ...
    task: instance_segmentation
    domain: electron_microscopy
    material_scope: generic_nanoparticles
    license: ...
    license_status: verified
    redistribution: ...
    citation: ...
    checksum: ...
    corpus_layer: real_near_emps
    enabled: true
```

Required fields:

- ID
- name
- URL
- DOI
- publisher/repository
- material/domain
- number of images
- annotation type
- ontology
- license
- license verification status
- redistribution permission
- citation
- checksum
- intended Corpus use
- download status

### Acceptance Criteria

- [ ] Every external training source appears in the registry.
- [ ] No dataset can be automatically downloaded if `license_status != verified`.
- [ ] Every model artifact can identify which registered datasets were used.

---

## Story A2 — Search additional datasets

Search:

- Zenodo
- Figshare
- DataCite-indexed repositories
- institutional research-data repositories
- GitHub only when dataset provenance is clear
- microscopy/software project repositories
- paper supplementary datasets

Preferred queries:

```text
TEM nanoparticle segmentation dataset
electron microscopy nanoparticle instance segmentation
gold nanoparticle TEM segmentation dataset
Au nanoparticle TEM annotations
Au SiO2 TEM dataset
core shell nanoparticle TEM segmentation
silica coated gold TEM dataset
electron microscopy particle masks
```

### Search Output

Generate:

```text
reports/dataset_discovery.csv
reports/dataset_discovery.md
```

Columns:

```text
dataset
url
doi
material
modality
images
annotation_type
license
license_status
exact_AuSiO2
training_value
decision
reason
```

### Acceptance Criteria

- [ ] At least 10 plausible sources inspected.
- [ ] At least 3 approved/usable sources if available.
- [ ] Exact Au@SiO2 availability explicitly reported.
- [ ] Rejected sources retain rejection reason.

---

# Workstream B — Safe Dataset Downloader

## Story B1 — Manifest-driven downloader

Create:

```text
training/download_datasets.py
```

Example:

```bash
python training/download_datasets.py --dataset emps
python training/download_datasets.py --all-approved
```

Downloader requirements:

- only approved datasets,
- retries,
- resumable where practical,
- checksum verification,
- no silent overwrite,
- provenance sidecar,
- source version/date,
- ignored external-data directory.

Target:

```text
data/external/<dataset_id>/
```

Each dataset gets:

```text
SOURCE.json
CHECKSUMS.txt
LICENSE.txt
README.source.md
```

### Acceptance Criteria

- [ ] Downloader refuses `needs_review` datasets.
- [ ] Corrupted downloads fail checksum.
- [ ] Downloaded files are ignored by Git unless explicitly redistributable demo assets.
- [ ] Provenance survives conversion into training datasets.

---

# Workstream C — Dataset Normalization

## Story C1 — Convert every approved source into Corpus master format

Corpus master annotation format remains:

```text
COCO instance segmentation
```

Create adapters:

```text
training/adapters/
├── emps.py
├── psdi_gold.py
├── generic_masks.py
└── corpus_native.py
```

Normalized output:

```text
data/normalized/<dataset_id>/
├── images/
├── annotations.json
├── manifest.csv
└── provenance.json
```

### Canonical intermediate ontology

For general pretraining:

```text
particle
```

For exact Corpus fine-tuning:

```text
Au_core
SiO2_outer
```

Never fake `Au_core` or `SiO2_outer` labels from a generic `particle` mask.

### Acceptance Criteria

- [ ] All normalized masks visually match images.
- [ ] COCO audit passes.
- [ ] Original IDs/source IDs remain traceable.
- [ ] Conversion is deterministic.

---

# Workstream D — Leakage-Proof Splitting

## Story D1 — Group-aware splits

Never split random tiles from the same parent micrograph across:

```text
train
val
test
```

Grouping keys in descending priority:

1. original micrograph ID
2. publication/source figure ID
3. dataset source group
4. acquisition session if known

Create:

```text
training/build_splits.py
training/audit_split_leakage.py
```

### Acceptance Criteria

- [ ] No group appears in more than one split.
- [ ] Duplicate or near-duplicate images are flagged.
- [ ] Test data remains locked during model selection.
- [ ] Exact Au@SiO2 test set contains only domain-exact samples.

---

# Workstream E — Baseline Training

## Story E1 — Establish a general segmentation baseline

Default first baseline:

```text
YOLO-seg
```

Do not optimize architecture prematurely.

Train on:

```text
EMPS
```

or:

```text
EMPS + approved robustness datasets
```

Output:

```text
runs/pretrain_em/
├── config.yaml
├── dataset_manifest.json
├── metrics.json
├── weights/
└── report.md
```

Track:

- mask mAP
- precision
- recall
- IoU/Dice if available
- object count error
- inference time
- model size

### Acceptance Criteria

- [ ] Training is reproducible from one command.
- [ ] Seed/config/environment are recorded.
- [ ] Best and last checkpoints retained.
- [ ] Evaluation uses held-out source groups.

---

# Workstream F — Gold TEM Domain Adaptation

## Story F1 — Fine-tune general model on gold nanoparticle TEM

Starting checkpoint:

```text
EMPS pretrained model
```

Fine-tune using approved gold TEM dataset.

Goal:

```text
generic particle representations
        ↓
gold-specific morphology/contrast
```

Evaluate separately on:

```text
experimental gold validation data
```

if licensing and intended use permit.

### Compare

```text
Model 0 — random initialization
Model 1 — EMPS pretrain
Model 2 — EMPS → Gold adaptation
```

### Acceptance Criteria

- [ ] Gold adaptation is compared against direct training.
- [ ] Improvement or lack of improvement is documented.
- [ ] Experimental validation is not mixed into training.

---

# Workstream G — Exact Au@SiO2 Fine-Tuning

## Story G1 — Prepare exact Corpus annotations

Use existing Corpus → CVAT → COCO workflow.

Labels:

```text
Au_core
SiO2_outer
```

Quality rules:

- clear polygon boundaries,
- paired core/outer IDs where possible,
- edge-cut particles flagged,
- ambiguous cores flagged,
- severe overlaps flagged,
- low-contrast examples retained with review status.

Recommended metadata:

```text
source_id
micrograph_id
magnification
scale_nm_per_px
particle_quality
contrast_class
aggregation_class
reviewer
review_status
```

### Acceptance Criteria

- [ ] Fine-tune set passes annotation audit.
- [ ] Test set is locked before training.
- [ ] Low-contrast samples exist in validation/test.
- [ ] No synthetic image appears in exact test.

---

## Story G2 — Fine-tune the two-class model

Training ladder:

```text
Experiment A
random → Au@SiO2

Experiment B
EMPS → Au@SiO2

Experiment C
EMPS → Gold → Au@SiO2
```

This ablation determines whether external pretraining actually helps.

### Primary question

> Does staged domain adaptation reduce the amount of exact Au@SiO2 annotation needed?

Run learning-curve experiments with approximately:

```text
10%
25%
50%
100%
```

of the available exact training data.

### Acceptance Criteria

- [ ] At least three initialization strategies compared.
- [ ] Learning curve generated.
- [ ] External pretraining retained only if it provides measurable benefit.

---

# Workstream H — Metrology-Aware Evaluation

## Story H1 — Evaluate scientific quantities, not only segmentation metrics

Segmentation accuracy alone is insufficient.

For accepted/matched particles calculate:

```text
D_core
D_total
t_shell = (D_total - D_core) / 2
```

Compare predicted masks against manual/reference annotations.

Metrics:

- core diameter MAE
- total diameter MAE
- shell thickness MAE
- relative error
- count precision/recall
- false positive rate
- false negative rate
- failed-core detection rate
- low-contrast performance

### Required subgroup analysis

Report results by:

```text
contrast
particle size
shell thickness
aggregation
source
magnification
```

### Acceptance Criteria

- [ ] Model report includes metrology errors.
- [ ] Low-contrast failure rate is explicitly reported.
- [ ] No failed segmentation is silently discarded.

---

# Workstream I — Classical vs AI vs Hybrid Benchmark

## Story I1 — Compare Corpus modes

Evaluate the same locked test set with:

```text
Classical Corpus
AI-only
Hybrid AI + human review
Manual reference
```

Measure:

- segmentation quality
- diameter error
- shell error
- count error
- time per image
- interventions required
- fraction of automatically acceptable particles

### Key product metric

A useful target is not:

> "AI beats all manual analysis."

It is:

> "AI-assisted Corpus reduces researcher effort while preserving measurement quality."

### Acceptance Criteria

- [ ] Benchmark produces CSV + JSON + figures.
- [ ] Human review time is measurable.
- [ ] Corpus README reports only validated claims.

---

# Workstream J — Model Packaging

## Story J1 — Export production model artifact

Each released model should include:

```text
model.onnx / model.pt
model_card.md
training_config.yaml
dataset_manifest.json
metrics.json
checksums.txt
```

Model card must state:

- intended use
- training domains
- exact ontology
- known limitations
- low-contrast behavior
- unsupported morphologies
- dataset citations
- license
- version

Recommended naming:

```text
corpus-seg-au-sio2-v0.1.0
```

### Acceptance Criteria

- [ ] Model version is independent from application version.
- [ ] Corpus can verify model checksum.
- [ ] Model provenance is fully reconstructable.

---

# Workstream K — Corpus Integration

## Story K1 — Implement segmentation backend

Use the interface planned in the Corpus v1 EPIC:

```python
class SegmentationBackend:
    def predict(self, image) -> SegmentationResult:
        ...
```

Implement:

```text
ClassicalBackend
AIBackend
HybridBackend
```

`HybridBackend`:

```text
AI predicts
    ↓
Corpus overlays masks
    ↓
researcher accepts/rejects/edits
    ↓
accepted measurements enter basket
```

Every result stores:

```text
backend
model_version
confidence
review_status
timestamp
```

### Acceptance Criteria

- [ ] Corpus works with no AI dependencies.
- [ ] AI backend is optional.
- [ ] Hybrid output is traceable.
- [ ] Human-rejected objects cannot silently enter final metrology.

---

# Workstream L — Reproducible Training Command

## Story L1 — One-command experiment runner

Create:

```bash
python training/train_corpus_seg.py \
  --config configs/training/au_sio2_v1.yaml
```

The command should:

1. load registry,
2. verify datasets,
3. verify split integrity,
4. generate/locate normalized data,
5. train,
6. evaluate,
7. calculate metrology metrics,
8. save model card,
9. save manifest,
10. save report.

### Acceptance Criteria

- [ ] Training can be reproduced from a clean environment.
- [ ] No hidden local paths.
- [ ] Missing datasets produce actionable messages.
- [ ] Exact command/config is stored with every run.

---

# Workstream M — Compute Strategy

## Local Machine

Use local machine for:

- dataset discovery,
- download,
- conversion,
- audit,
- annotation review,
- smoke tests,
- inference checks.

## GPU Training

Use:

- Colab,
- available CUDA GPU,
- or another compatible training environment.

The repository should remain hardware-neutral.

Recommended training phases:

```text
Smoke:
5–10 images / 1–3 epochs

Baseline:
EMPS / modest resolution

Domain adaptation:
Gold dataset

Final:
Au@SiO2 high-resolution fine-tune
```

Do not spend large GPU hours until:

- annotations pass audit,
- splits pass leakage checks,
- smoke training works.

---

# Kill / Go Gates

## Gate 1 — Dataset legality

### GO

- explicit usable license,
- clear provenance,
- required attribution known.

### KILL / HOLD

- unclear reuse rights,
- source figures scraped from papers without confirmed reuse rights,
- unknown provenance.

---

## Gate 2 — External pretraining usefulness

### GO

If EMPS/gold pretraining improves:

- exact-domain performance,
- data efficiency,
- convergence,
- low-contrast robustness,

then retain it.

### KILL

If direct Au@SiO2 training performs equivalently, simplify the stack.

---

## Gate 3 — AI integration

### GO

AI backend should enter a public Corpus release only if it provides at least one of:

- better recall than classical segmentation,
- materially lower manual correction time,
- improved low-contrast recovery,
- better metrology agreement.

### HOLD

If it only produces visually impressive masks with no workflow benefit.

---

# Recommended Execution Order

## Sprint 1 — Dataset discovery + legal gates

- A1 registry
- A2 search
- B1 downloader

Deliverable:

```text
approved dataset inventory
```

---

## Sprint 2 — Normalize + audit

- C1 adapters
- D1 group splits
- dataset integrity checks

Deliverable:

```text
reproducible COCO master datasets
```

---

## Sprint 3 — General EM pretraining

- E1 EMPS baseline

Deliverable:

```text
general nanoparticle segmentation checkpoint
```

---

## Sprint 4 — Gold adaptation

- F1 gold TEM fine-tune

Deliverable:

```text
gold-domain checkpoint + benchmark
```

---

## Sprint 5 — Exact Au@SiO2

- G1 annotation curation
- G2 three-way initialization ablation

Deliverable:

```text
two-class Au_core / SiO2_outer model
```

---

## Sprint 6 — Scientific evaluation

- H1 metrology metrics
- I1 classical vs AI vs hybrid

Deliverable:

```text
locked-test benchmark
```

---

## Sprint 7 — Product integration

- J1 model package
- K1 backend
- L1 one-command runner

Deliverable:

```text
Corpus AI optional backend
```

---

# Minimum Viable AI Release

Do not wait for a perfect model.

A valid **Corpus AI v0.1** requires:

- [ ] verified external datasets
- [ ] reproducible downloads
- [ ] leakage-safe splits
- [ ] EMPS baseline
- [ ] exact Au@SiO2 fine-tune
- [ ] two-class masks
- [ ] metrology-aware evaluation
- [ ] documented limitations
- [ ] optional AI backend
- [ ] human review before final measurement

---

# Stretch Goal — Low-Contrast Failure

The current classical Corpus self-check showed a real failure mode:

```text
outer carriers detected
core detections = 0
```

on a low-contrast phantom.

Use this as a targeted AI benchmark.

Build a low-contrast subset and compare:

```text
Classical
vs
AI
vs
Hybrid
```

Measure:

```text
core recall
false positives
D_core error
D_total error
shell error
```

A meaningful improvement here would be one of the strongest reasons to include AI in Corpus.

---

# Final Architecture

```text
                    ┌──────── EMPS ────────┐
                    │                      │
External datasets ──┼──── Gold TEM ────────┼──► registry/license gate
                    │                      │
                    └── robustness data ───┘
                                   │
                                   ▼
                         normalized COCO
                                   │
                         leakage-safe splits
                                   │
                                   ▼
                          general pretraining
                                   │
                         gold domain adaptation
                                   │
                                   ▼
                         Au@SiO2 fine-tuning
                                   │
                                   ▼
                          metrology benchmark
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
             AIBackend                       model artifact
                   │
                   ▼
             Hybrid review
                   │
                   ▼
           accepted measurements
```

---

# Definition of Done

The Epic is complete when a developer can run a documented workflow that:

```text
searches/reads registry
        ↓
downloads only legally approved data
        ↓
verifies checksums
        ↓
normalizes annotations
        ↓
creates leakage-safe splits
        ↓
pretrains on general EM nanoparticles
        ↓
fine-tunes toward gold/Au@SiO2
        ↓
evaluates segmentation + metrology
        ↓
packages the model
        ↓
loads it in Corpus
        ↓
lets a researcher review predictions
```

and every final measurement retains:

```text
image provenance
calibration
segmentation backend
model version
confidence
review status
measurement output
```

---

# Key Principle

> **Corpus AI should not replace the researcher. It should turn segmentation from a manual drawing problem into a rapid review problem.**
