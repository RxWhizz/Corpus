# Corpus Training

Training starts after real Au@SiO2 TEM images are curated and annotated. Corpus is the local curation, metadata, scale, COCO, and YOLO-seg conversion tool; CVAT or Label Studio is used for manual instance masks.

## Ontology v0

- `0: Au_core`: visible Au core mask.
- `1: SiO2_outer`: full visible outer boundary of the Au@SiO2 particle.

`SiO2_outer` is not only the silica ring; it is the total outside contour used to calculate `D_total`.

## Dataset Layers

- `real_exact`: real Au@SiO2 core-shell TEM images; main training truth.
- `real_near`: related nanoparticle TEM/SEM datasets; transfer/pretraining only.
- `synthetic_core_shell`: generated TEM-like images for curriculum and augmentation.
- `public_demo`: redistributable subset with clear CC BY/CC0/public license.
- `private_training`: internal-only images that must not be redistributed.

## Registry and Legal Gate

External datasets are listed in `training/datasets/registry.yaml`.
The downloader refuses any entry unless all of these are true:

- `license_status: verified`
- `enabled: true`
- at least one `download_urls` value is present
- verified downloadable entries record a checksum

Inspect the registry:

```bash
python training/dataset_registry.py
python training/dataset_discovery_report.py
```

Download every currently approved dataset:

```bash
python training/download_datasets.py --all-approved
```

Datasets that still say `needs_review` are intentionally blocked until their
license, citation, source version, and checksum are reviewed.

## Local PDF Sources

Local paper PDFs can be placed in `Examples/pdfs TEM/`. They are ignored by Git and should be treated as manual-review/private-training material until license and figure reuse rights are confirmed.

```powershell
python training\fetch_training_assets.py local-pdfs
python training\fetch_training_assets.py seed-sources
```

After extracting embedded PDF images, triage them before CVAT curation:

```powershell
python training\extract_paper_figures.py --pdf "Examples\pdfs TEM\materials-17-02213-v2.pdf" --out data\interim\figures\pdf_extracted\materials-17-02213
python training\triage_pdf_figures.py --input data\interim\figures\pdf_extracted --out data\interim\figures\triaged --clean
```

The triage step separates TEM candidates, graphs, abstracts/schemes, mixed figures, and dense TEM tiles. Review `data/interim/figures/triaged/figure_triage_manifest.csv` before exporting anything to CVAT.

The machine-readable inventory is `training/seed_sources.json`; the human guide is `docs/local_pdf_sources.md`.

## Local Flow

1. Curate images in Corpus Builder and confirm metadata, license, scale, and checksum.

2. Export accepted images for CVAT:

   ```powershell
   python training\export_cvat_package.py --out data\training\cvat_package
   ```

   Public-only package:

   ```powershell
   python training\export_cvat_package.py --layer public_demo --out data\training\cvat_public_demo
   ```

3. Annotate in CVAT using polygon instance segmentation labels:

   - `Au_core`
   - `SiO2_outer`

   Exclude scale bars, panel letters, severe blur, edge-cut particles, and ambiguous overlaps. If an object must remain in the master COCO but should not train, mark it with `review_status=needs_review`.

4. Export from CVAT as COCO instance segmentation, then import it:

   ```powershell
   python training\import_cvat_coco.py --coco path\to\instances_default.json
   ```

5. Prepare YOLO-seg and audit it:

   ```powershell
   python training\prepare_yolo_seg.py
   python training\audit_training_dataset.py --min-images 5 --min-au-core 5 --min-sio2-outer 5
   python training\audit_split_leakage.py --manifest data\training\yolo_seg\manifest.csv
   ```

6. Run a dry reproducibility gate locally:

   ```powershell
   python training\train_corpus_seg.py --config configs\training\au_sio2_v1.yaml --dry-run --force
   ```

7. Upload `data\training\yolo_seg` to Colab/Drive and run `training\colab_train_yolo_seg.ipynb`,
   or run the same config in a GPU environment:

   ```powershell
   python training\train_corpus_seg.py --config configs\training\au_sio2_v1.yaml
   ```

## Normalization and Splits

Approved or curated sources normalize into:

```text
data/normalized/<dataset_id>/
├── images/
├── annotations.json
├── manifest.csv
└── provenance.json
```

Adapters live in `training/adapters/`:

- `bam_tio2.py`: BAM/Zenodo TiO2 SEM images + aligned manual masks as one-class `particle`.
- `corpus_native.py`: Corpus/CVAT COCO with `Au_core` and `SiO2_outer`.
- `emps.py`: EMPS image/segmap folders as one-class `particle`.
- `generic_masks.py`: image + instance-mask datasets as one-class `particle`.
- `psdi_gold.py`: gold-particle COCO while preserving its non-core-shell ontology.

Build leakage-safe splits from a normalized manifest or COCO file:

```bash
python training/build_splits.py --manifest data/normalized/corpus_exact_au_sio2/manifest.csv --out data/normalized/corpus_exact_au_sio2/manifest.split.csv
python training/audit_split_leakage.py --manifest data/normalized/corpus_exact_au_sio2/manifest.split.csv --require-test --exact-test-only
```

The split auditor also flags duplicate checksums and near-duplicate image
hashes, especially when they cross train/val/test.

## Particle Pretraining Run

PSDI Gold TEM and BAM TiO2 can be combined because both export the same
one-class ontology:

- `0: particle`

Build the combined local pretraining dataset:

```bash
npm run training:combine-particle
node training/run_python.js training/train_corpus_seg.py --config configs/training/particle_pretrain_psdi_bam.yaml --dry-run --force
```

The current combined dataset contains 1,714 images and 74,636 particle
polygons. It is pretraining/domain-adaptation material only; it does not contain
`Au_core`/`SiO2_outer` masks.

For the local RX570/gfx803 stack, use the revive ROCm container:

```bash
npm run training:rocm-particle-pretrain
```

The preliminary `0.4.0` continuation starts from the interrupted ROCm checkpoint
at `runs/training/particle_pretrain_psdi_bam_rocm_v0_1_03/weights/last.pt` and
writes a fresh run named `particle_pretrain_psdi_bam_rocm_v0_4_0`:

```bash
npm run training:rocm-particle-pretrain-0.4.0
```

This run uses the combined PSDI Gold TEM + BAM TiO2 particle dataset:
1,714 images and 74,636 one-class `particle` polygons. It is pretraining and
domain-adaptation material only; final Au@SiO2 core/shell metrology still needs
Corpus/CVAT `Au_core` and `SiO2_outer` annotations.

The revive stack was intentionally built without DDP. Multi-GPU usage is
therefore one process per GPU:

```bash
npm run training:rocm-particle-pretrain-dual
```

This launches two independent runs, one on each GPU, and is useful for
parallel hyperparameter exploration. It does not merge both GPUs into one
faster training job.

## One-ZIP Colab Bundle

Generate a Colab-ready ZIP from real CVAT COCO:

```powershell
python training\package_colab_bundle.py --coco data\annotations\cvat_coco_imported.json --clean
```

Or generate a synthetic smoke bundle before real annotation exists:

```powershell
python training\package_colab_bundle.py --synthetic-smoke --clean
```

Convenience npm commands use a Python launcher that works around Windows Python aliases:

```powershell
npm.cmd run training:synthetic-bundle
npm.cmd run training:package
npm.cmd run training:audit
```

The output is:

```text
data\training\colab_bundle\corpus_colab_training_bundle.zip
```

Upload that ZIP in `training\colab_train_yolo_seg.ipynb` and run all cells. The ZIP contains:

- `dataset/data.yaml`
- `dataset/images/{train,val,test}`
- `dataset/labels/{train,val,test}`
- `dataset/manifest.csv`
- `reports/training_dataset_audit.md`
- `docs/dataset_sources.md`
- `training/seed_sources.json`
- `training/colab_run_training.py`

You can also run the Colab runner directly:

```bash
python colab_run_training.py --bundle /content/corpus_colab_training_bundle.zip --smoke-only
python colab_run_training.py --dataset /content/corpus_yolo_seg --full --epochs 75 --imgsz 1024 --batch 4
```

## Reproducible Experiment Runner

The one-command runner is hardware-neutral. It loads
`configs/training/au_sio2_v1.yaml`, validates the dataset registry, runs the
YOLO dataset audit, checks split leakage, records the exact command/config, and
then trains only if all gates pass.

Outputs are written under `runs/training/<run_id>/`:

- `training_config.yaml`
- `dataset_manifest.json`
- `metrics.json`
- `checksums.txt`
- `model_card.md`
- `training_dataset_audit.md`
- `split_leakage_audit.md`

## Synthetic Smoke Dataset

Generate a small synthetic COCO dataset:

```powershell
python training\generate_synthetic_core_shell.py --count 25
python training\prepare_yolo_seg.py --coco data\training\synthetic_core_shell\synthetic_core_shell_coco.json --out data\training\synthetic_yolo_seg
python training\audit_training_dataset.py --dataset data\training\synthetic_yolo_seg --min-images 5 --min-au-core 5 --min-sio2-outer 5
```

Synthetic data is for smoke tests, curriculum, and robustness checks. It does not replace real annotated Au@SiO2 TEM images.

## EMPS Pretraining Dataset

For a general EM/TEM particle detector, use EMPS as `real_near_emps`. It has one YOLO-seg class:

- `0: particle`

This pretraining layer teaches particle boundaries, overlap, and EM texture. It is not Au@SiO2 truth and should not be used to report core/shell metrology.

```powershell
git clone --depth 1 https://github.com/by256/emps.git data\external\emps
npm.cmd run training:prepare-emps
npm.cmd run training:package-emps
```

The prepared dataset is `data\training\emps_yolo_seg`; the Colab ZIP is `data\training\colab_bundle\corpus_colab_training_bundle.zip`. Train this first as a single-class `particle` model, then fine-tune on Corpus Au@SiO2 annotations with `Au_core` and `SiO2_outer`.

## PSDI Gold TEM Dataset

The PSDI record `sgvf0-j3g53` provides a CC BY 4.0 gold-nanoparticle TEM
segmentation dataset with COCO annotations:

- `train`: 1,501 synthetic TEM images.
- `test`: 149 synthetic TEM images.
- `val`: 24 experimental manually annotated TEM images.
- ontology: one class, `0: particle`.

Place the downloaded source files in `data/external/psdi_gold_tem_2026/`:

```text
images.zip
instances_annotations_train.json
instances_annotations_test.json
instances_annotations_val.json
val_binary_masks.zip
croissant_metadata.json
ro-crate-metadata.json
README.md
```

Normalize and prepare:

```bash
npm run training:normalize-psdi-gold
npm run training:prepare-psdi-gold
npm run training:audit-psdi-gold
npm run training:audit-psdi-gold-splits
```

`training:audit-psdi-gold-splits` checks exact checksum/group leakage. The
stricter perceptual audit can be run with `npm run training:audit-psdi-gold-near`;
it is expected to flag many train/test near-duplicates because the synthetic
images share visual generation patterns. Treat PSDI as particle-boundary
pretraining/domain-adaptation material, not Au@SiO2 core-shell truth.

## BAM TiO2 Pretraining Dataset

The Zenodo/BAM TiO2 dataset in `4563942.zip` contains SEM/TSEM micrographs and
several segmentation-mask branches. Its source license is `CC BY-NC-ND 4.0`, so
Corpus treats it as `verified_restricted`: useful for local/private training,
blocked from automatic download and public-demo redistribution.

The direct, pixel-aligned branch is:

- `Electron Microscopy Images/SEM`
- `Electron Microscopy Image Masks/TiO2_Masks_Manual_4connected`

The TSEM registration branch and the 2-class/4-class mask branches are left out
until their frame transforms are handled explicitly.

```bash
unzip -q 4563942.zip -d /tmp/corpus_bam_tio2
unzip -q /tmp/corpus_bam_tio2/Datasets.zip -d data/external/agglomerated_non_spherical_em
npm run training:normalize-bam-tio2
npm run training:prepare-bam-tio2
npm run training:audit-bam-tio2
npm run training:audit-bam-tio2-splits
```

The output is `data/training/agglomerated_non_spherical_em_yolo_seg` with one
YOLO-seg class:

- `0: particle`

Use it only as `real_near` particle-boundary pretraining. It is not Au@SiO2
core-shell truth and must not be used to report core/shell metrology.

## Metrology Export

After importing CVAT COCO, calculate per-object metrology rows:

```powershell
python training\metrology_from_coco.py --coco data\annotations\cvat_coco_imported.json --out data\training\metrology_from_coco.csv
```

The CSV includes `D_core_nm`, `D_total_nm`, and `t_shell_nm` for paired core-shell instances. Use it for comparison against Fiji/ImageJ manual measurements.

## Backend Benchmark

Benchmark classical, manual-reference, AI-only, or hybrid backends against a
locked COCO file:

```bash
python training/benchmark_segmentation_backends.py --coco data/annotations/cvat_coco_imported.json --backend classical --backend manual
python training/benchmark_segmentation_backends.py --coco data/annotations/cvat_coco_imported.json --backend ai --model runs/training/corpus-seg-au-sio2-v0.1.0/weights/best.pt
```

The benchmark writes `benchmark.csv`, `benchmark.json`, and `backend_iou.png`
under `reports/segmentation_benchmark/`. AI and hybrid predictions retain
`review_required=True`; accepted metrology still comes from the review flow.

## Dataset Targets

- Smoke dataset: 5-10 annotated images.
- Pilot: 150-250 images/tiles.
- Useful v0: 300-600 images/tiles.
- Publication target: 600-1200 images/tiles, 3-5 sources, multiple magnifications and shell thickness ranges.

## Rules

- COCO is the master annotation format.
- YOLO-seg is generated only for training.
- Only images with at least one valid polygon label are exported to YOLO-seg.
- Splits are assigned by source group when no explicit split exists.
- Do not split tiles from the same source/micrograph across train/val/test.
- Local machine is for curation/conversion/audit; Colab/cloud GPU is for training.
