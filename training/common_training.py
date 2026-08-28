import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
TRAINING_DIR = DATA_DIR / "training"
DEFAULT_IMPORTED_COCO = ANNOTATIONS_DIR / "cvat_coco_imported.json"
DEFAULT_YOLO_DIR = TRAINING_DIR / "yolo_seg"
DEFAULT_CVAT_DIR = TRAINING_DIR / "cvat_package"
SYNTHETIC_DIR = TRAINING_DIR / "synthetic_core_shell"
TRAINING_AUDIT_MD = ROOT / "reports" / "training_dataset_audit.md"

CLASS_NAMES = ["Au_core", "SiO2_outer"]
COCO_CATEGORIES = [
    {"id": 0, "name": "Au_core", "supercategory": "nanoparticle"},
    {"id": 1, "name": "SiO2_outer", "supercategory": "nanoparticle"},
]
CLASS_ALIASES = {
    "au_core": "Au_core",
    "au core": "Au_core",
    "core": "Au_core",
    "gold core": "Au_core",
    "gold_core": "Au_core",
    "au": "Au_core",
    "gold": "Au_core",
    "core_au": "Au_core",
    "sio2_outer": "SiO2_outer",
    "sio2 outer": "SiO2_outer",
    "sio2": "SiO2_outer",
    "silica": "SiO2_outer",
    "silica outer": "SiO2_outer",
    "silica_outer": "SiO2_outer",
    "sio2 carrier": "SiO2_outer",
    "sio2_carrier": "SiO2_outer",
    "carrier": "SiO2_outer",
    "shell": "SiO2_outer",
    "outer": "SiO2_outer",
}

PUBLIC_LICENSE_HINTS = ("cc by", "cc-by", "cc0", "public domain")
BLOCKED_LICENSE_HINTS = ("nc", "nd", "noncommercial", "no derivatives")
REVIEW_STATUSES = {"needs_review", "uncertain", "ambiguous", "ignore", "ignored", "difficult"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_class_name(name):
    normalized = " ".join(str(name or "").replace("-", " ").replace("_", " ").split()).lower()
    return CLASS_ALIASES.get(normalized, name)


def category_mapping(categories):
    mapping = {}
    unknown = []
    for category in categories:
        canonical = normalize_class_name(category.get("name", ""))
        if canonical in CLASS_NAMES:
            mapping[category["id"]] = CLASS_NAMES.index(canonical)
        else:
            unknown.append(category.get("name", ""))
    return mapping, unknown


def canonical_categories():
    return [dict(category) for category in COCO_CATEGORIES]


def is_public_license(row):
    status = str(row.get("license_status", "")).strip().lower()
    license_text = " ".join(
        str(row.get(key, "")).lower()
        for key in ("license", "license_url", "rights", "usage_terms")
    )
    if status == "accepted" and not any(hint in license_text for hint in BLOCKED_LICENSE_HINTS):
        return True
    return any(hint in license_text for hint in PUBLIC_LICENSE_HINTS) and not any(
        hint in license_text for hint in BLOCKED_LICENSE_HINTS
    )


def has_confirmed_scale(row):
    status = str(row.get("scale_status", "")).strip().lower()
    nm_per_px = str(row.get("nm_per_px", "")).strip()
    return bool(nm_per_px) and status in {"confirmed", "manual_line", "manual", "metadata"}


#: The dataset layers documented in training/README.md. They are two
#: independent axes that the flat list in the docs hides:
#:
#:   content      what the image *is*      real_exact | real_near | synthetic_core_shell
#:   distribution what may be *shared*     public_demo | private_training
#:
#: A publicly licensed Au@SiO2 micrograph is both `real_exact` and
#: `public_demo`; collapsing that to one label loses whichever half you did not
#: pick. `training_layer` keeps the historical single-label view for callers
#: that need one string; `content_layer` and `distribution_layer` give the axes.
CONTENT_LAYERS = ("real_exact", "real_near", "synthetic_core_shell", "unknown")
DISTRIBUTION_LAYERS = ("public_demo", "private_training")
DATASET_LAYERS = ("real_exact", "real_near", "synthetic_core_shell", "public_demo", "private_training")

#: Text that identifies a genuine Au@SiO2 core-shell subject -- the only
#: material Corpus reports core/shell metrology for.
CORE_SHELL_HINTS = (
    "au@sio2", "au@sio\u2082", "au-sio2", "au/sio2", "au sio2",
    "core-shell", "core shell", "coreshell",
    "gold core", "au core", "silica shell", "sio2 shell",
)

#: Text that identifies a related electron-microscopy particle dataset:
#: useful for transfer/pretraining, never for core-shell truth.
NEAR_HINTS = (
    "nanoparticle", "nanoparticles", "nanorod", "nanosphere", "colloid",
    "tem", "stem", "haadf", "bf-tem", "sem", "micrograph",
    "electron microscopy", "electron micrograph", "emps",
)

SYNTHETIC_HINTS = ("synthetic", "phantom", "simulated")


def _layer_text(row, source_row):
    """Free-text signals used to classify an image's content layer."""
    row = row or {}
    source_row = source_row or {}
    parts = [
        row.get(key, "") for key in
        ("notes", "caption", "modality", "figure_label", "source_type", "title")
    ] + [
        source_row.get(key, "") for key in
        ("title", "abstract", "keywords", "modality", "source_type", "journal")
    ]
    return " ".join(str(part) for part in parts).lower()


def declared_layer(row, source_row=None):
    """The curator's explicit `dataset_layer`, if one was recorded.

    An explicit decision always wins over text heuristics.
    """
    row = row or {}
    source_row = source_row or {}
    declared = str(row.get("dataset_layer", "") or source_row.get("dataset_layer", "")).strip().lower()
    return declared or ""


def content_layer(row, source_row=None):
    """What the image is: `real_exact`, `real_near`, `synthetic_core_shell`, `unknown`.

    `real_exact` is real Au@SiO2 core-shell data -- the only layer that may be
    used to report core/shell metrology. `real_near` is related EM particle
    data for transfer/pretraining only. Variants such as `real_near_emps`
    collapse to `real_near`.
    """
    declared = declared_layer(row, source_row)
    if declared:
        if declared in CONTENT_LAYERS:
            return declared
        # `real_near_emps` and friends belong to the real_near family.
        if declared.startswith("real_near"):
            return "real_near"
        if declared.startswith("real_exact"):
            return "real_exact"
        if any(hint in declared for hint in SYNTHETIC_HINTS):
            return "synthetic_core_shell"

    text = _layer_text(row, source_row)
    if any(hint in text for hint in SYNTHETIC_HINTS):
        return "synthetic_core_shell"
    if any(hint in text for hint in CORE_SHELL_HINTS):
        return "real_exact"
    if any(hint in text for hint in NEAR_HINTS):
        return "real_near"
    return "unknown"


def distribution_layer(row, source_row=None):
    """What may be shared: `public_demo` when redistributable, else `private_training`."""
    return "public_demo" if is_public_license(row) or is_public_license(source_row or {}) else "private_training"


def training_layer(row, source_row=None):
    """Historical single-label view, kept for callers that need one string.

    Distribution wins over content here, because the question this answers is
    "which bucket does this image belong to for export?". Its result is
    deliberately limited to the distribution buckets plus `synthetic_core_shell`,
    unchanged from before the content axis existed, so export filtering keeps
    behaving the same. Use :func:`content_layer` when you need to know whether
    an image is Au@SiO2 truth regardless of its licence.
    """
    source_row = source_row or {}
    if is_public_license(source_row) or is_public_license(row):
        return "public_demo"
    source_type = str(source_row.get("source_type", "") or row.get("source_type", "")).lower()
    notes = str(row.get("notes", "")).lower()
    if "synthetic" in source_type or "synthetic" in notes:
        return "synthetic_core_shell"
    return "private_training"


def layers_for(row, source_row=None):
    """Both axes plus the single-label view, for manifests and reports."""
    return {
        "dataset_layer": training_layer(row, source_row),
        "content_layer": content_layer(row, source_row),
        "distribution_layer": distribution_layer(row, source_row),
    }


def annotation_review_status(annotation):
    attrs = annotation.get("attributes") or {}
    if isinstance(attrs, list):
        attrs = {
            str(item.get("name", "")).lower(): item.get("value", "")
            for item in attrs
            if isinstance(item, dict)
        }
    values = [
        annotation.get("review_status", ""),
        annotation.get("status", ""),
        annotation.get("quality_status", ""),
        attrs.get("review_status", ""),
        attrs.get("status", ""),
        attrs.get("quality_status", ""),
    ]
    if annotation.get("iscrowd"):
        return "needs_review"
    for value in values:
        if str(value).strip().lower() in REVIEW_STATUSES:
            return "needs_review"
    return "ready"


def resolve_image_path(file_name, coco_path=None):
    raw = Path(str(file_name))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    if coco_path:
        candidates.append(Path(coco_path).resolve().parent / raw)
    candidates.append(ROOT / raw)
    candidates.append(DATA_DIR / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def safe_stem(value):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("._") or "image"


def stable_hash(value):
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def file_sha256(path, chunk_size=1 << 20):
    """SHA-256 of a file, streamed. Recorded in the dataset manifest so an
    exported image can be traced back to the exact bytes it came from."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_polygon(points, width, height):
    if len(points) < 6 or len(points) % 2:
        return None
    output = []
    for index, value in enumerate(points):
        limit = width if index % 2 == 0 else height
        if not limit:
            return None
        normalized = max(0.0, min(1.0, float(value) / float(limit)))
        output.append(normalized)
    return output


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


#: Manifest columns. Provenance columns (`file_sha256`, `license`,
#: `license_status`, `doi`, `source_url`) and annotation state
#: (`annotation_review`, `skipped_review_labels`) are part of the contract:
#: a training bundle without them cannot be traced back to its sources.
MANIFEST_FIELDS = [
    "image_id",
    "source_id",
    "split",
    "dataset_layer",
    "content_layer",
    "distribution_layer",
    "image_path",
    "label_path",
    "file_sha256",
    "labels",
    "au_core_labels",
    "sio2_outer_labels",
    "annotation_review",
    "skipped_review_labels",
    "nm_per_px",
    "calibration_state",
    "license",
    "license_status",
    "doi",
    "source_url",
    "figure_label",
    "panel_label",
    "caption",
]


def write_manifest(path, rows):
    fields = MANIFEST_FIELDS
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def copy_image(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
