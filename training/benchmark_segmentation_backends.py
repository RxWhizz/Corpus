import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from common_training import category_mapping, load_json, resolve_image_path, write_json

from corpus.segmentation import ClassicalBackend, ManualBackend, get_backend

DEFAULT_OUT = Path("reports") / "segmentation_benchmark"


def _polygon_to_contour(polygon):
    points = np.asarray(list(zip(polygon[0::2], polygon[1::2])), dtype=np.int32)
    if len(points) < 3:
        return None
    return points.reshape(-1, 1, 2)


def reference_contours(coco):
    mapping, unknown = category_mapping(coco.get("categories", []))
    if unknown:
        raise SystemExit(f"Unsupported benchmark categories: {unknown}")
    grouped = {}
    for annotation in coco.get("annotations", []):
        if annotation.get("category_id") not in mapping:
            continue
        for polygon in annotation.get("segmentation") or []:
            contour = _polygon_to_contour(polygon)
            if contour is not None:
                grouped.setdefault(annotation.get("image_id"), []).append(contour)
    return grouped


def mask_from_contours(shape, contours):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    return mask


def mask_metrics(reference, predicted):
    ref = reference > 0
    pred = predicted > 0
    intersection = int(np.logical_and(ref, pred).sum())
    union = int(np.logical_or(ref, pred).sum())
    ref_count = int(ref.sum())
    pred_count = int(pred.sum())
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    denom = ref_count + pred_count
    dice = 1.0 if denom == 0 else (2 * intersection) / denom
    return {"iou": iou, "dice": dice, "intersection_px": intersection, "union_px": union}


def make_backend(name, model_path="", confidence_threshold=0.25):
    if name == "classical":
        return ClassicalBackend()
    if name == "manual":
        return ManualBackend()
    if name in {"ai", "hybrid"}:
        if not model_path:
            raise SystemExit(f"--model is required for backend {name!r}.")
        return get_backend(name, model_path=model_path, confidence_threshold=confidence_threshold)
    raise SystemExit(f"Unknown benchmark backend: {name}")


def benchmark(coco_path, out_dir=DEFAULT_OUT, backend_names=("classical", "manual"), model_path="", max_images=0):
    coco_path = Path(coco_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco = load_json(coco_path)
    references = reference_contours(coco)
    backends = {name: make_backend(name, model_path=model_path) for name in backend_names}
    rows = []

    images = coco.get("images", [])
    if max_images:
        images = images[:max_images]
    for image in images:
        image_path = resolve_image_path(image.get("file_name", ""), coco_path)
        if not image_path:
            continue
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        gt_contours = references.get(image.get("id"), [])
        gt_mask = mask_from_contours(frame.shape, gt_contours)
        for name, backend in backends.items():
            start = time.perf_counter()
            if name == "manual":
                result = backend.predict(frame, contours=gt_contours)
            else:
                result = backend.predict(frame)
            elapsed = time.perf_counter() - start
            predicted_mask = mask_from_contours(frame.shape, result.contours)
            metrics = mask_metrics(gt_mask, predicted_mask)
            rows.append(
                {
                    "image_id": image.get("id"),
                    "file_name": image.get("file_name", ""),
                    "backend": name,
                    "reference_instances": len(gt_contours),
                    "predicted_instances": len(result.contours),
                    "count_error": len(result.contours) - len(gt_contours),
                    "iou": f"{metrics['iou']:.6f}",
                    "dice": f"{metrics['dice']:.6f}",
                    "seconds": f"{elapsed:.6f}",
                    "review_required": result.review_required,
                    "review_status": result.review_status,
                    "model_version": result.model_version,
                }
            )

    csv_path = out_dir / "benchmark.csv"
    fields = [
        "image_id",
        "file_name",
        "backend",
        "reference_instances",
        "predicted_instances",
        "count_error",
        "iou",
        "dice",
        "seconds",
        "review_required",
        "review_status",
        "model_version",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path = out_dir / "benchmark.json"
    write_json(json_path, summary)
    figure_path = write_figure(out_dir / "backend_iou.png", rows)
    return {"ok": True, "rows": len(rows), "csv": str(csv_path), "json": str(json_path), "figure": str(figure_path)}


def summarize(rows):
    by_backend = {}
    for row in rows:
        block = by_backend.setdefault(row["backend"], {"images": 0, "mean_iou": 0.0, "mean_dice": 0.0, "mean_seconds": 0.0})
        block["images"] += 1
        block["mean_iou"] += float(row["iou"])
        block["mean_dice"] += float(row["dice"])
        block["mean_seconds"] += float(row["seconds"])
    for block in by_backend.values():
        count = max(1, block["images"])
        block["mean_iou"] /= count
        block["mean_dice"] /= count
        block["mean_seconds"] /= count
    return {"ok": True, "backends": by_backend}


def write_figure(path, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    summary = summarize(rows)["backends"]
    if not summary:
        return ""
    names = sorted(summary)
    values = [summary[name]["mean_iou"] for name in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, values, color="#789f8a")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Segmentation backend benchmark")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Benchmark Corpus segmentation backends against COCO masks.")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--backend", action="append", choices=["classical", "manual", "ai", "hybrid"])
    parser.add_argument("--model", default="", help="Model path for ai/hybrid backends.")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    result = benchmark(
        args.coco,
        out_dir=args.out,
        backend_names=tuple(args.backend or ["classical", "manual"]),
        model_path=args.model,
        max_images=args.max_images,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
