"""Testable controller and exporter used by the Tkinter manual workflow."""
from __future__ import annotations
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
import shutil
import cv2
from evaluation import character_metrics, load_ground_truth, normalize_text
from ocr_pipeline import normalize_detection, render_annotated_image
from preprocess import recipe_label, run_pipeline, save_image


FINAL_FIELDS = [
    "image_name", "ground_truth",
    "raw_pipeline", "raw_ocr_text", "raw_confidence", "raw_cer", "raw_one_minus_cer_percent",
    "raw_image", "raw_annotated_image",
    "final_pipeline", "final_ocr_text", "final_confidence", "final_cer", "final_one_minus_cer_percent",
    "cer_improvement_percentage_points", "processed_image", "annotated_image",
]
DISPLAY_RESULT_FIELDS = ["image_name", "section", "item", "result"]
ATTEMPT_FIELDS = [
    "image_name", "attempt_id", "label", "pipeline", "selected", "ocr_text",
    "weighted_confidence", "cer", "one_minus_cer_percent", "ground_truth",
]


def _reading_order(detections: list[dict]) -> list[list[dict]]:
    """Arrange ordinary horizontal OCR regions from top-to-bottom, left-to-right."""
    lines: list[dict] = []
    for detection in sorted(
        detections,
        key=lambda item: ((item["y_min"] + item["y_max"]) / 2, item["x_min"]),
    ):
        center = (detection["y_min"] + detection["y_max"]) / 2
        height = max(1, detection["y_max"] - detection["y_min"])
        for line in lines:
            if abs(center - line["center"]) <= 0.4 * min(height, line["height"]):
                line["items"].append(detection)
                break
        else:
            lines.append({"center": center, "height": height, "items": [detection]})
    return [sorted(line["items"], key=lambda item: item["x_min"]) for line in lines]


def _assemble_text(detections: list[dict]) -> str:
    return "\n".join(item["text"] for line in _reading_order(detections) for item in line)


def candidate_statistics(result: dict) -> None:
    """Calculate the OCR summary for the selected image."""
    detections = result["detections"]
    result["text"] = _assemble_text(detections)
    weights = [max(1, len(normalize_text(item["text"], "normalized"))) for item in detections]
    result["weighted_confidence"] = (
        sum(item["confidence"] * weight for item, weight in zip(detections, weights))
        / max(1, sum(weights))
    )


def run_ocr(reader, image_name: str, processed, threshold: float = 0.20) -> dict:
    image_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR) if processed.ndim == 2 else processed
    raw = reader.readtext(
        image_bgr,
        detail=1,
        rotation_info=None,
        workers=0,
        contrast_ths=0.0,
        paragraph=False,
    )
    result = {
        "detections": [normalize_detection(image_name, index, item, threshold) for index, item in enumerate(raw, 1)]
    }
    candidate_statistics(result)
    result["annotated"] = render_annotated_image(image_bgr, result["detections"])
    return result


def execute_attempt(
    reader,
    image_path: Path,
    original,
    recipe: list,
    ground_truth: str = "",
    threshold: float = 0.20,
    label: str = "Selected steps",
    context: dict | None = None,
) -> dict:
    pipeline = run_pipeline(original, recipe, context or {})
    ocr = run_ocr(reader, image_path.name, pipeline.image, threshold)
    metric = character_metrics(ground_truth, ocr["text"], "normalized") if ground_truth.strip() else None
    return {
        "label": label,
        "recipe": pipeline.recipe,
        "pipeline_label": recipe_label(pipeline.recipe),
        "processed": pipeline.image,
        "steps": pipeline.steps,
        "metadata": pipeline.metadata,
        "detections": ocr["detections"],
        "text": ocr["text"],
        "weighted_confidence": ocr["weighted_confidence"],
        "annotated": ocr["annotated"],
        "evaluation": metric,
    }


def save_ground_truth(path: Path, image_name: str, text: str) -> dict[str, str]:
    answers = load_ground_truth(path)
    answers[image_name] = text
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "ground_truth"])
        writer.writerows(answers.items())
    return answers


def load_recipes(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "images": {}}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("images", {}), dict):
        raise ValueError("Invalid manual_recipes.json format")
    return {"version": 1, "images": data.get("images", {})}


def save_recipes(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_values(attempt: dict) -> tuple[str, str]:
    metric = attempt.get("evaluation") or {}
    if metric.get("status") != "ok":
        return "", ""
    return str(metric["cer"]), str(metric["one_minus_cer_percent"])


def _display_result_rows(final_rows: list[dict]) -> list[dict]:
    """Build a vertical, human-readable CSV while retaining detailed data separately."""
    rows: list[dict] = []
    for result in final_rows:
        name = result["image_name"]

        def add(part: str, item: str, value) -> None:
            rows.append({"image_name": name, "section": part, "item": item, "result": value})

        raw_cer = f"{float(result['raw_cer']):.2%}" if result["raw_cer"] else "not calculated"
        final_cer = f"{float(result['final_cer']):.2%}" if result["final_cer"] else "not calculated"
        raw_confidence = f"{float(result['raw_confidence']):.4f}" if result["raw_confidence"] != "" else "—"
        final_confidence = f"{float(result['final_confidence']):.4f}" if result["final_confidence"] != "" else "—"
        improvement = (
            f"{float(result['cer_improvement_percentage_points']):.2f} percentage points"
            if result["cer_improvement_percentage_points"]
            else "not available"
        )
        add("Image", "Ground truth", result["ground_truth"] or "not entered")
        add("Original baseline", "Steps", result["raw_pipeline"] or "raw")
        add("Original baseline", "OCR text", result["raw_ocr_text"] or "no text detected")
        add("Original baseline", "Confidence", raw_confidence)
        add("Original baseline", "CER", raw_cer)
        add("Final result", "Steps", result["final_pipeline"])
        add("Final result", "OCR text", result["final_ocr_text"] or "no text detected")
        add("Final result", "Confidence", final_confidence)
        add("Final result", "CER", final_cer)
        add("Final result", "CER improvement", improvement)
    return rows


def _html_report(entries: list[dict], final_rows: list[dict], attempt_rows: list[dict]) -> str:
    final_by_name = {row["image_name"]: row for row in final_rows}
    attempts_by_name: dict[str, list[dict]] = {}
    for row in attempt_rows:
        attempts_by_name.setdefault(row["image_name"], []).append(row)
    cards = []
    for entry in entries:
        name = entry["image_name"]
        final = final_by_name.get(name)
        if not final:
            continue
        attempt_cards = []
        for row in attempts_by_name.get(name, []):
            state = " selected" if row["selected"] == "true" else ""
            cer = f"{float(row['cer']):.2%}" if row["cer"] else "no ground truth"
            attempt_cards.append(
                f'<div class="attempt{state}"><b>{escape(row["label"])}</b>'
                f'<div>{escape(row["pipeline"])}</div><div>confidence: {float(row["weighted_confidence"]):.3f} · CER: {cer}</div>'
                f'<div><b>OCR text:</b></div><pre>{escape(row["ocr_text"])}</pre></div>'
            )
        raw_cer = f"{float(final['raw_cer']):.2%}" if final["raw_cer"] else "not tested or no ground truth"
        final_cer = f"{float(final['final_cer']):.2%}" if final["final_cer"] else "no ground truth"
        improvement = (
            f"{float(final['cer_improvement_percentage_points']):.2f} percentage points"
            if final["cer_improvement_percentage_points"] else "not available"
        )
        raw_confidence = f"{float(final['raw_confidence']):.3f}" if final["raw_confidence"] != "" else "—"
        final_confidence = f"{float(final['final_confidence']):.3f}" if final["final_confidence"] != "" else "—"
        original_rel = final.get("raw_image") or f'originals/{entry["artifact_id"]}{Path(name).suffix.lower()}'
        raw_annotated_rel = final.get("raw_annotated_image", "")
        raw_annotated_figure = (
            f'<figure><img src="{escape(raw_annotated_rel)}"><figcaption>Original OCR annotations</figcaption></figure>'
            if raw_annotated_rel
            else '<figure class="missing"><div>No original annotations</div><figcaption>Run the original baseline first</figcaption></figure>'
        )
        cards.append(
            f'<section class="image-card"><h2>{escape(name)}</h2>'
            f'<div class="truth"><span>Ground truth</span><pre>{escape(final["ground_truth"] or "not entered")}</pre></div>'
            f'<article class="result-block raw-block"><div class="block-title"><span>1</span><h3>Original baseline</h3></div>'
            f'<div class="visuals"><figure><img src="{escape(original_rel)}"><figcaption>Original image</figcaption></figure>{raw_annotated_figure}</div>'
            f'<div class="metrics"><div><small>Steps</small><strong>{escape(final["raw_pipeline"] or "raw")}</strong></div>'
            f'<div><small>confidence</small><strong>{raw_confidence}</strong></div><div><small>CER</small><strong>{raw_cer}</strong></div></div>'
            f'<div class="ocr-text"><h4>Original OCR text</h4><pre>{escape(final["raw_ocr_text"] or "no text detected")}</pre></div></article>'
            f'<div class="divider"><span>↓</span> Compared with the original</div>'
            f'<article class="result-block final-block"><div class="block-title"><span>2</span><h3>Final result</h3></div>'
            f'<div class="visuals"><figure><img src="{escape(final["processed_image"])}"><figcaption>Processed image</figcaption></figure>'
            f'<figure><img src="{escape(final["annotated_image"])}"><figcaption>Final OCR annotations</figcaption></figure></div>'
            f'<div class="metrics"><div><small>Final steps</small><strong>{escape(final["final_pipeline"])}</strong></div>'
            f'<div><small>confidence</small><strong>{final_confidence}</strong></div><div><small>CER</small><strong>{final_cer}</strong></div>'
            f'<div><small>CER improvement</small><strong>{improvement}</strong></div></div>'
            f'<div class="ocr-text"><h4>Final OCR text</h4><pre>{escape(final["final_ocr_text"] or "no text detected")}</pre></div></article>'
            f'<details><summary>View attempts for this image</summary>{"".join(attempt_cards)}</details></section>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ME5405 Manual Preprocessing Results</title>
<style>
body{{font-family:"Microsoft YaHei",Arial,sans-serif;margin:24px auto;max-width:1120px;padding:0 16px;background:#eef1f5;color:#1f2937}}
h1{{margin-bottom:4px}} .note{{color:#5b6472}} .image-card{{background:white;padding:22px;margin:22px 0;border-radius:14px;box-shadow:0 3px 12px #0001}}
.truth{{background:#fff7ed;border-left:5px solid #f59e0b;padding:10px 14px;margin:12px 0 20px;border-radius:7px}} .truth span{{font-weight:700;color:#9a3412}}
.truth pre,.ocr-text pre{{white-space:pre-wrap;word-break:break-word;margin:7px 0 0;font:600 18px/1.55 Consolas,"Microsoft YaHei",sans-serif}}
.result-block{{padding:18px;border-radius:12px;border:2px solid #cbd5e1}} .raw-block{{background:#f8fafc}} .final-block{{background:#f0fdf4;border-color:#22c55e}}
.block-title{{display:flex;align-items:center;gap:10px;margin-bottom:14px}} .block-title span{{display:grid;place-items:center;width:30px;height:30px;border-radius:50%;background:#475569;color:white;font-weight:700}}
.final-block .block-title span{{background:#16a34a}} .block-title h3{{margin:0;font-size:22px}}
.visuals{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} figure{{margin:0}} img{{width:100%;height:330px;object-fit:contain;background:#e5e7eb;border-radius:8px}}
figcaption{{text-align:center;margin-top:7px;color:#475569}} .missing{{height:330px;display:grid;place-items:center;background:#e5e7eb;border-radius:8px;color:#64748b}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}} .metrics div{{background:white;border:1px solid #d7dce2;padding:10px;border-radius:8px}}
.metrics small{{display:block;color:#64748b;margin-bottom:5px}} .metrics strong{{display:block;word-break:break-word}}
.ocr-text{{background:#111827;color:#f9fafb;padding:14px;border-radius:9px}} .ocr-text h4{{margin:0;color:#93c5fd}}
.divider{{text-align:center;margin:14px 0;color:#475569;font-weight:700}} .divider span{{display:block;font-size:25px;color:#16a34a}}
details{{margin-top:18px}} .attempt{{border:1px solid #d7dce2;padding:10px;margin:8px 0;border-radius:8px}} .attempt.selected{{border:2px solid #16a34a;background:#f0fdf4}}
@media(max-width:760px){{.visuals,.metrics{{grid-template-columns:1fr}} img{{height:250px}}}}
</style></head><body><h1>ME5405 OCR Results</h1>
<p class="note">Each image shows the original baseline and the final result. CER uses the ground truth; confidence is not accuracy.</p>{''.join(cards)}</body></html>"""


def export_manual_run(output_root: Path, entries: list[dict]) -> Path:
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    final_rows: list[dict] = []
    attempt_rows: list[dict] = []
    recipe_export = {"version": 1, "images": {}}
    for entry in entries:
        name = entry["image_name"]
        artifact_id = entry["artifact_id"]
        original_path = Path(entry["image_path"])
        original_target = run_dir / "originals" / f"{artifact_id}{original_path.suffix.lower()}"
        original_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_path, original_target)
        selected_index = entry.get("selected_attempt")
        for index, attempt in enumerate(entry.get("attempts", []), 1):
            attempt_id = f"a{index:03}"
            folder = run_dir / "details" / artifact_id / attempt_id
            save_image(folder / "processed.png", attempt["processed"])
            save_image(folder / "annotated.png", attempt["annotated"])
            for step_name, step_image in attempt.get("steps", []):
                save_image(folder / "steps" / f"{step_name}.png", step_image)
            cer, score = _metric_values(attempt)
            attempt_rows.append({
                "image_name": name,
                "attempt_id": attempt_id,
                "label": attempt["label"],
                "pipeline": attempt["pipeline_label"],
                "selected": str(index - 1 == selected_index).lower(),
                "ocr_text": attempt["text"],
                "weighted_confidence": attempt["weighted_confidence"],
                "cer": cer,
                "one_minus_cer_percent": score,
                "ground_truth": entry.get("ground_truth", ""),
            })
        if selected_index is None or not 0 <= selected_index < len(entry.get("attempts", [])):
            continue
        selected = entry["attempts"][selected_index]
        raw = next(
            (attempt for attempt in entry["attempts"] if [step["op"] for step in attempt.get("recipe", [])] == ["raw"]),
            None,
        )
        raw_image_rel = f"originals/{artifact_id}{original_path.suffix.lower()}"
        raw_annotated_rel = f"raw/{artifact_id}_annotated.png" if raw else ""
        processed_rel = f"final/{artifact_id}_processed.png"
        annotated_rel = f"final/{artifact_id}_annotated.png"
        if raw:
            save_image(run_dir / raw_annotated_rel, raw["annotated"])
        save_image(run_dir / processed_rel, selected["processed"])
        save_image(run_dir / annotated_rel, selected["annotated"])
        final_cer, final_score = _metric_values(selected)
        raw_cer, raw_score = _metric_values(raw) if raw else ("", "")
        improvement = str(100 * (float(raw_cer) - float(final_cer))) if raw_cer and final_cer else ""
        final_rows.append({
            "image_name": name,
            "ground_truth": entry.get("ground_truth", ""),
            "raw_pipeline": raw["pipeline_label"] if raw else "",
            "raw_ocr_text": raw["text"] if raw else "",
            "raw_confidence": raw["weighted_confidence"] if raw else "",
            "raw_cer": raw_cer,
            "raw_one_minus_cer_percent": raw_score,
            "raw_image": raw_image_rel,
            "raw_annotated_image": raw_annotated_rel,
            "final_pipeline": selected["pipeline_label"],
            "final_ocr_text": selected["text"],
            "final_confidence": selected["weighted_confidence"],
            "final_cer": final_cer,
            "final_one_minus_cer_percent": final_score,
            "cer_improvement_percentage_points": improvement,
            "processed_image": processed_rel,
            "annotated_image": annotated_rel,
        })
        recipe_export["images"][name] = {
            "label": selected["label"],
            "recipe": selected["recipe"],
        }
    _write_csv(run_dir / "results.csv", _display_result_rows(final_rows), DISPLAY_RESULT_FIELDS)
    _write_csv(run_dir / "results_detailed.csv", final_rows, FINAL_FIELDS)
    _write_csv(run_dir / "attempts.csv", attempt_rows, ATTEMPT_FIELDS)
    (run_dir / "recipes.json").write_text(json.dumps(recipe_export, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "overview.html").write_text(_html_report(entries, final_rows, attempt_rows), encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    return run_dir
