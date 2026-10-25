"""Ground-truth evaluation. Confidence is never used as recognition accuracy."""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


def normalize_text(text: str, policy: str) -> str:
    # Region/line breaks are layout separators; strict preserves within-line spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if policy == "strict":
        return text.replace("\n", "")
    if policy != "normalized":
        raise ValueError(f"Unknown evaluation policy: {policy}")
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def character_metrics(reference: str, prediction: str, policy: str = "normalized") -> dict:
    ref, pred = normalize_text(reference, policy), normalize_text(prediction, policy)
    if not ref:
        return {"status": "empty_reference", "policy": policy}
    # Hirschberg is unnecessary for short sign text. Keep rows, carrying operation counts.
    # Tuple = (distance, substitutions, deletions, insertions); ties prefer substitution.
    previous = [(j, 0, 0, j) for j in range(len(pred) + 1)]
    for i, a in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, b in enumerate(pred, 1):
            if a == b:
                current.append(previous[j - 1])
            else:
                diag, delete, insert = previous[j - 1], previous[j], current[j - 1]
                current.append(min([
                    (diag[0] + 1, diag[1] + 1, diag[2], diag[3]),
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
                ], key=lambda entry: entry[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    n = len(ref)
    return {
        "status": "ok", "policy": policy, "reference_length": n,
        "prediction_length": len(pred), "substitutions": substitutions,
        "deletions": deletions, "insertions": insertions, "edit_distance": errors,
        "cer": errors / n,
        # Insertion penalties can make this negative. Never silently clamp it.
        "one_minus_cer_percent": 100 * (1 - errors / n),
        "reference_character_accuracy_percent": 100 * (n - substitutions - deletions) / n,
        "exact_match": ref == pred,
    }


def load_ground_truth(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"image_name", "ground_truth"}.issubset(reader.fieldnames):
            raise ValueError("ground_truth.csv 必须包含 image_name,ground_truth 两列")
        answers = {}
        for row in reader:
            name = row["image_name"]
            if not name or name in answers:
                raise ValueError(f"ground truth 文件名为空或重复: {name!r}")
            answers[name] = row["ground_truth"] or ""
        return answers
