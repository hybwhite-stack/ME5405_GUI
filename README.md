# ME5405 GUI OCR preprocessing tool

This repository contains a visual OCR preprocessing experiment tool for the
ME5405 Machine Vision assignment. It lets each image use a different processing
recipe, previews every change before OCR, and compares the processed result with
the raw-image baseline using confidence and character error rate (CER).

## Main features

- EasyOCR recognition for Simplified Chinese and English
- Manually selected preprocessing operations in an explicit order
- Per-image parameters and recipes
- Raw-image baseline and processed-image comparison
- Manual ground truth and CER calculation
- HTML, CSV and annotated-image export

## Installation

Use Python 3.11 or 3.12 when possible, then run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

EasyOCR model files are downloaded into `models/easyocr/` on first use and are
not committed to Git.

## Usage

1. Put local test images in `images/raw/`.
2. Run `GUI.py`.
3. Select one image and add preprocessing operations in the order you want them to run.
4. Adjust the parameters and inspect the preview.
5. Enter the correct text and run OCR.
6. Compare CER with the raw-image baseline and select the final recipe.
7. Export the overview and final results.

See `使用教程.md` for the detailed Chinese guide.

## Privacy and generated files

Input photographs, ground-truth text, downloaded model weights, local recipes,
virtual environments and generated outputs are intentionally ignored by Git.
This prevents personal or experiment-specific data from being published. Each
collaborator keeps these files locally.
