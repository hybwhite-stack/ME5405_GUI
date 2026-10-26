# GUI OCR preprocessing tool
This repository contains a visual OCR preprocessing experiment tool for the
ME5405 Machine Vision assignment. It lets each image use a different processing
recipe, previews every change before OCR, and compares the processed result with
the raw-image baseline using confidence and character error rate (CER).

## Main features
- EasyOCR recognition for Simplified Chinese and English recognition
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

## AI usage declaration
GUI.py, ocr_pipeline.py, manual_workflow.py and pipeline.py were generated with the help of Codex to form a workflow with GUI to improve the efficiency of images processing and data collection.

Other files were typed by the developers with limited AI assistance. AI tools were mainly used to learn the libaries and their functions.

During every step of the project, the final implementation and logic were always reviewed and modified by the developers to ensure correctness and functionality.