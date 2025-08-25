```md
# Microgel & Cell Annotator — Backend (Flask + Ultralytics YOLO)

Flask API that:
- loads a pretrained YOLO model (`models/best.pt`),
- runs detection for uploaded images,
- exports **YOLO Darknet `.txt`** and **Excel** reports,
- supports **counting rules** that exclude microgels by **edge crop %** and **overlap IoU** (cells inside excluded microgels are also excluded).

---

## Prerequisites

- **Python 3.10** (recommended)
- **Virtualenv** (optional but recommended)
- The trained weights at `backend/models/best.pt`  
  - If the file is >100 MB, prefer Git LFS or download in a build script on Render.

---

## Quick Start (Local)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt    # includes numpy<2 to avoid known NumPy 2.x issues with some torch builds
python3 app.py                     # http://localhost:8000
# if port is busy:
# PORT=5001 python3 app.py
Model path: by default models/best.pt. You can override via MODEL_PATH env var.
Device: the app will use CUDA if available, otherwise CPU.

API
Health
GET /api/health → {"ok": true}
Detection
POST /api/detect (form-data)
* files: one or more image files (jpg, jpeg, png, bmp, tif, tiff)
* conf (optional): confidence threshold (accepts comma decimals like 0,25)
* iou (optional): NMS IoU threshold (comma decimals allowed)
Response
{
  "classMap": { "0": "microgel", "1": "cell" },
  "images": [
    {
      "id": "...",
      "filename": "Image001.jpg",
      "storedFilename": "<uuid>_Image001.jpg",
      "url": "/uploads/<uuid>_Image001.jpg",
      "width": 2048, "height": 1536,
      "boxes": [
        { "id":"...", "classId":0, "className":"microgel", "score":0.72, "x":123, "y":45, "w":80, "h":80 },
        { "id":"...", "classId":1, "className":"cell",     "score":0.41, "x":210, "y":60, "w":14, "h":16 }
      ],
      "counts": { "microgel": 37, "cell": 112 }
    }
  ]
}
Coordinates are pixels (x,y,w,h) with top‑left origin.The frontend converts to YOLO’s normalized format on export.
YOLO export (ZIP)
POST /api/export/yolo (JSON)
{ "images": [ { "filename":"...", "width":1234, "height":987, "boxes":[...]} ],
  "classMap": { "0": "microgel", "1": "cell" } }
Response: ZIP containing <image_stem>.txt per image + classes.txt.Each .txt line: class cx cy w h (normalized).
YOLO export (single txt for one image)
POST /api/export/yolo/txt (JSON)
{ "image": { "filename":"...", "width":1234, "height":987, "boxes":[...] } }
Response: one .txt (Darknet format).
Excel export — one file per image
* Single file: POST /api/export/excel/one
* ZIP of all Excels: POST /api/export/excel/each
Payload (both):
{
  "image": { ... },             // for /one
  "images": [ ... ],            // for /each
  "rules": {
    "overlap_iou": 0.10,        // exclude microgels if IoU >= this
    "edge_outside_percent": 50  // exclude microgels if > this % area lies outside the image
  }
}
What goes into each Excel (2 sheets):
* microgel_id_cell_count — for each valid microgel (not excluded by rules), count of cells whose centers lie inside that microgel; plus a Total row.
* microgel_id_cell_id — (microgel_id, cell_id) pairs mapping each cell to its containing valid microgel.
A microgel is excluded if edge_inside_ratio < 1 - edge_outside_percent/100or if its IoU with another microgel is ≥ overlap_iou.Cells inside excluded microgels are also excluded from the counts.

Deploy (Free) — Render
Create a Web Service from the backend directory.
* Build Commandpip install --upgrade pip
* pip install -r requirements.txt
* 
* Start Commandgunicorn --bind 0.0.0.0:$PORT app:app
* 
* Environment
    * PYTHON_VERSION=3.10
    * (optional) MODEL_PATH=models/best.pt
If your best.pt is too large for Git:
* Use Git LFS, or
* Create a render-build.sh that downloads it from a public URL during build and set Build Command to bash render-build.sh.
Free-plan notes
* Service sleeps when idle → expect a cold‑start delay on the first request.
* Disk is ephemeral; /uploads is not persistent. (Frontend already downloads results to the browser.)

Performance & Tuning
* max_det=5000, augment=True are used for dense scenes.
* To try larger images, adjust preprocessing in your training/inference notebooks; the API currently relies on Ultralytics defaults at predict time.
* GPU: Render free tier is CPU; local or paid tiers can use CUDA if available.

Troubleshooting
* Port 8000 is in useFind and kill: lsof -i :8000 → kill -9 <PID>Or run with PORT=5001 python3 app.py.
* NumPy / Torch import errors on macOSThis repo pins numpy<2 to avoid crashing native extensions compiled against NumPy 1.x.If you still see issues, clear caches and reinstall:pip uninstall -y numpy
* pip cache purge
* pip install "numpy<2"
* pip install -r requirements.txt --force-reinstall
* 
* CORSWe ship with flask-cors enabled. In production we recommend Netlify edge _redirects to proxy /api/* to Render so the app behaves as same‑origin.

Project Layout (backend)
backend/
  app.py
  requirements.txt
  models/
    best.pt         # your trained weights
  uploads/          # transient uploads (ephemeral on Render)
