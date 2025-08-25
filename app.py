# -*- coding: utf-8 -*-
"""
Flask backend for Microgel & Cell Annotator (EN)
- /api/detect                 : YOLO inference on uploaded images
- /api/export/yolo            : ZIP with per-image YOLO .txt + classes.txt
- /api/export/yolo/txt        : single YOLO .txt for one image
- /api/export/excel           : (legacy) one Excel for all images (Summary + Boxes)
- /api/export/excel/one       : **one Excel per image** (2 sheets: microgel_id_cell_count, microgel_id_cell_id)
- /api/export/excel/each      : ZIP of Excels (one per image)

Rules for Excel:
- Exclude microgels if more than X% is outside the image (edge crop).
- Exclude microgels if IoU with another microgel >= overlap_iou.
- Cells inside excluded microgels are also excluded.
- Both thresholds are configurable from the UI.
"""

import os
import io
import uuid
import zipfile
from typing import List, Dict, Any, Tuple

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import torch
import pandas as pd

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------- Config ----------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "best.pt"))
CONF_DEFAULT = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IOU_DEFAULT = float(os.environ.get("IOU_THRESHOLD", "0.45"))
ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}

# Default Excel rules (can be overridden per-request)
DEFAULT_OVERLAP_IOU = 0.10           # exclude microgels if IoU >= this
DEFAULT_EDGE_OUTSIDE_PERCENT = 50.0   # exclude microgels if > this % area is outside the image

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/uploads/*": {"origins": "*"}})
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Model ----------------
model = None
class_id_to_name: Dict[int, str] = {}
class_name_to_id: Dict[str, int] = {}

def parse_decimal(s, default: float) -> float:
    try:
        if isinstance(s, str):
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        return default

def load_model():
    global model, class_id_to_name, class_name_to_id
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run `pip install ultralytics` in backend venv.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    names = getattr(model.model, "names", None) or getattr(model, "names", None)
    class_id_to_name.update({int(k): str(v) for k, v in dict(names).items()})
    class_name_to_id.update({v.lower(): int(k) for k, v in class_id_to_name.items()})
    print(f"Loaded model '{MODEL_PATH}'. Classes: {class_id_to_name}")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

def to_yolo_line(class_id: int, x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> str:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

# ---------------- Routes ----------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.post("/api/detect")
def detect():
    if model is None:
        load_model()

    if "files" not in request.files:
        return jsonify({"error": "No files uploaded. Use form-data with 'files'."}), 400

    conf = parse_decimal(request.form.get("conf"), CONF_DEFAULT)
    iou = parse_decimal(request.form.get("iou"), IOU_DEFAULT)

    images_payload: List[Dict[str, Any]] = []
    for f in request.files.getlist("files"):
        if not (f and allowed_file(f.filename)):
            continue
        original_name = secure_filename(f.filename)
        uid = str(uuid.uuid4())[:8]
        stored = f"{uid}_{original_name}"
        save_path = os.path.join(UPLOAD_DIR, stored)
        f.save(save_path)

        res_list = model.predict(
            source=save_path,
            conf=conf, iou=iou,
            max_det=5000, augment=True, agnostic_nms=False,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        r0 = res_list[0]
        img_h, img_w = r0.orig_shape  # (H, W)

        boxes_payload = []
        counts_map: Dict[str, int] = {}
        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            scores = r0.boxes.conf.cpu().numpy()
            classes = r0.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes_xyxy)):
                x, y, w, h = xyxy_to_xywh(boxes_xyxy[i])
                cls_id = int(classes[i]); cls_name = class_id_to_name.get(cls_id, str(cls_id))
                score = float(scores[i]) if i < len(scores) else None
                # clamp
                x = max(0.0, min(x, img_w - 1)); y = max(0.0, min(y, img_h - 1))
                w = max(1.0, min(w, img_w - x)); h = max(1.0, min(h, img_h - y))
                boxes_payload.append({
                    "id": str(uuid.uuid4()),
                    "classId": cls_id, "className": cls_name, "score": score,
                    "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                })
                counts_map[cls_name] = counts_map.get(cls_name, 0) + 1

        images_payload.append({
            "id": str(uuid.uuid4()),
            "filename": original_name,
            "storedFilename": stored,
            "url": f"/uploads/{stored}",
            "width": int(img_w), "height": int(img_h),
            "boxes": boxes_payload, "counts": counts_map
        })

    return jsonify({"classMap": class_id_to_name, "images": images_payload})

# ---------- YOLO exports ----------
@app.post("/api/export/yolo")
def export_yolo():
    payload = request.get_json(force=True, silent=True)
    if not payload or "images" not in payload:
        return jsonify({"error": "Invalid payload. Expect JSON with 'images'."}), 400

    images = payload["images"]
    cmap_in = payload.get("classMap") or class_id_to_name
    cmap: Dict[str, str] = {str(k): str(v) for k, v in (cmap_in.items())}

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        ids_sorted = sorted([int(k) for k in cmap.keys()])
        classes_txt = "\n".join([cmap[str(i)] for i in ids_sorted])
        zf.writestr("classes.txt", classes_txt)

        for item in images:
            filename = item.get("filename") or item.get("storedFilename") or "image"
            stem = os.path.splitext(os.path.basename(filename))[0]
            img_w = int(item["width"]); img_h = int(item["height"])
            lines = []
            for b in item.get("boxes", []):
                cls_id = int(b["classId"])
                x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
                lines.append(to_yolo_line(cls_id, x, y, w, h, img_w, img_h))
            zf.writestr(f"{stem}.txt", "\n".join(lines))

    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="labels_yolo.zip")

@app.post("/api/export/yolo/txt")
def export_yolo_single_txt():
    payload = request.get_json(force=True, silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Invalid payload. Expect JSON with 'image'."}), 400

    item = payload["image"]
    filename = item.get("filename") or item.get("storedFilename") or "image"
    stem = os.path.splitext(os.path.basename(filename))[0]
    img_w = int(item["width"]); img_h = int(item["height"])
    lines = []
    for b in item.get("boxes", []):
        cls_id = int(b["classId"])
        x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        lines.append(to_yolo_line(cls_id, x, y, w, h, img_w, img_h))
    buf = io.BytesIO("\n".join(lines).encode("utf-8")); buf.seek(0)
    # Matches your sample format (class cx cy w h). :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
    return send_file(buf, mimetype="text/plain", as_attachment=True, download_name=f"{stem}.txt")

# ---------- Excel helpers (per-image) ----------
def _iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw*ih
    area_a = max(0, ax2-ax1)*max(0, ay2-ay1)
    area_b = max(0, bx2-bx1)*max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def _build_excel_for_image(item: Dict[str, Any],
                           overlap_iou: float = DEFAULT_OVERLAP_IOU,
                           edge_outside_percent: float = DEFAULT_EDGE_OUTSIDE_PERCENT) -> bytes:
    """
    Build two-sheet Excel for ONE image:
    - Sheet 1: microgel_id_cell_count
    - Sheet 2: microgel_id_cell_id
    Exclusions:
      * microgel overlap IoU >= overlap_iou
      * microgel inside-area ratio < (1 - edge_outside_percent/100)
      * cells inside excluded microgels are also excluded
    """
    img_w = int(item["width"]); img_h = int(item["height"])
    boxes = item.get("boxes", [])
    inside_ratio_thr = 1.0 - float(edge_outside_percent) / 100.0

    # Resolve class ids
    cid_by_name = {}
    for b in boxes:
        if "className" in b and b["className"]:
            cid_by_name[b["className"].lower()] = b["classId"]
    MICROGEL = cid_by_name.get("microgel", 0)
    CELL     = cid_by_name.get("cell", 1)

    # keep boxes (no model filtering here; we rely on user's edits)
    filtered = []
    for b in boxes:
        cls_id = int(b["classId"])
        x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        x1, y1, x2, y2 = x, y, x+w, y+h
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(img_w, x2), min(img_h, y2)
        original_w, original_h = (x2 - x1), (y2 - y1)
        clipped_w, clipped_h   = max(0, x2c - x1c), max(0, y2c - y1c)
        cx, cy = x1 + original_w/2.0, y1 + original_h/2.0
        filtered.append({
            "cls": cls_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "original_w": original_w, "original_h": original_h,
            "clipped_w": clipped_w, "clipped_h": clipped_h,
            "cx": cx, "cy": cy
        })

    micro_idx = [i for i,d in enumerate(filtered) if d["cls"] == MICROGEL]
    cell_idx  = [i for i,d in enumerate(filtered) if d["cls"] == CELL]

    # Overlap exclusion
    overlap_excl = set()
    for i in range(len(micro_idx)):
        for j in range(i+1, len(micro_idx)):
            ia, ib = micro_idx[i], micro_idx[j]
            a = (filtered[ia]["x1"], filtered[ia]["y1"], filtered[ia]["x2"], filtered[ia]["y2"])
            b = (filtered[ib]["x1"], filtered[ib]["y1"], filtered[ib]["x2"], filtered[ib]["y2"])
            if _iou_xyxy(a,b) >= overlap_iou:
                overlap_excl.add(ia); overlap_excl.add(ib)

    # Severe edge-crop exclusion
    severe_excl = set()
    for i in micro_idx:
        d = filtered[i]
        inside_ratio = (d["clipped_w"] * d["clipped_h"]) / (d["original_w"] * d["original_h"] + 1e-9)
        if inside_ratio < inside_ratio_thr:
            severe_excl.add(i)

    valid_micro_idx = [i for i in micro_idx if (i not in overlap_excl and i not in severe_excl)]

    # per-microgel cell count + mapping
    summary_rows, mapping_rows = [], []
    total_cells_in_microgels = 0
    for mg_display_id, i in enumerate(valid_micro_idx, start=1):
        mg = filtered[i]
        mg_x1, mg_y1, mg_x2, mg_y2 = mg["x1"], mg["y1"], mg["x2"], mg["y2"]
        count = 0
        for cell_local_id, j in enumerate(cell_idx, start=1):
            c = filtered[j]
            if mg_x1 <= c["cx"] <= mg_x2 and mg_y1 <= c["cy"] <= mg_y2:
                count += 1
                mapping_rows.append({"microgel_id": mg_display_id, "cell_id": cell_local_id})
        summary_rows.append({"microgel_id": mg_display_id, "cell_count": count})
        total_cells_in_microgels += count

    summary_rows.append({"microgel_id": "Total", "cell_count": total_cells_in_microgels})

    df_summary = pd.DataFrame(summary_rows)
    df_detail  = pd.DataFrame(mapping_rows).sort_values(by=["microgel_id", "cell_id"])

    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="microgel_id_cell_count", index=False)
        df_detail.to_excel(writer,  sheet_name="microgel_id_cell_id",   index=False)
    mem.seek(0)
    return mem.read()

# Legacy: one workbook for all images (kept for compatibility)
@app.post("/api/export/excel")
def export_excel_legacy():
    payload = request.get_json(force=True, silent=True)
    if not payload or "images" not in payload:
        return jsonify({"error": "Invalid payload. Expect JSON with 'images'."}), 400

    images = payload["images"]
    rows = []
    for item in images:
        filename = item.get("filename") or item.get("storedFilename") or "image"
        img_w = int(item["width"]); img_h = int(item["height"])
        for b in item.get("boxes", []):
            cls_name = b.get("className"); cls_id = int(b.get("classId"))
            x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            cx = (x + w/2.0) / img_w; cy = (y + h/2.0) / img_h
            nw = w / img_w; nh = h / img_h
            rows.append({
                "image": filename, "img_w": img_w, "img_h": img_h,
                "class_id": cls_id, "class_name": cls_name, "score": b.get("score"),
                "x": x, "y": y, "w": w, "h": h,
                "cx_norm": cx, "cy_norm": cy, "w_norm": nw, "h_norm": nh,
                "area_px": w*h, "aspect_ratio": (w/(h+1e-6))
            })
    boxes_df = pd.DataFrame(rows)
    if len(rows) == 0:
        summary_df = pd.DataFrame(columns=["image", "class_name", "count"])
    else:
        summary_df = boxes_df.groupby(["image", "class_name"]).size().reset_index(name="count")
        totals = boxes_df.groupby(["class_name"]).size().reset_index(name="count")
        totals.insert(0, "image", "TOTAL")
        summary_df = pd.concat([summary_df, totals], ignore_index=True)

    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        boxes_df.to_excel(writer, index=False, sheet_name="Boxes")
    mem.seek(0)
    return send_file(mem, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="counts_and_boxes.xlsx")

# one Excel per image (supports rules)
@app.post("/api/export/excel/one")
def export_excel_one():
    payload = request.get_json(force=True, silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Invalid payload. Expect JSON with 'image'."}), 400

    item = payload["image"]
    rules = payload.get("rules", {}) or {}
    overlap_iou = float(rules.get("overlap_iou", DEFAULT_OVERLAP_IOU))
    edge_out_pct = float(rules.get("edge_outside_percent", DEFAULT_EDGE_OUTSIDE_PERCENT))

    filename = item.get("filename") or item.get("storedFilename") or "image"
    stem = os.path.splitext(os.path.basename(filename))[0]

    data = _build_excel_for_image(item, overlap_iou=overlap_iou, edge_outside_percent=edge_out_pct)
    buf = io.BytesIO(data); buf.seek(0)
    return send_file(buf, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name=f"microgel_cell_report_{stem}.xlsx")

# ZIP of per-image Excels (supports rules)
@app.post("/api/export/excel/each")
def export_excel_each():
    payload = request.get_json(force=True, silent=True)
    if not payload or "images" not in payload:
        return jsonify({"error": "Invalid payload. Expect JSON with 'images'."}), 400

    images = payload["images"]
    rules = payload.get("rules", {}) or {}
    overlap_iou = float(rules.get("overlap_iou", DEFAULT_OVERLAP_IOU))
    edge_out_pct = float(rules.get("edge_outside_percent", DEFAULT_EDGE_OUTSIDE_PERCENT))

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in images:
            filename = item.get("filename") or item.get("storedFilename") or "image"
            stem = os.path.splitext(os.path.basename(filename))[0]
            data = _build_excel_for_image(item, overlap_iou=overlap_iou, edge_outside_percent=edge_out_pct)
            zf.writestr(f"microgel_cell_report_{stem}.xlsx", data)
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="excel_each.zip")


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
