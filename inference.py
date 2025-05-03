from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import torch

app = FastAPI(title="Mining Site Segmentation API")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("mine.pt")
model.to(device)

NEON_GOLD = (0, 215, 255)  

@app.post("/segment/")
async def segment_image(
    file: UploadFile = File(...),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Transparency level of mask (0.0-1.0)"),
    border_thickness: int = Query(2, ge=0, le=10, description="Thickness of mask border"),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold for detection")
):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    original_img = img.copy()

    results = model(img, conf=conf_threshold)[0]
    overlay = np.zeros_like(img, dtype=np.uint8)

    for mask in results.masks.data:
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8)

        colored_mask = np.zeros_like(img, dtype=np.uint8)
        colored_mask[mask_resized == 1] = NEON_GOLD

        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 1.0, 0)

        if border_thickness > 0:
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, NEON_GOLD, border_thickness)

    result = cv2.addWeighted(original_img, 1.0, overlay, alpha, 0)

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"{results.names[int(box.cls[0])]} {conf:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/segment/download/")
async def segment_image_download(
    file: UploadFile = File(...),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    border_thickness: int = Query(2, ge=0, le=10),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0)
):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    original_img = img.copy()

    results = model(img, conf=conf_threshold)[0]
    overlay = np.zeros_like(img, dtype=np.uint8)

    for mask in results.masks.data:
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8)

        colored_mask = np.zeros_like(img, dtype=np.uint8)
        colored_mask[mask_resized == 1] = NEON_GOLD

        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 1.0, 0)

        if border_thickness > 0:
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, NEON_GOLD, border_thickness)

    result = cv2.addWeighted(original_img, 1.0, overlay, alpha, 0)

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"{results.names[int(box.cls[0])]} {conf:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    _, buffer = cv2.imencode(".jpg", result)

    headers = {
        "Content-Disposition": f"attachment; filename=segmented_mining_site.jpg"
    }

    return StreamingResponse(
        BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers=headers
    )

@app.get("/")
async def root():
    return {
        "name": "Mining Site Segmentation API",
        "endpoints": [
            "/segment/ - Process and view an image",
            "/segment/download/ - Process and download an image"
        ],
        "version": "First API Test (with GPU + fixed mask resizing)"
    }
