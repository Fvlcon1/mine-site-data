from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "best.pt")
    model = YOLO(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type in ["image/jpeg", "image/png"]:
        return Image.open(BytesIO(request_body)).convert("RGB")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")



def predict_fn(input_data, model):
    results = model(input_data)
    return results[0]


def output_fn(prediction, response_content_type):
    buffer = BytesIO()
    result_image = prediction.plot()
    Image.fromarray(result_image).save(buffer, format="PNG")
    return buffer.getvalue()
