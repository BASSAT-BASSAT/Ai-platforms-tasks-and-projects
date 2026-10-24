import time
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.config import FRAMEWORK, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

app = FastAPI(title="Brain Tumor MRI Classifier API")

model = None

if FRAMEWORK == "pytorch":
    import torch
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.on_event("startup")
def load_model():
    global model
    if FRAMEWORK == "pytorch":
        import torch
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
    elif FRAMEWORK == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        raise ValueError(f"Unsupported framework: {FRAMEWORK}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "framework": FRAMEWORK,
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG or PNG).")

    start_time = time.time()
    contents = await file.read()
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    if FRAMEWORK == "pytorch":
        import torch
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()
    else:
        img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        probs = model.predict(img_array)[0]

    latency_ms = int((time.time() - start_time) * 1000)
    top_idx = int(np.argmax(probs))
    label = CLASS_NAMES[top_idx]
    confidence = round(float(probs[top_idx]), 2)

    probabilities = {
        name: round(float(prob), 2)
        for name, prob in zip(CLASS_NAMES, probs)
    }

    return {
        "framework": FRAMEWORK,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_ms": latency_ms
    }