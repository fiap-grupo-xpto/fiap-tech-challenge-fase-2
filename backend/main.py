from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pathlib import Path
import numpy as np
import aiofiles
from tensorflow import keras
import tensorflow as tf

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
IMG_SIZE = (224,224)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model after the process has started to avoid pre-fork initialization issues
    model_path = Path(__file__).parent / "model" / "mobileNetV2.keras"
    app.state.model = keras.models.load_model(str(model_path))
    yield
    # Clean up the ML model and release resources
    app.state.model = None


app = FastAPI(debug=True, lifespan=lifespan)


@app.post("/analyze-images")
async def analyze_images(files: List[UploadFile] = File(...), probability_threshold: float = Form(...)):
    results = []
    print(f"Received {len(files)} files for analysis with threshold {probability_threshold}")
    
    for file in files:
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Invalid file type. Only ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') are allowed"
            })
            continue
            
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        try:
            content = await file.read()
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

            # Load image, preprocess and predict with the model
            img_data = tf.io.read_file(str(file_path))
            img = tf.image.decode_image(img_data, channels=3, expand_animations=False)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            image_array = np.expand_dims(img.numpy(), axis=0)  # batch dimension

            model = getattr(app.state, "model", None)
            if model is None:
                # Fallback: load on demand (e.g., if lifespan didn't run)
                model_path = Path(__file__).parent / "model" / "mobileNetV2.keras"
                model = keras.models.load_model(str(model_path))
                app.state.model = model

            prediction = model.predict(image_array)
            probability = float(prediction[0][0])
            has_disease = probability >= probability_threshold

            results.append({
                "filename": file.filename,
                "status": "success",
                "file_path": str(file_path),
                "disease_detected": has_disease,
                "probability": probability
            })
            
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            
    return {"results": results}
