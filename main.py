from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import base64
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the pretrained model
model = load_model("my_model.keras")

class FrameData(BaseModel):
    frame: str  # base64-encoded image

def predict_from_base64(base64_str):
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)
    return prediction[0][0] > 0.5  # Example threshold for binary classification

@app.post("/detect-accident")
async def detect(data: FrameData):
    try:
        result = predict_from_base64(data.frame)
        return {"accident": bool(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "running"}
