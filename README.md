
# Project Title

TRAFFIC ACCIDENT DETECTOR MODEL

## Accident Detection API 🚦
```This project provides a simple accident detection API using a deep learning model trained with TensorFlow/Keras. The model classifies images as “accident” or “non-accident” and is deployed as a REST API using FastAPI. Deployment is configured for Render via render.yml.```

## Features
Image Classification: Detects accidents in image frames.
REST API: FastAPI backend exposes endpoints for predictions.
Pretrained Model: Uses a Keras model (my_model.keras).
Easy Deployment: Ready-to-deploy on Render with included configuration.
Training Script: train.py for model training with your own dataset.

## Project Structure
```
.
├── main.py             # FastAPI app for prediction
├── train.py            # TensorFlow/Keras training script
├── requirements.txt    # Python dependencies
├── render.yml          # Render deployment configuration
├── my_model.keras      # Trained model (should be present for API to work)
└── dataset/
    ├── train/          # Training images (accident/non-accident)
    └── val/            # Validation images (accident/non-accident)
```
## Setup & Installation
Clone the repository
```
git clone https://github.com/Shrinkhal01/ML-deployment-simple-public.git
cd ML-deployment-simple-public
```
## Install dependencies
```
pip install -r requirements.txt
```

- Place your training and validation images under dataset/train/ and dataset/val/, each with subfolders for accident and non-accident.
- Run:
```
python train.py
```

This will create a model file (e.g., saved_model/my_model_16x16.keras). Rename/move it to my_model.keras in the root directory for use by the API.

## Run the API locally
```
uvicorn main:app --host 0.0.0.0 --port 8009
```

## API Usage
```
Health Check
GET /
```

- Response:
```
{ "status": "running" }
Accident Detection
POST /detect-accident
```
- Body: JSON with a base64-encoded image frame.
```
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA..."
}
```
- Response: Whether an accident was detected.
```
{ "accident": true }
```
- Deployment on Render

    The included render.yml configures Render to:

Install dependencies via : 
 - pip install -r requirements.txt
Start the API with:

 - uvicorn main:app --host 0.0.0.0 --port 8009


Steps:

1. Push your repo to GitHub.
2. Create a new Web Service on Render.
3. Connect your GitHub repo and select render.yml for configuration.
4. Deploy!
- Requirements
```Python 3.8+```

- Notes
1. Ensure my_model.keras is present in the repository root for API predictions.
2. The model expects images resized to 224x224 pixels (handled in main.py).
 
## License
- MIT
