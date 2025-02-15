import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import os
import tempfile

app = FastAPI()

# Define the model directory path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")

# Load models
model = joblib.load(os.path.join(MODEL_DIR, "model_amount.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_and_preprocess_image(img_path):
    """Load and preprocess image for ResNet50."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict")
async def predict_amount(
    image_file: UploadFile = File(...), 
    insurance_company: str = Form(...), 
    cost: float = Form(...), 
    min_coverage: float = Form(...), 
    max_coverage: float = Form(...), 
    days_expiry: int = Form(...)
):
    try:
        if days_expiry == 0:
            return JSONResponse(content={"predicted_amount": 0.0})
        
        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(await image_file.read())
            temp_img_path = temp_img.name

        # Extract image features
        img_array = load_and_preprocess_image(temp_img_path)
        img_features = base_model.predict(img_array)
        os.remove(temp_img_path)  # Clean up

        # Process numerical features
        features = np.array([[cost, min_coverage, max_coverage, days_expiry]])
        features_scaled = scaler.transform(features)

        # One-hot encode the insurance company
        insurance_encoded = encoder.transform([[insurance_company]])

        # Merge encoded categorical and numerical features
        final_tabular_features = np.hstack([insurance_encoded, features_scaled])
        final_features = np.concatenate([img_features, final_tabular_features], axis=1)
        
        # Ensure feature dimensions match model input
        expected_features = model.n_features_in_
        if final_features.shape[1] != expected_features:
            raise ValueError(f"Feature shape mismatch, expected: {expected_features}, got {final_features.shape[1]}")
        
        predicted_amount = model.predict(final_features)
        return JSONResponse(content={"predicted_amount": float(predicted_amount[0])})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def main_page():
    """Serve index.html if available."""
    index_path = os.path.join(os.path.dirname(__file__), "../public/index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    return JSONResponse(content={"message": "Index.html not found."}, status_code=404)
