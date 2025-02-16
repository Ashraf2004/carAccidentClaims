
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import uvicorn
import os
import base64
from pymongo import MongoClient
from datetime import datetime
from fastapi.responses import Response
from bson import ObjectId
# Initialize FastAPI app
app = FastAPI()

# Load models and encoders
model = joblib.load("model_amount.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Connect to MongoDB
MONGO_URI = "mongodb+srv://ashraf3:reeha@cluster0.urj35.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client.Claims # Database name
collection = db.pred  # Collection name

def load_and_preprocess_image(img_path):
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
    days_expiry: int = Form(...)):
    try:
        if days_expiry == 0:
            return JSONResponse(content={"predicted_amount": 0.0})
        
        # Save uploaded image temporarily
        image_path = f"temp_{image_file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await image_file.read())

        # Convert image to base64 for storage in MongoDB
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Extract image features
        img_array = load_and_preprocess_image(image_path)
        img_features = base_model.predict(img_array)
        os.remove(image_path)  # Clean up

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
        
        # Save request data, image, and prediction result to MongoDB
        claim_data = {
            "insurance_company": insurance_company,
            "cost": cost,
            "min_coverage": min_coverage,
            "max_coverage": max_coverage,
            "days_expiry": days_expiry,
            "predicted_amount": float(predicted_amount[0]),
            "image": image_base64,
            "timestamp": datetime.utcnow()
        }
        inserted_id = collection.insert_one(claim_data).inserted_id
        
        return JSONResponse(content={"predicted_amount": float(predicted_amount[0]), "claim_id": str(inserted_id)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_image/{claim_id}")
async def get_image(claim_id: str):
    try:
        claim_data = collection.find_one({"_id": ObjectId(claim_id)})  # Convert to ObjectId
        if not claim_data or "image" not in claim_data:
            return JSONResponse(content={"error": "Image not found"}, status_code=404)
        
        image_data = base64.b64decode(claim_data["image"])
        return Response(content=image_data, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def main_page():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
