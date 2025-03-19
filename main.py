from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import joblib  # Import for loading the crop recommendation model
from PIL import Image

app = FastAPI()

# Load the trained models
# modelWeed = load_model("weed_detection_model.h5")
modelPest = load_model("pestIdentification.h5")
modelCrop = joblib.load("CropRecommendetion_RF_Model.pkl")  # Using joblib to load the .pkl file

# Define class names
# class_names_weed = [
#     "Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat",
#     "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed",
#     "Shepherd’s Purse", "Small-flowered Cranesbill", "Sugar beet"
# ]

class_names_pest = [
    "aphids", "armyworm", "beetle", "bollworm", "grasshopper",
    "mites", "mosquito", "sawfly", "stem_borer"
]

crop_mapping = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}

# Reverse the crop mapping to get names from numerical predictions
crop_labels = {v: k for k, v in crop_mapping.items()}

# Define input schema using Pydantic
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


def preprocess_image(img_bytes):
    """Convert image bytes to a preprocessed model input."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Convert to RGB
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# @app.post("/predict-weed")
# async def predict_weed(file: UploadFile = File(...)):
#     try:
#         img_bytes = await file.read()
#         processed_img = preprocess_image(img_bytes)

#         # Get prediction
#         prediction = modelWeed.predict(processed_img)
#         predicted_class = np.argmax(prediction)
#         confidence = float(np.max(prediction))

#         return {"predicted_class": class_names_weed[predicted_class], "confidence": confidence}
#     except Exception as e:
#         return {"error": str(e)}


@app.post("/predict-pest")
async def predict_pest(file: UploadFile = File(...)):  # Changed function name
    try:
        img_bytes = await file.read()
        processed_img = preprocess_image(img_bytes)

        # Get prediction
        prediction = modelPest.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return {"predicted_class": class_names_pest[predicted_class], "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict_best_crops(input_data: CropInput):
    try:
        # Convert input to numpy array
        input_features = np.array([
            input_data.N, input_data.P, input_data.K, 
            input_data.temperature, input_data.humidity, 
            input_data.ph, input_data.rainfall
        ]).reshape(1, -1)

        # Predict probabilities for all crops
        probabilities = modelCrop.predict_proba(input_features)[0]

        # Get top 5 crop indices with highest probabilities
        top_5_indices = np.argsort(probabilities)[::-1][:5]

        # Filter crops with probability > 0%
        top_5_crops = [{"crop": crop_labels[i+1], "probability": round(probabilities[i], 4)}
                    for i in top_5_indices if probabilities[i] > 0]

        return {"recommended_crops": top_5_crops}
    except Exception as e:
        return {"error": str(e)}
