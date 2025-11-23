from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

# Model yolunu güvenli hale getir
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")

model = joblib.load(model_path)

app = FastAPI(title="Sleep Health ML API", version="1.0.0")

class SleepData(BaseModel):
    sleep_duration: float
    heart_rate: float

@app.post("/predict")
def predict(data: SleepData):
    features = pd.DataFrame({
        'Sleep Duration': [data.sleep_duration],
        'Heart Rate': [data.heart_rate]
    })
    prediction = model.predict(features)[0]
    return {
        "prediction": prediction,
        "stress_level": round(prediction, 2),
        "status": "High Stress" if prediction > 5 else "Low Stress"
    }

@app.get("/")
def home():
    return {"message": "Sleep Health ML API Calisiyor! (MLOps Level 3)"}