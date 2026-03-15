import joblib
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Annotated
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API to predict Heart Disease probability"
)


# Model Loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_disease_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# Input Schema
class PatientDetails(BaseModel):
    age:      Annotated[int,   Field(gt=0,            description="Age must be a positive integer")]
    sex:      Annotated[int,   Field(ge=0, le=1,      description="Sex must be 0 (female) or 1 (male)")]
    cp:       Annotated[int,   Field(ge=0, le=3,      description="Chest pain type (0–3)")]
    trestbps: Annotated[int,   Field(gt=0,            description="Resting blood pressure (positive integer)")]
    chol:     Annotated[int,   Field(gt=0,            description="Serum cholesterol (positive integer)")]
    fbs:      Annotated[int,   Field(ge=0, le=1,      description="Fasting blood sugar: 0 (false) or 1 (true)")]
    restecg:  Annotated[int,   Field(ge=0, le=2,      description="Resting ECG results (0–2)")]
    thalach:  Annotated[int,   Field(gt=0,            description="Max heart rate achieved (positive integer)")]
    exang:    Annotated[int,   Field(ge=0, le=1,      description="Exercise induced angina: 0 (no) or 1 (yes)")]
    oldpeak:  Annotated[float, Field(ge=0,            description="ST depression (non-negative float)")]
    slope:    Annotated[int,   Field(ge=0, le=2,      description="Slope of peak exercise ST segment (0–2)")]
    ca:       Annotated[int,   Field(ge=0, le=3,      description="Major vessels colored by fluoroscopy (0–3)")]
    thal:     Annotated[int,   Field(ge=0, le=3,      description="Thalassemia (0–3)")]


# Output Schema
class PredictionResponse(BaseModel):
    heart_disease_prediction: int
    disease_probability: float
    result: str

# Main Page
@app.get("/", response_class=HTMLResponse)
def greet():
    return """
        <h1>🫀 Heart Disease Prediction API</h1>
        <p>Visit <a href="/docs">/docs</a> to test the API</p>
        <p>Visit <a href="/health">/health</a> to check model status</p>
    """

# Health Check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# Prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientDetails):
    try:
        df = pd.DataFrame([patient.model_dump()])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return PredictionResponse(
            heart_disease_prediction=int(prediction),
            disease_probability=round(float(probability), 4),
            result="Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )