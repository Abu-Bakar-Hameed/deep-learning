from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
from Ann import accuracy_score, precision_score, recall_score, f1_score
 
app = FastAPI(title="Fraud Detection API")

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("fraud_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# =========================
# INPUT MODEL
# =========================
class Transaction(BaseModel):
    features: list

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(transaction: Transaction):

    if len(transaction.features) != 30:
        return {"error": "Expected 30 features"}

    data = np.array(transaction.features).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prob = float(model.predict(data_scaled)[0][0])

    return {
        "fraud_probability": prob,
        "prediction": "Fraud 🚨" if prob > 0.5 else "Legit ✅"
    }

# =========================
# METRICS (STATIC FOR DEMO)
# =========================

@app.get("/metrics")
def metrics():
    return {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1_score": f1_score
    }

