from fastapi import FastAPI
import joblib
import numpy as np
import os
import uvicorn

# ----------------------------
# Paths (FIXED - NO HARD CODE)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# ----------------------------
# Load models
# ----------------------------
model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))

# ----------------------------
# App init (ONLY ONCE)
# ----------------------------
app = FastAPI(title="Churn Prediction API")

# ----------------------------
# Recommendation logic
# ----------------------------
def get_recommendation(churn_prob, cluster):

    if churn_prob > 0.75:
        return "High Risk: Offer discount + priority support"

    elif churn_prob > 0.4:
        return "Medium Risk: Send engagement emails"

    else:
        return "Stable Customer → Loyalty program"

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def home():
    return {"message": "API is running successfully 🚀"}

# ----------------------------
# Predict endpoint
# ----------------------------
@app.post("/predict")
def predict(data: dict):

    features = np.array(list(data.values())).reshape(1, -1)

    scaled_data = scaler.transform(features)

    churn_prob = model.predict_proba(scaled_data)[0][1]
    cluster = kmeans.predict(scaled_data)[0]

    recommendation = get_recommendation(churn_prob, cluster)

    return {
        "churn_probability": float(churn_prob),
        "cluster": int(cluster),
        "recommendation": recommendation
    }

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)