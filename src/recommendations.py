# 1. Recommendation logic
def get_recommendation(churn_prob, cluster):

    if churn_prob > 0.75:
        if cluster == 1:
            return "High Risk: Offer 20-30% discount + priority support call"
        elif cluster == 3:
            return "Re-engagement campaign + free trial extension"
        else:
            return "Immediate retention offer + manager follow-up"

    elif churn_prob > 0.4:
        if cluster == 2:
            return "Offer loyalty rewards based on usage"
        else:
            return "Send personalized engagement emails"

    else:
        return "Customer is stable → enroll in loyalty program"


# 2. Load models
import joblib

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")


# 3. Full prediction pipeline
def predict_customer(data):

    scaled_data = scaler.transform([data])

    churn_prob = model.predict_proba(scaled_data)[0][1]
    cluster = kmeans.predict(scaled_data)[0]

    recommendation = get_recommendation(churn_prob, cluster)

    return {
        "churn_probability": float(churn_prob),
        "cluster": int(cluster),
        "recommendation": recommendation
    }
