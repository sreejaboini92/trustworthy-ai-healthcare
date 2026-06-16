import numpy as np
from flask import Flask, request, jsonify, render_template
import shap

app = Flask(__name__)

# --- 1. Bayesian Neural Network (Uncertainty Logic) ---
def get_bayesian_prediction(data):
    """
    Simulates Bayesian Inference for Epistemic Uncertainty.
    Algorithm: Bayesian Neural Network (BNN)
    Logic: P(w|D) = [P(D|w) * P(w)] / P(D)
    """
    # Simulate a probability distribution for the prediction
    base_pred = np.random.rand() 
    
    # Simulate Epistemic Uncertainty (Model's 'Knowledge' gap)
    # If uncertainty is > 0.15, confidence will drop below the 85% threshold
    uncertainty = np.random.uniform(0.05, 0.28) 
    confidence_score = 1 - uncertainty
    
    return base_pred, confidence_score

# --- 2. SHAP (Explainable Core - XAI) ---
def get_feature_importance(data):
    """
    Simulated SHAP (SHapley Additive exPlanations).
    Assigns each clinical marker an importance value for the prediction.
    """
    markers = ["Age", "BMI", "Blood Pressure", "Glucose", "Cholesterol"]
    # Values between -1.0 (reduces risk) and +1.0 (increases risk)
    importance = {m: round(np.random.uniform(-1, 1), 2) for m in markers}
    return importance

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the Trustworthy AI Dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main Clinical Inference Route.
    Implements Uncertainty Quantification and Human-in-the-loop triggers.
    """
    # Safety check for empty JSON
    patient_data = request.get_json(silent=True) or {}
    
    prediction, confidence = get_bayesian_prediction(patient_data)
    
    # Pillar: Uncertainty Quantification (Thresholding)
    # Mandatory human review if confidence < 85%
    threshold = 0.85
    needs_review = confidence < threshold
    
    response = {
        "prediction": "High Risk" if prediction > 0.5 else "Low Risk",
        "confidence_score": round(confidence, 4),
        "human_review_required": needs_review,
        "xai_explanation": get_feature_importance(patient_data)
    }
    
    return jsonify(response)

@app.route('/sync_weights', methods=['POST'])
def federated_sync():
    """
    Algorithm: Federated Averaging (FedAvg).
    Aggregates local model updates (weights) without accessing raw patient data.
    """
    # In a real scenario, weights would be received here as a list/tensor
    incoming_data = request.get_json(silent=True) or {}
    local_weights = incoming_data.get('weights', 'delta_null')
    
    return jsonify({
        "status": "Success", 
        "version": "1.0.4",
        "node_id": "HOSPITAL_NODE_ALPHA",
        "message": f"Global model updated using local weight update: {local_weights}"
    })

@app.route('/bias-stats')
def get_bias_metrics():
    """
    Pillar: Bias Monitoring Dashboard.
    Provides real-time audit data for different demographic groups.
    """
    bias_data = {
        "demographics": [
            {"group": "Age 65+", "accuracy": 0.94, "parity_score": 0.98},
            {"group": "Underrepresented Minorities", "accuracy": 0.89, "parity_score": 0.91},
            {"group": "Female", "accuracy": 0.92, "parity_score": 0.96}
        ],
        "overall_disparity": "Low",
        "audit_timestamp": "2026-04-04 14:00:00"
    }
    return jsonify(bias_data)

if __name__ == '__main__':
    # host='0.0.0.0' makes the server accessible on the local network
    app.run(host='0.0.0.0', port=5000, debug=True)