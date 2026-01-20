from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained wine cultivar model
model = load_model("wine_cultivar_model.h5")

# Load scaler used during training
scaler = joblib.load("scaler.pkl")

# Feature names (13 numeric features in sklearn wine dataset)
feature_labels = [
    "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins",
    "Color Intensity", "Hue", "OD280/OD315 of Diluted Wines", "Proline"
]

@app.route('/')
def home():
    return render_template('index.html', feature_labels=feature_labels)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        features = []
        for label in feature_labels:
            value = float(request.form[label])
            features.append(value)

        features = np.array(features).reshape(1, -1)

        # Scale features
        features = scaler.transform(features)

        # Predict class probabilities
        prediction_probs = model.predict(features)[0]
        predicted_class = np.argmax(prediction_probs)

        # Map to wine cultivar names (optional)
        cultivar_mapping = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
        result = cultivar_mapping.get(predicted_class, "Unknown")

        return render_template(
            'index.html',
            feature_labels=feature_labels,
            prediction_text=f"Predicted Wine Cultivar: {result}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            feature_labels=feature_labels,
            prediction_text="Error in prediction. Check input values."
        )

if __name__ == "__main__":
    app.run(debug=True)
