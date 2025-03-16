from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load the trained model and scaler
try:
    model = joblib.load("xgb_parkinsons_model.pkl")  # Ensure the filename is correct
    scaler = joblib.load("scaler.pkl")  # Load the scaler
except Exception as e:
    print("Error loading model:", e)
    model, scaler = None, None  # Prevent crashing if model load fails

@app.route('/')
def home():
    return render_template("index.html")  # Default landing page

@app.route('/index1')
def index1():
    return render_template("index1.html")  # Page to submit input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from form
        form_values = request.form.values()
        
        # Ensure input is provided
        if not form_values:
            return jsonify({"error": "No input provided!"})

        features = [float(x) for x in form_values]
        final_features = np.array(features).reshape(1, -1)

        # Scale the input if scaler is loaded
        if scaler:
            final_features = scaler.transform(final_features)

        # Make prediction
        if model:
            prediction = model.predict(final_features)[0]
            return render_template("index1.html", prediction=str(prediction))
        else:
            return jsonify({"error": "Model not found!"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
