import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import logging
import webbrowser

# Configure logging - Changed level to DEBUG for more info
logging.basicConfig(filename='app_cancer_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)

# --- Global Variables ---
model = None
scaler = None
feature_names = [ # This order MUST match the training data
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# --- Load Model and Scaler ---
def load_resources():
    global model, scaler
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'models', 'breast_cancer_xgb_model.pkl')
    scaler_path = os.path.join(base_path, 'models', 'scaler_cancer.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ XGBoost model loaded successfully.")
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"❌ Model file not found at {model_path}")
        print(f"❌ ERROR: Model file not found at {model_path}")
        model = None 
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        print(f"❌ ERROR: Could not load model: {e}")
        model = None

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully.")
        logging.info("Scaler loaded successfully.")
    except FileNotFoundError:
        logging.error(f"❌ Scaler file not found at {scaler_path}")
        print(f"❌ ERROR: Scaler file not found at {scaler_path}")
        scaler = None
    except Exception as e:
        logging.error(f"❌ Error loading scaler: {e}")
        print(f"❌ ERROR: Could not load scaler: {e}")
        scaler = None

# --- Flask Routes ---
@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template('index_cancer.html')

@app.route("/api/predict_cancer", methods=['POST'])
def predict_cancer_api():
    """Handles prediction requests."""
    global model, scaler
    logging.debug("Received request for /api/predict_cancer") # Log request entry
    
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        logging.error("Model or Scaler not loaded.")
        return jsonify({'status': 'error', 'message': 'Model or Scaler not loaded. Check server logs.'}), 500
        
    try:
        form_data = request.form.to_dict()
        logging.debug(f"Received form data: {form_data}") # Log received data
        
        # Prepare input data in the correct order and type
        input_values = []
        missing_features = []
        invalid_features = []

        for feature in feature_names:
            value = form_data.get(feature)
            if value is None or value == '':
                 missing_features.append(feature)
                 continue # Collect all missing features first
            try:
                input_values.append(float(value))
            except ValueError:
                 invalid_features.append(feature)
                 continue # Collect all invalid features

        # Return errors if any features are missing or invalid
        if missing_features:
            msg = f'Missing value(s) for feature(s): {", ".join(missing_features)}'
            logging.error(msg)
            return jsonify({'status': 'error', 'message': msg}), 400
        if invalid_features:
            msg = f'Invalid numeric value(s) for feature(s): {", ".join(invalid_features)}'
            logging.error(msg)
            return jsonify({'status': 'error', 'message': msg}), 400
            
        logging.debug(f"Processed input values: {input_values}")

        values_np = np.asarray(input_values).reshape(1, -1)
        logging.debug(f"Numpy array shape before scaling: {values_np.shape}")
        
        # Scale the input data
        logging.debug("Attempting to scale data...")
        values_scaled = scaler.transform(values_np)
        logging.debug(f"Scaled data shape: {values_scaled.shape}")
        
        # Make prediction
        logging.debug("Attempting prediction...")
        prediction = model.predict(values_scaled)[0]
        logging.debug(f"Raw prediction: {prediction}")
        prediction_proba = model.predict_proba(values_scaled)[0] 
        logging.debug(f"Prediction probabilities: {prediction_proba}")

        # Ensure prediction is a standard Python int for JSON serialization
        prediction = int(prediction) 

        # Generate result text based on prediction (0=Benign, 1=Malignant)
        if prediction == 1:
            risk_proba = prediction_proba[1] * 100
            result_text = f"High risk of Breast Cancer detected (Malignant). Confidence: {risk_proba:.1f}%. Please consult a doctor."
        else:
            risk_proba = prediction_proba[0] * 100
            result_text = f"No signs of Breast Cancer detected (Benign). Confidence: {risk_proba:.1f}%. You appear healthy."
            
        logging.info(f"Prediction successful: {result_text}")
        return jsonify({'status': 'success', 'result': result_text})

    except Exception as e:
        # Log the full traceback for detailed debugging
        logging.exception(f"Unhandled exception in /api/predict_cancer:") 
        return jsonify({'status': 'error', 'message': f'Prediction failed due to an internal error. Check logs.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    load_resources() # Load model and scaler when the script starts
    port = 5004 # Use a different port if 5000 is taken
    url = f"http://127.0.0.1:{port}/"
    print(f"\n🚀 Breast Cancer Prediction App is running!")
    print(f"   Model Loaded: {'Yes' if model else 'No'}")
    print(f"   Scaler Loaded: {'Yes' if scaler else 'No'}")
    print(f"   Open this link in your browser: {url}\n")
    
    if model and scaler: # Only open browser if resources loaded
        try:
           webbrowser.open(url)
        except Exception as e:
           print(f"Could not open browser automatically: {e}")
        
    # Set debug=True temporarily for better error messages in console
    # REMEMBER TO SET debug=False for production/sharing
    app.run(debug=True, use_reloader=False, port=port) 

