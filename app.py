import numpy as np
import pickle
from flask import Flask, request, render_template
import sys # Import sys for flushing stdout

# Initialize the flask app
app = Flask(__name__)

# Load the model
# Make sure 'heart.pkl' is in the same directory as app.py
try:
    model = pickle.load(open('heart.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'heart.pkl' model file not found.")
    print("Please make sure the model file is in the same directory as app.py")
    # In a real app, you might want to handle this more gracefully
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    """
    if model is None:
        return render_template('index.html', 
                               prediction_text="Error: Model is not loaded.",
                               prediction_status="danger")

    try:
        # Get all the feature values from the form
        # OLD LINE: form_values = [float(x) for x in request.form.values()]
        
        # NEW CODE: Get features by name to ensure correct order
        # This order MUST match the order your model was trained on
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Create the feature list in the correct order
        form_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        # --- DEBUGGING ---
        # Print the received values to the terminal
        print("--- NEW PREDICTION ---")
        print(f"Received data: {form_values}")
        sys.stdout.flush() # Force print to appear
        # --- END DEBUGGING ---
        
        # The order of features must match the order your model was trained on.
        # Based on your notebook, the features are:
        # 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        # 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        # There are 13 features.
        
        if len(form_values) != 13:
            return render_template('index.html',
                                   prediction_text=f"Error: Expected 13 features, but got {len(form_values)}.",
                                   prediction_status="danger")

        # Convert features to a NumPy array and reshape for a single prediction
        features = np.array(form_values).reshape(1, -1)
        
        # --- DEBUGGING ---
        print(f"Features sent to model: {features}")
        sys.stdout.flush()
        # --- END DEBUGGING ---

        # Make the prediction
        prediction = model.predict(features)
        
        # --- DEBUGGING ---
        print(f"Model prediction: {prediction}")
        sys.stdout.flush()
        # --- END DEBUGGING ---

        # Interpret the. result
        if prediction[0] == 1:
            result_text = "This person has a High Risk of Heart Disease."
            result_status = "danger"
        else:
            result_text = "This person has a Low Risk of Heart Disease."
            result_status = "safe"
            
        return render_template('index.html', 
                               prediction_text=result_text,
                               prediction_status=result_status)

    except ValueError:
        return render_template('index.html',
                               prediction_text="Error: Please enter valid numbers for all fields.",
                               prediction_status="danger")
    except Exception as e:
        # --- DEBUGGING ---
        print(f"An error occurred: {e}")
        sys.stdout.flush()
        # --- END DEBUGGING ---
        return render_template('index.html',
                               prediction_text=f"An error occurred during prediction: {e}",
                               prediction_status="danger")

if __name__ == "__main__":
    # Get the port from environment variable or default to 5000
    import os
    port = int(os.environ.get("PORT", 5003))
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=port)

