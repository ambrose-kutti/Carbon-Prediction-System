from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load models
rf_model = joblib.load('model/rf_model.pkl')
svr_model = joblib.load('model/svr_model.pkl')
ann_model = tf.keras.models.load_model('model/ann_model.h5')

# Preprocessing pipeline from RF model
preprocessor = rf_model.named_steps['preprocessor']

# Carbon cleaning threshold
CLEANING_THRESHOLD = 250

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    engine_size = float(request.form['engine_size'])
    cylinders = int(request.form['cylinders'])
    fuel_type = request.form['fuel_type']
    fuel_cons = float(request.form['fuel_cons'])

    # Create input DataFrame
    input_df = pd.DataFrame({
        'Engine Size(L)': [engine_size],
        'Cylinders': [cylinders],
        'Fuel Type': [fuel_type],
        'Fuel Consumption Comb (mpg)': [fuel_cons]
    })

    # Preprocess for ANN
    X_processed = preprocessor.transform(input_df)

    # Predictions from each model
    rf_pred = rf_model.predict(input_df)[0]
    svr_pred = svr_model.predict(input_df)[0]
    ann_pred = ann_model.predict(X_processed)[0][0]

    # Average final prediction
    final_prediction = round(np.mean([rf_pred, svr_pred, ann_pred]), 2)
    suggestion = "Needed" if final_prediction > CLEANING_THRESHOLD else "Not Needed"

    # Go to new result page
    return render_template('result.html',
                           prediction=final_prediction,
                           suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)
