from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and other required files
model = pickle.load(open('Churn_model.pkl', 'rb'))

# Load the trained scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs
    inputs = {
        'CreditScore': float(request.form['CreditScore']),
        'Geography': int(request.form['Geography']),
        'Gender': int(request.form['Gender']),
        'Age': float(request.form['Age']),
        'Tenure': int(request.form['Tenure']),
        'Balance': float(request.form['Balance']),
        'NumOfProducts': int(request.form['NumOfProducts']),
        'IsActiveMember': int(request.form['IsActiveMember']),
        'EstimatedSalary': float(request.form['EstimatedSalary']),
        'Credit_limit': float(request.form['Credit_limit']),
        'Age_Group': int(request.form['Age_Group'])
    }

    # Preprocess the input data
    input_data = pd.DataFrame([inputs])

    # Scale the numerical features using StandardScaler
    numerical_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Credit_limit']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Make predictions using the model
    prediction = model.predict(input_data)

    return redirect(url_for('result', prediction=prediction[0]))
@app.route('/result')
def result():
    prediction = int(request.args.get('prediction'))

    if prediction == 1:
        prediction_text = "This customer will churn"
    else:
        prediction_text = "This customer will not churn"

    return render_template('result.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
