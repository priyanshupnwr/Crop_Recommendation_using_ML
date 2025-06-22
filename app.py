from flask import Flask, request, render_template
import numpy as np
import pickle

# Load Model and Scalers
model = pickle.load(open('model.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
standard_scaler = pickle.load(open('standscaler.pkl', 'rb'))

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        scaled_features = standard_scaler.transform(minmax_scaler.transform(features))

        prediction = model.predict(scaled_features)[0]
        return render_template('index.html', result=f"The recommended crop is: {prediction.capitalize()}")
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
