
# 🌾 Crop Recommendation System Using Machine Learning

**Description:**
An AI-powered web application that recommends the most suitable crops based on soil and environmental conditions. The system uses machine learning (Random Forest Classifier) on real agricultural data to help farmers make informed decisions and optimize crop yields.

## Features

* Predicts the best crop for a given soil and environmental parameters.
* Web interface for easy input of Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
* Scales and normalizes input features for accurate predictions.
* Uses a trained **Random Forest Classifier** for robust crop recommendation.

## Dataset

* The dataset used is **“Crop\_recommendation.csv”** from Kaggle.
* Contains features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.
* Target column: `label` (crop name).

## Tech Stack

* **Backend & ML:** Python, Flask, Scikit-learn, Pandas, Numpy
* **Frontend:** HTML
* **Model Persistence:** Pickle
* **Data Preprocessing:** MinMaxScaler, StandardScaler

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/priyanshupnwr/Crop_Recommendation_using_ML.git
   ```
2. Navigate into the project directory:

   ```bash
   cd Crop_Recommendation_using_ML
   ```
3. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure `model.pkl`, `minmaxscaler.pkl`, and `standscaler.pkl` are present in the directory.
   *(These are created using the training script provided in the repo.)*
2. Start the Flask server:

   ```bash
   python app.py
   ```
3. Open your browser and go to:

   ```
   http://127.0.0.1:5000/
   ```
4. Enter soil and environmental parameters (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall) in the form.
5. Receive the recommended crop based on the input data.

## How It Works

1. **Data Preprocessing:**

   * Input features are scaled using `MinMaxScaler` and `StandardScaler` to normalize the data.
2. **Model Prediction:**

   * The Random Forest model predicts the crop based on scaled input features.
3. **Web Interface:**

   * Users interact via a simple HTML form, and results are displayed instantly on the page.

## Project Structure

```
Crop_Recommendation_using_ML/
│
├── app.py               # Flask web app
├── model_training.py    # Script to train Random Forest and save model/scalers
├── Crop_recommendation.csv  # Dataset
├── model.pkl            # Trained Random Forest model
├── minmaxscaler.pkl     # MinMaxScaler
├── standscaler.pkl      # StandardScaler
├── templates/
│   └── index.html       # Web interface
└── requirements.txt     # Python dependencies
```



## Contact

For questions or suggestions, contact **Priyanshu Panwar** at \[[priyanshupanwarrr@gmail.com](mailto:priyanshupanwarrr@gmail.com)].

