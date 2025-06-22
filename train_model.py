import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# Load Dataset
data = pd.read_csv('Crop_recommendation.csv')

# Features and Labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_train_scaled = standard_scaler.fit_transform(X_train_minmax)

X_test_minmax = minmax_scaler.transform(X_test)
X_test_scaled = standard_scaler.transform(X_test_minmax)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Saving the Model and Scalers
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(minmax_scaler, open('minmaxscaler.pkl', 'wb'))
pickle.dump(standard_scaler, open('standscaler.pkl', 'wb'))

print("Model and scalers saved successfully!")
