import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv('housing_prices.csv')

# Print column names to debug
print("Columns in the dataset:", data.columns)

# Remove any leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Separate features and target
X = data.drop('Price', axis=1)
y = data['Price']

# One-hot encode categorical features and standardize numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Bedrooms', 'Bathrooms', 'SquareFootage', 'Age']),
        ('cat', OneHotEncoder(), ['Location'])
    ])

X = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the feedforward neural network
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=2)
print(f"Mean Squared Error on test data: {mse[1]}")

# Predict prices
predictions = model.predict(X_test)
print("Predicted prices:", predictions.flatten())