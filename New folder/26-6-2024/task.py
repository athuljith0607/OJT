import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Sample dataset
data = {
    'age': [55, 60, 45, 50, 65],
    'gender': ['male', 'female', 'male', 'female', 'male'],
    'cholesterol': [220, 180, 190, 200, 230],
    'bp': [140, 130, 110, 120, 150],
    'smoking': ['yes', 'no', 'yes', 'no', 'yes'],
    'diabetes': ['no', 'yes', 'no', 'no', 'yes'],
    'exercise': ['yes', 'no', 'yes', 'yes', 'no'],
    'heart_attack': [1, 1, 0, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing: Convert categorical variables to numerical
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
df['smoking'] = df['smoking'].map({'yes': 1, 'no': 0})
df['diabetes'] = df['diabetes'].map({'yes': 1, 'no': 0})
df['exercise'] = df['exercise'].map({'yes': 1, 'no': 0})

# Split data into features and target variable
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
