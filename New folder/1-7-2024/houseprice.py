import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load our data
df = pd.read_csv('housing_price.csv')

# Split the dataset into features (X) and target (y)
X = df[['size', 'bedrooms']].values
Y = df['price'].values

# Initiate or define our model
model = LinearRegression()

# Define our cross-validation method which is KFold
kf = KFold(n_splits=5)

mae_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # Train the model with the training set
    model.fit(X_train, Y_train)
    
    # Predict the test set
    Y_pred = model.predict(X_test)
    
    # Calculate the mean absolute error for this fold
    mae = mean_absolute_error(Y_test, Y_pred)
    mae_scores.append(mae)

# Calculate the average mean absolute error across all folds
average_mae = np.mean(mae_scores)
print(f"Average Mean Absolute Error: {average_mae}")

