import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data for demonstration
# Replace with actual data loading and preprocessing
x_data = np.random.rand(1000, 20)
y_data = np.random.randint(0, 2, size=(1000, 1))

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define a more complex model
model = Sequential([
    Dense(128, activation='relu', input_shape=(20,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model with more epochs and a validation set
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val, verbose=2)
print(f'\nLoss: {loss}, Accuracy: {accuracy}')
