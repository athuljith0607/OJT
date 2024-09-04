import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build an ANN Model
ann_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),  # Added to the model
    Dense(64, activation='relu'),   # Added to the model
    Dense(10, activation='softmax') # Added to the model
])

# Compile the model
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train ANN model
ann_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Build the model with KNN
x_train_knn = x_train.reshape(-1, 28*28)
x_test_knn = x_test.reshape(-1, 28*28)

# Standardization of the data
scaler = StandardScaler()
x_train_knn = scaler.fit_transform(x_train_knn)
x_test_knn = scaler.transform(x_test_knn)

# Create the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model
knn_model.fit(x_train_knn, y_train)

# Evaluate the KNN model
knn_accuracy = knn_model.score(x_test_knn, y_test)
print(f"KNN accuracy: {knn_accuracy * 100:.2f}%")

# ANN Predictions
ann_predictions = ann_model.predict(x_test)

# Plot predictions for the first 5 test images
for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {np.argmax(y_test[i])}, Predicted: {np.argmax(ann_predictions[i])}")
    plt.show()