import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,      # Randomly rotate images by 10 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally by 10% of width
    height_shift_range=0.1, # Randomly shift images vertically by 10% of height
    horizontal_flip=False,  # Randomly flip images horizontally (not applicable for MNIST)
)

# Fit the data generator to our training data
datagen.fit(x_train)

# Visualize augmented data
sample_image = x_train[0]
sample_image = sample_image.reshape((1, 28, 28, 1))  # Reshape for ImageDataGenerator

# Create a plot to visualize the augmented images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    augmented_image = datagen.flow(sample_image, batch_size=1).__next__()  # Use __next__()
    plt.imshow(augmented_image[0].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

# Define and compile the model using Input layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with augmented data
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
