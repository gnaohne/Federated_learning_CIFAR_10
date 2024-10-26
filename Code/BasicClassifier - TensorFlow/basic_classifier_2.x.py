# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

# You can safely tune these variables
BATCH_SIZE = 32
EPOCHS = 5

# Load dataset as numpy arrays
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Data loaded')
print('Local dataset size: {}'.format(train_images.shape[0]))

# List with class names to see the labels of the images with matplotlib
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a callback to save the model
checkpoint_path = "logs_dir/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq='epoch')

# Define the model using tf.keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten layer
    keras.layers.Dense(128, activation='relu'),  # Dense layer with ReLU
    keras.layers.Dense(10, activation='softmax')  # Output layer with softmax
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model with training data
model.fit(train_images, train_labels, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          validation_split=0.1, 
          callbacks=[cp_callback])

# Evaluate the model
print('--- Begin Evaluation ---')
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy: {:4f}'.format(test_accuracy))

# Make predictions
predicted = model.predict(test_images)

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predicted[i])
    true_label = test_labels[i]
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]), color=color)

plt.show()
