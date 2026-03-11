# Import necessary libraries
import numpy as np
import tensorflow as tf

# Define the CNN model
# This model consists of several layers including convolutional, pooling, flattening, and dense layers.
# Define a sequential model that stacks layers to create a convolutional neural network (CNN).
model = tf.keras.models.Sequential([
    # First convolutional layer with ReLU activation
# This layer applies 32 filters of size 3x3 to the input image, using ReLU as the activation function.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer
# This layer reduces the spatial dimensions of the output from the previous layer by taking the maximum value over a 2x2 window.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten the output
    tf.keras.layers.Flatten(),
    # Fully connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Using Adam optimizer and sparse categorical crossentropy loss for multi-class classification.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Fitting the model on training data for 5 epochs.
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
# Assessing the model's performance on the test dataset.
model.evaluate(x_test, y_test)