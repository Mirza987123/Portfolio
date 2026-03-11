# Import necessary libraries
import numpy as np
import tensorflow as tf

# Define the CNN model
# This model consists of several layers including convolutional, pooling, flattening, and dense layers.
model = tf.keras.models.Sequential([
    # First convolutional layer with ReLU activation
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten the output
    tf.keras.layers.Flatten(),
    # Fully connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)