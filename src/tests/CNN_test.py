# Import necessary libraries
import numpy as np
import tensorflow as tf

# Define the CNN model
# This model consists of several layers including convolutional, pooling, flattening, and dense layers. This architecture is commonly used for image classification tasks.
# Define a sequential model that stacks layers to create a convolutional neural network (CNN).
# Define a sequential model that stacks layers to create a convolutional neural network (CNN).
model = tf.keras.models.Sequential([
    # First convolutional layer with ReLU activation
# This layer applies 32 filters of size 3x3 to the input image, using ReLU as the activation function. It helps in extracting features from the input images.
# First convolutional layer with ReLU activation
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer
# This layer reduces the spatial dimensions of the output from the previous layer by taking the maximum value over a 2x2 window. This helps in down-sampling the feature maps.
# Max pooling layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten the output
# This layer converts the 2D matrix output from the previous layer into a 1D vector to feed into the dense layer. It prepares the data for the fully connected layer.
# Flatten the output
    tf.keras.layers.Flatten(),
    # Fully connected layer
# This layer connects every neuron in the previous layer to every neuron in this layer, allowing for complex decision making.
# Fully connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with softmax activation
# This layer outputs the probabilities of each class, summing to 1, for multi-class classification.
# Output layer with softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Using Adam optimizer and sparse categorical crossentropy loss for multi-class classification. Adam is an adaptive learning rate optimization algorithm.
# Compile the model
# Adam optimizer is an adaptive learning rate optimization algorithm that is popular for training deep learning models.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Fitting the model on training data for 5 epochs. This allows the model to learn from the training data.
# Train the model
# The model will learn from the training data for 5 complete passes through the dataset.
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
# Assessing the model's performance on the test dataset. This step evaluates how well the model generalizes to unseen data.
# Evaluate the model
# This step evaluates how well the model generalizes to unseen data.
model.evaluate(x_test, y_test)