# Import necessary libraries
import numpy as np
import tensorflow as tf

# Define the CNN model
# This model consists of several layers including convolutional, pooling, flattening, and dense layers. This architecture is commonly used for image classification tasks. # This model is designed to learn features from image data.
# This model consists of several layers including convolutional, pooling, flattening, and dense layers. This architecture is commonly used for image classification tasks. # This model is designed to learn features from image data.
# Define a sequential model that stacks layers to create a convolutional neural network (CNN). # This model will process image data and learn features.
# This model will be used to process image data and learn features.
# Define a sequential model that stacks layers to create a convolutional neural network (CNN). # This model will process image data and learn features.
# This model will be used to process image data and learn features.
model = tf.keras.models.Sequential([
    # First convolutional layer with ReLU activation
# This layer applies 32 filters of size 3x3 to the input image, using ReLU as the activation function. It helps in extracting features from the input images. # This is the first convolutional layer.
# This layer applies 32 filters of size 3x3 to the input image, using ReLU as the activation function. It helps in extracting features from the input images. # This is the first convolutional layer.
# First convolutional layer with ReLU activation
# This layer applies 32 filters of size 3x3 to the input image, using ReLU as the activation function. It helps in extracting features from the input images. # This is the first convolutional layer.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer
# This layer reduces the spatial dimensions of the output from the previous layer by taking the maximum value over a 2x2 window. This helps in down-sampling the feature maps. # This is the max pooling layer.
# This layer reduces the spatial dimensions of the output from the previous layer by taking the maximum value over a 2x2 window. This helps in down-sampling the feature maps. # This is the max pooling layer.
# Max pooling layer
# This layer reduces the spatial dimensions of the output from the previous layer by taking the maximum value over a 2x2 window. This helps in down-sampling the feature maps. # This is the max pooling layer.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten the output
# This layer converts the 2D matrix output from the previous layer into a 1D vector to feed into the dense layer. It prepares the data for the fully connected layer. # Flattening the output for the dense layer.
# This layer converts the 2D matrix output from the previous layer into a 1D vector to feed into the dense layer. It prepares the data for the fully connected layer. # Flattening the output for the dense layer.
# Flatten the output
# This layer converts the 2D matrix output from the previous layer into a 1D vector to feed into the dense layer. It prepares the data for the fully connected layer. # Flattening the output for the dense layer.
    tf.keras.layers.Flatten(),
    # Fully connected layer
# This layer connects every neuron in the previous layer to every neuron in this layer, allowing for complex decision making.
# This layer connects every neuron in the previous layer to every neuron in this layer, allowing for complex decision making.
# Fully connected layer
# This layer connects every neuron in the previous layer to every neuron in this layer, allowing for complex decision making.
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with softmax activation
# This layer outputs the probabilities of each class, summing to 1, for multi-class classification.
# This layer outputs the probabilities of each class, summing to 1, for multi-class classification.
# Output layer with softmax activation
# This layer outputs the probabilities of each class, summing to 1, for multi-class classification.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# Using Adam optimizer and sparse categorical crossentropy loss for multi-class classification. Adam is an adaptive learning rate optimization algorithm. # Compiling the model with optimizer and loss function.
# Using Adam optimizer and sparse categorical crossentropy loss for multi-class classification. Adam is an adaptive learning rate optimization algorithm. # Compiling the model with optimizer and loss function.
# Compile the model
# Using Adam optimizer and sparse categorical crossentropy loss for multi-class classification. Adam is an adaptive learning rate optimization algorithm. # Compiling the model with optimizer and loss function.
# Adam optimizer is an adaptive learning rate optimization algorithm that is popular for training deep learning models.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Fitting the model on training data for 5 epochs. This allows the model to learn from the training data. # Training the model.
# Fitting the model on training data for 5 epochs. This allows the model to learn from the training data. # Training the model.
# Train the model
# Fitting the model on training data for 5 epochs. This allows the model to learn from the training data. # Training the model.
# The model will learn from the training data for 5 complete passes through the dataset.
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
# Assessing the model's performance on the test dataset. This step evaluates how well the model generalizes to unseen data. # Evaluating the model's performance.
# Assessing the model's performance on the test dataset. This step evaluates how well the model generalizes to unseen data. # Evaluating the model's performance.
# Evaluate the model
# Assessing the model's performance on the test dataset. This step evaluates how well the model generalizes to unseen data. # Evaluating the model's performance.
# This step evaluates how well the model generalizes to unseen data.
model.evaluate(x_test, y_test)