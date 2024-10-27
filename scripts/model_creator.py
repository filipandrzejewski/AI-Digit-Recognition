import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from flask import Flask


# Loading training data | testing data
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Using sequential model
model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.Flatten(input_shape=(28, 28, ))) #Flatten layer (changes a grid into a 'flat' string)
model1.add(tf.keras.layers.Dense(128, activation='relu')) #Dense layer (most basic layer, includes units and activation method)
model1.add(tf.keras.layers.Dense(10, activation='softmax')) #Dense output layer with activation method softmax (all neurons combine into 1, after all the result must choose the most confident option [1] and categorized others as 0)

model1.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Flatten(input_shape=(28, 28, )))
model2.add(tf.keras.layers.Dense(128, activation='relu'))
model2.add(tf.keras.layers.Dense(128, activation='relu'))
model2.add(tf.keras.layers.Dense(128, activation='relu'))
model2.add(tf.keras.layers.Dense(128, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.5))
model2.add(tf.keras.layers.Dense(10, activation='softmax'))

model2.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Training model
model1.fit(x_train, y_train, epochs=5)
model2.fit(x_train, y_train, epochs=10)

# with this call fit will execute with following arguments:
#   training data = x_train, y_train
#   epochs = 5 iterations / 10 iterations
#   [ALL BELOW SET AS DEFAULT]
#   batch size = 32 samples per update
#   verbose = 'auto' - number of information displayed in console
#   callbacks = None - no additional functions executing while model training
#   validation_split = 0.0 - amount of training data to be set aside and used for validation only
#   validation_data = None - a separate data set for validation
#   shuffle = True - will shuffle the set before each iteration
#   class/sample_weight = None - no specialized weight for specific classes
#   steps_per_epoch / validation_steps = None - steps defined by the size of the sets (if set is changing in size define custom)
#   validation_batch_size = None - no specialized sie for validation set
#   validation_freq = 1 - validate every iteration (x evaluates every x'th iteration)
#   max_queue_size = 10 batches of data in waiting queue waiting to be processed by model
#   workers = 1 thread working on loading data
#   use_multiprocessing = False - will not spread the process across multiple cores

prediction_probability = model1.predict(x_test)
prediction = np.array([np.argmax(pred) for pred in prediction_probability])

# Display the model performance
print(classification_report(y_test, prediction))

# Saving model
model1.save('../assets/models/digit_recognition_lite.keras')
model2.save('../assets/models/digit_recognition_medium.keras')
