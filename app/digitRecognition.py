import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# prediction_probability = model1.predict(x_test)
# prediction = np.array([np.argmax(pred) for pred in prediction_probability])

# # Display the model performance
# print(classification_report(y_test, prediction))

models = {}
models_directory = '../assets/models'
images_directory = '../assets/images'

for model_file in os.listdir(models_directory):
    model_path = os.path.join(models_directory, model_file)
    try:
        model = keras.models.load_model(model_path)
        models[model_file] = model
    except Exception as e:
        print(f"Could not lead model {model_file}: {e}")

for image_file in os.listdir(images_directory):
    try:
        image_path = os.path.join(images_directory, image_file)
        img = np.array([cv2.imread(image_path)[:,:,0]])
        img = np.invert(img)
        plt.imshow(img[0], cmap='binary')
        plt.show()
        for model in models:
            prediction = models[model].predict(img)
            print(f"{model} says - The digit is: {np.argmax(prediction)}")

    except Exception as e:
        print(e)