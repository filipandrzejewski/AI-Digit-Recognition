import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

models = {}
models_directory = '../assets/models'

for filename in os.listdir(models_directory):
    model_path = os.path.join(models_directory, filename)
    try:
        model = keras.models.load_model(model_path)
        models[filename] = model
    except Exception as e:
        print(f"Could not lead model {filename}: {e}")

test_image = 'test3.png'
try:
    img = np.array([cv2.imread(f"../assets/images/{test_image}")[:,:,0]])
    img = np.invert(img)
    plt.imshow(img[0], cmap='binary')
    plt.show()
    for model in models:
        prediction = models[model].predict(img)
        print(f"{model} - The digit is: {np.argmax(prediction)}")

except Exception as e:
    print(e)