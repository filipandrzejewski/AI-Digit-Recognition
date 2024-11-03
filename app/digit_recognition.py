import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



models = []
models_names = []

models_directory = '../assets/models'
images_directory = '../assets/images'

for model_file in os.listdir(models_directory):
    model_path = os.path.join(models_directory, model_file)
    try:
        model = tf.keras.models.load_model(model_path)
        models.append(model)
        models_names.append(model_file)
    except Exception as e:
        print(f"Could not lead model {model_file}: {e}")

print(f"Loaded {len(models)} models from assets directory")

def recognize_digit(image_path, plotShow = True):
    try:
        loaded_image = cv2.imread(image_path)[:, :, 0]
        img = np.array([cv2.resize(loaded_image, (28, 28))])
        img = np.invert(img)
        if plotShow:
            plt.imshow(img[0], cmap='binary')
            plt.show()
        for index, model in enumerate(models):
            prediction = model.predict(img)
            print(f"{models_names[index]} says - The digit is: {np.argmax(prediction)}")

    except Exception as e:
        print(e)


def recognize_digit_all_assets():
    for image_file in os.listdir(images_directory):
        recognize_digit(os.path.join(images_directory, image_file))
        input()

def display_all_model_statistics():
    data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data.load_data()
    for index, model in enumerate(models):
        print(f"{models_names[index]} : {model.evaluate(x_test,y_test)}")

def display_all_model_summary():
    for index, model in enumerate(models):
        print(f"{models_names[index]} : {model.summary()}")
