import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



models = {}
models_directory = '../assets/models'
images_directory = '../assets/images'

for model_file in os.listdir(models_directory):
    model_path = os.path.join(models_directory, model_file)
    try:
        model = tf.keras.models.load_model(model_path)
        models[model_file] = model
    except Exception as e:
        print(f"Could not lead model {model_file}: {e}")

print(f"Loaded {len(models)} models from assets directory")

def DigitRecognition(image_path):
    try:
        img = np.array([cv2.imread(image_path)[:, :, 0]])
        img = np.invert(img)
        plt.imshow(img[0], cmap='binary')
        plt.show()
        for model in models:
            prediction = models[model].predict(img)
            print(f"{model} says - The digit is: {np.argmax(prediction)}")

    except Exception as e:
        print(e)

def DigitRecognitionFromAssets():
    for image_file in os.listdir(images_directory):
        DigitRecognition(os.path.join(images_directory, image_file))
        input()

def DisplayAllModelStatistics():
    data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data.load_data()
    for model in models:
        print(f"{model} : {models[model].evaluate(x_test,y_test)}")

def DisplayAllModelSummary():
    for model in models:
        print(f"{model} : {models[model].summary()}")

while True:
    print("\nDigit Recognition App - option enter")
    print("1. Recognition on provided image path")
    print("2. Recognition on all images loaded in assets directory")
    print("3. Display model performance statistics for currently loaded models")
    print("4. Display model summary for currently loaded models")
    print("5. Exit\n")

    choice = input("Enter your choice: ")

    if choice == '1':
        image_path = input("Enter image path: ")
        if os.path.isfile(image_path):
            DigitRecognition(image_path)
        else:
            print("Invalid image path")

    elif choice == '2':
        DigitRecognitionFromAssets()

    elif choice == '3':
        DisplayAllModelStatistics()

    elif choice == '4':
        DisplayAllModelSummary()

    elif choice == '5':
        break

    else:
        print("Invalid choice")







