# AI Digit Recognition Console App
This small app utilizes Tensorflow Keras library to test couple of neural network models in terms of hand drawn digit recognition

## Configuration
This app comes with 4 prebuilt models: 
- Sequential Primitive Model
- Sequential Light Model
- Sequential Medium Model
- Convoluted Sequential Model

Model library is easily scalable and can be expanded by adding code for model generation in model_creation.py located in the scripts directory.
To generate new models this script has to expanded and executed. It is recommented to change the crude config located in the same file and disable other models from generating again by changing the boolean value for each of them to False

## Usage
There are 2 methods of using the app. By following the menu commands we have an option to:
- provide a path to the image we want to recognize digit on
- use all images saved in assets/images directory

note that all images must have a white background and be exactly 28 by 28 pixels.

Additionally users can viev accuracy and loss of loaded models based on the test samples from Keras library
and viev summary of each model displaying detailed information on data size and network layers structure 

## Libraries Used
- tensorflow (distributed under Apache License 2.0)
- opencv-python (distributed under MIT License)
- matplotlib (distributed under BSD License)
- numpy (https://numpy.org/devdocs/license.html)
