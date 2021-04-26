SETUP:
For this model you need tensorflow 2.4.1 and python 3.7

RUNNING A TEST:
To make a prediction on an image follow these steps:
Your image should be a face and should be cropped to focus on the face.
ensure your image is 200x200px, and named test.jpg
ensure your image, multiMobileModelTest.py and, 'model_100' are all in the same folder
run MultiMobileModelTest.py

TEST IMAGES AND FEATURES:
As described before, the test inputs are 200x200px images that are cropped on faces.
there isn't really meaning in the example data beyond the color of the pixels.
rather than modify existing example data, take new cropped pictures of faces for evaluation

FILES:
Mobilenetv2.py is the file that describes the model

the folder model_100 is the saved finished model

example images for classification can be found in "dataset"

MultiMobileModleTest.py is the test program for evaluating a single image

notebook.ipynb were the training programs


