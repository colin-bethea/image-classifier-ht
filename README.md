# Image Classifier
This image classifier -- written in Python, based on Keras + Tensorflow -- is a Convolutional Neural Network.
When (main.py) is run, the program will train on the given data (training_set) - based on the configuration of:
- Epochs (#)
- Steps per Epoch (#)

Once the program is finished training, it will predict whether the test image(s) (configured in main.py) pictures a man or a tree (the combination is not intended to be serious).
However, if you do want to use larger image datasets, different subjects, and so on - the program is versatile, heavily documented, and easy to configure.
If you are new to neural networks, this <a href="https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8" target="_blank">fantastic article</a> by @venkateshtata can be referenced (he uses dogs/cats instead - with a much, much larger dataset).
# Dependencies
- Numpy
- Keras
- Tensorflow
