# 0 -- General Package Import
import numpy as np
# 0 -- Keras Import
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# 1 -- Classify neural network as sequence-based
classifier = Sequential()

# 2 -- Add a convolution layer (Conv2D), with 32 filters (3x3 size),
    # Input shape (64x64 resolution, 3 = rgb), relu is a rectifier function

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# 3 -- Add pooling layer to classifier object, 2x2 matrix = minimum pixel loss,
    # Reduced complexity w.out reducing performance
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 4 -- Convert all pooled images into continuous vector (w. flattening)
classifier.add(Flatten())

# 5 -- Create a fully connected layer, connect post-flattening nodes,
    # Between input/out --> makes hidden layer
    # Units = number of nodes present --> between amt of input and output nodes
classifier.add(Dense(units = 128, activation = "relu"))

# 6 -- Initialize output layer, contains only one node (units),
    # Gives a binary output (human or tree)
classifier.add(Dense(units = 1, activation = "sigmoid"))

# 7 -- Compile network, select a gradient descent algorithm (optimizer),
    # Loss function (entropy), and performance metric
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# 8 -- Preprocess/augment the training/testing sets of images to prevent overfitting
    # Various operations flip, re-orient, resize photos to create synthetic data
    # This creates better, more consistent testing accuracy (better machine understanding)
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

# 9 -- Fit the data to the CNN model, steps per epoch = number of training images
    # An epoch is equal to one training step (ie: 25 epochs = 25 training sessions)
classifier.fit_generator(training_set,
steps_per_epoch = 10,
epochs = 5,
validation_data = test_set,
validation_steps = 2000)

# 10 -- Train the data set on human_or_tree_1.jpg @ 64px resolution
    # A prediction of 1 = tree, 0 = human
test_image = image.load_img('test_set/h_or_t.1.jpeg', target_size = (64, 64))
# You can replace test image with entire array of test images or your own images
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'human'
else:
    prediction = 'tree'

print(prediction)
