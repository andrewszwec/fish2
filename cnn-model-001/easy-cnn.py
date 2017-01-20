"""
Based on tutorial
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.54v56wei4
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np
import os
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import train_test_split
import pandas as pd

## FUNCTIONS 
# load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.find('.jpg') != -1:  # if it contains .jpg
            img = imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images

## DEFINES
# Small 19 record dataset
#dir_ALB = '/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB_small'
#dir_OTHER = '/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/OTHER_small'

# Large dataset
dir_ALB = '/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB/resized'
dir_OTHER = '/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/OTHER/resized'


## Load the data set (replace this with something that loads our JPEGS)
# Load all the ALBs
ALB_images = load_images_from_folder(dir_ALB)
ALB_labels = np.repeat('ALB', len(ALB_images), axis=0)

# Load all the OTHERs
multiply_samples = 1
OTHER_images = load_images_from_folder(dir_OTHER)
OTHER_labels = np.repeat('OTHER', len(OTHER_images), axis=0)

# The ALB pics are 6 times more than the OTHER pics so duplicate OTHERS 6x
#OTHER_images = np.repeat(OTHER_images, multiply_samples, axis=0)

# Union datasets
df = [] 
df.extend(ALB_images); df.extend(OTHER_images)
ALB_images = []; OTHER_images = []
labels = []
labels.extend(ALB_labels); labels.extend(OTHER_labels) 
ALB_labels = []; OTHER_labels = []

# Change labels into 1s and 0s
for i in range(len(labels)):
      # Converting 'ALB' field to float 
      labels[i] = 1.0 if labels[i] == 'ALB' else 0.0

# Cast all image numbers to floats
df = np.asarray(df)
df = df.astype(float)


# Split into train and test
X, X_test, Y, Y_test = train_test_split(df, labels, test_size=0.33, random_state=42)


# Original load data code
# X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

# Shuffle the data (shuffles both arrays by the same offsets)
X, Y = shuffle(X, Y)
X_test, Y_test = shuffle(X_test, Y_test)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='fish-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='fish-classifier')

# Save model when training is complete to a file
model.save("fish-classifier.tfl")
print("Network trained and saved as fish-classifier.tfl!")

## Test network performance

# Predict
prediction = model.predict(X_test)

# Output Scores to CSV
pred_df = pd.DataFrame(prediction, columns=["IS ALB","IS OTHER"])
obs_df = pd.DataFrame(Y_test, columns=["OBSERVATIONS"])
results = pd.concat([pred_df, obs_df], axis=1)
results.to_csv('/Users/andrew/Documents/kaggle/fish/cnn-model-001/results.csv', header=True)


