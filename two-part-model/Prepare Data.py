
from __future__ import division, print_function, absolute_import
import numpy as np
import os
from scipy.misc import imread, imresize, imsave
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
np.random.seed(123)  # for reproducibility
import pickle

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

## LOAD IMAGES
types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
codes = [0,1,2,3,4,5,6,7]
directory = '/home/andrew/Documents/kaggle/fish/two-part-model/data/'

i = 0
df = [] 
labels = []
for fish in types:
    images = load_images_from_folder(directory+fish)
    label = np.repeat(codes[i], len(images), axis=0)
    i = i + 1
    print(i)
    df.append(images)       # Append images
    labels.append(label)   # Apend labels
    images = []     # Clear array
    label = []     # Clear Array

## END LOAD IMAGES
## RETURNS df (images) and labels

# Save Labels to HDD
#with open(directory+'labels', 'wb') as f:
#    pickle.dump(labels, f)
    
# Save Image Array to HDD (306MB)
#with open(directory+'raw_images', 'wb') as f:
#    pickle.dump(df, f)

# Load images from disk
with open(directory+'raw_images', 'r') as f:
    df = [line.rstrip('\n') for line in f]

# Load labels from disk
with open(directory+'labels', 'r') as f:
    labels = [line.rstrip('\n') for line in f]


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

