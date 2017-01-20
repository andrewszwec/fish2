## AUTHOR: ANDREW SZWEC
## TITLE: FISH IDENTIFIER
## DESCRIPTION: THIS MODEL WILL TAKE AN IMAGE AND RETURN THE COORDS OF THE FISH IN THE IMAGE
## PLAN:        
## LOAD IN ALL IMAGES FROM KAGGLE COMP
## ADD COORDS OF FISH AS LABEL
## SPLIT INTO TRAIN AND TEST
## BUILD MODEL TO IDENTIFY LOCATION OF FISH IN IMAGES
## 
## BASED ON TUTORIAL: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
## TRAINING TIME: ~30min



from __future__ import division, print_function, absolute_import
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from random import randint
np.random.seed(123)  # for reproducibility
import shutil
import h5py

## SPLIT DATA INTO TRAIN AND TEST
## Takes the files out of the train folder and puts them into the test folder randomly 
## for a 30% test set
types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
dir = '/home/andrew/Documents/kaggle/fish/two-part-model/data/train/'
testdir = '/home/andrew/Documents/kaggle/fish/two-part-model/data/test/'

test_size = 0.3

## SPLIT DATA INTO TRAIN AND TEST
#for fish in types:
#    directory = dir+fish+'/'
#    filenames = os.listdir(directory)
#    nrows = len(filenames)
#    nsamples = int(round(nrows * test_size,))
#    d = np.random.choice(filenames, size = nsamples, replace = 0)
#    
#    for movefile in d:
#        shutil.move(dir + fish + '/' + movefile, testdir + fish +'/'+ movefile)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(224,224,3))) # original kernel model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# original kernal model.add(Convolution2D(32, 3, 3))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'],
              learning_rate=0.001)


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/home/andrew/Documents/kaggle/fish/two-part-model/data/train/',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '/home/andrew/Documents/kaggle/fish/two-part-model/data/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

## TRAIN THE MODEL!
model.fit_generator(
        train_generator,
        samples_per_epoch= 2000,
        nb_epoch= 50,
        validation_data=validation_generator,
        nb_val_samples= 800)
## SAVE THE MODEL
model.save_weights('fish_cnn_v002.h5')  # always save your weights after training or during training

## PREDICT ON TEST DATA 
# perform some augmentations on the submission data
generator = datagen.flow_from_directory(
        '/home/andrew/Documents/kaggle/fish/data/test_stg1/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
predictions = model.predict_generator(generator, 1000)

# find the index of the max probability in the array (0 to 7) 
# 0 to 7 corresponds to a class e.g. DOL, LAG

class_codes = np.argmax(predictions, axis=1)

# Now convert class codes to words

dict = {0:'ALB', 1:'BET', 2:'DOL', 3:'LAG', 4:'NoF', 5:'OTHER', 6:'SHARK', 7:'YFT' }

myclass = []
for cls in class_codes:
     myclass.append( dict[cls])
     


## SAVE myclass AND PREDICTIONS
with open('/home/andrew/Documents/kaggle/fish/data/Submission_Classes', 'w') as f:
    for s in myclass:
        f.write(s + '\n')




#predictions.savetxt('/home/andrew/Documents/kaggle/fish/data/Submission_Class_Probabilities', delimiter=',', newline='\n')

#predictions.tofile('/home/andrew/Documents/kaggle/fish/data/Submission_Class_Probabilities', sep=",")

np.save(file='/home/andrew/Documents/kaggle/fish/data/Submission_Class_Probabilities', arr=predictions )



np.load('/home/andrew/Documents/kaggle/fish/data/Submission_Class_Probabilities.npy')




# Loading on Mac
import numpy as np
predictions = np.load('/Users/andrew/Downloads/results/Submission_Class_Probabilities.npy')
# Head
predictions[0:3,:]


import pandas as pd
pred = pd.DataFrame(data=predictions,    # values
              index=range(0,1000),    # 1st column as index
            columns=range(0,8))  # 1st row as the column names


filenames = os.listdir('/Users/andrew/Documents/kaggle/fish/data/test_stg1')

files = pd.DataFrame(data=filenames)

result = pd.concat([files, pred], axis=1)

result.columns = ['image',  'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER',   'SHARK',   'YFT']

# Header: [image ALB BET DOL LAG NoF OTHER   SHARK   YFT]
result.to_csv('/Users/andrew/Downloads/results/fish_submission.csv', index = False)



























