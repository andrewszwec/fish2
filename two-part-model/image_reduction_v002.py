## Load the Image Files and reduce size to 32x32 pixels

#from scipy.misc import imread, imresize, imsave
import os
import sys
from PIL import Image

# DTT UBUNTU
types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
dir = '/home/andrew/Documents/kaggle/fish/data/train/'
outdir = '/home/andrew/Documents/kaggle/fish/two-part-model/data/'

for fishtype in types:
    directory = dir+fishtype+'/'
    outdirectory = outdir+fishtype+'/'

    width = 112
    height = 112

    # Iterate through every image given in the directory argument and resize it.
    for image in os.listdir(directory):
        print('Resizing image ' + image)
        
        # Open the image file.
        img = Image.open(os.path.join(directory, image))
        
        # Resize it.
        img = img.resize((width, height), Image.BILINEAR)
        
        # Save it back to disk.
        img.save(os.path.join(outdirectory, 'resized-' + image))



print('Batch processing complete.')

