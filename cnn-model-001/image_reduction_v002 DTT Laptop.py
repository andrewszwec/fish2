## Load the Image Files and reduce size to 32x32 pixels

from scipy.misc import imread, imresize, imsave
import numpy as np
#import matplotlib.pyplot as plt
import os
import sys

# Automated version
# dir = sys.argv
# MAC
#dir = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/DOL/"
# DTT UBUNTU
dir = r"/home/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB/"
dir = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/YFT/"

# this step should really be batched
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.find('.jpg') != -1:  # if it contains .jpg
            img = imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


images, filenames = load_images_from_folder(dir)


# fix the file names to add "resized"
new_file_names = []
for file in filenames:
    new_file_names.append( file.replace(".jpg", "_resized.jpg") )

# Make folder for saving
if not os.path.exists(dir+'resized/'):
    os.makedirs(dir+'resized/')

# used when putting into an array
#images_resized=[]
for i in range(0,len(images)):
    im = imresize(images[i], [32,32], interp='bilinear', mode=None )
    # used when putting into an array
    #images_resized.append(im)
    # saving to disk
    imsave(dir+'resized/'+new_file_names[i], im)


quit()






# now taken care of above
# save resized files to disk
#k=0
#for filename in new_file_names:
#    imsave(dir+'resized/'+filename, images_resized[k])
#    k=k+1


# Show the images on screen
#plt.figure()
#plt.imshow(images[5] )
#plt.figure()
#plt.imshow(images_resized[5])
#plt.show()




# WORKING CODE
#f = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB/img_00003.jpg"
#im0 = imread(f)
#f = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB/img_00010.jpg"
#im1 = imread(f)
#f = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB/img_00012.jpg"
#im2 = imread(f)
#f = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/ALB/img_00015.jpg"
#im3 = imread(f)

#im0_new = imresize(im0, [32,32], interp='bilinear', mode=None )


