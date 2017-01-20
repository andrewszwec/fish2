## Load the Image Files and reduce size to 32x32 pixels

#from scipy.misc import imread, imresize, imsave
import numpy as np
import os
import sys
from PIL import Image


# Automated version
# dir = sys.argv
dir = r"/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/WIP/"
suffix = r"YFT/"

# this step should really be batched
def get_file_names(folder):
    filenames = []
    for filename in os.listdir(folder):
        if filename.find('.jpg') != -1:  # if it contains .jpg
            filenames.append(filename)
    return filenames

filenames = get_file_names(dir+suffix)



for fn in filenames:
    # Load the original image:
    img = Image.open(dir+suffix+fn)
    img2 = img.crop((0, 0, 100, 100))  # now crop it to the correct dims
    img2.save("/Users/andrew/Documents/kaggle/fish/cnn-model-001/data/cropped/"+fn)









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


