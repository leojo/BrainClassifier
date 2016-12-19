import nibabel as nib
import sys

imgNumber = str(sys.argv[1])
image = nib.load('../data/set_train/train_'+imgNumber+'.nii')
epi_img_data = image.get_data()
imgShape = epi_img_data.shape
imgHalfX = imgShape[0]/2
imgHalfY = imgShape[1]/2
imgHalfZ = imgShape[2]/2
print imgHalfZ

import matplotlib.pyplot as plt
def show_slices(slices):
    """ Function to display row of image slices """
<<<<<<< HEAD
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        #axes[3+i].hist(slices[i].ravel())
=======
    fig, axes = plt.subplots(1, 2*len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[3+i].hist(slices[i].ravel())
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6


    

#slice_0 = epi_img_data[imgHalfX, 75:153, 74:107, 0]
#slice_1 = epi_img_data[65:109, imgHalfY, 55:110, 0]
#slice_2 = epi_img_data[52:120, 65:150, imgHalfZ, 0]
<<<<<<< HEAD
print imgShape
slice_0 = epi_img_data[imgHalfX, 35:163, 30:158, 0]
slice_1 = epi_img_data[ 20:148, imgHalfY, 30:158, 0]
slice_2 = epi_img_data[ 20:148,  35:163, imgHalfZ, 0]
=======

slice_0 = epi_img_data[imgHalfX, 40:170, 40:150, 0]
slice_1 = epi_img_data[ 40:130, imgHalfY, 40:150, 0]
slice_2 = epi_img_data[ 40:130,  40:170, imgHalfZ, 0]
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6




# This prints the third image with white instead of black zones.
# This is just so we can find an adequate threshold (e.g. 400) for what
# we consider as 'black'.

print "Before"
print slice_2
for i in range(0,len(slice_2)):
	for j in range(0, len(slice_2[i])):
		if(slice_2[i][j] < 450): slice_2[i][j] = 4000

print "After: "
print slice_2


show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  


plt.show()


