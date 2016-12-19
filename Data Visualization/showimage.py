import nibabel as nib
import sys
<<<<<<< HEAD
import copy
=======
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6

imgNumber = str(sys.argv[1])
image = nib.load('../data/set_train/train_'+imgNumber+'.nii')
epi_img_data = image.get_data()
imgShape = epi_img_data.shape
imgHalfX = imgShape[0]/2
imgHalfY = imgShape[1]/2
imgHalfZ = imgShape[2]/2

import matplotlib.pyplot as plt
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[imgHalfX, :, :, 0]
slice_1 = epi_img_data[:, imgHalfY, :, 0]
slice_2 = epi_img_data[:, :, imgHalfZ, 0]
<<<<<<< HEAD
modifiedSlice = copy.deepcopy(slice_2)

darkColor = 350
upperDark = 460
grayColor = 750
upperGray = 1100
whiteColor = 2000

for i in range(0, epi_img_data.shape[0]):
	for j in range(0, epi_img_data.shape[1]):
		value = modifiedSlice[i][j]
		if value > 0:
			if value <= upperDark:
				modifiedSlice[i][j] = darkColor
			if value <= upperGray and value > upperDark:
				modifiedSlice[i][j] = grayColor
			if value > upperGray:
				modifiedSlice[i][j] = whiteColor


show_slices([slice_2, modifiedSlice])
=======
show_slices([slice_0, slice_1, slice_2])
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6
plt.suptitle("Center slices for EPI image")  
plt.show()

