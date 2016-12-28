import nibabel as nib
import sys


imgNumber = str(sys.argv[1])
image = nib.load('../data/set_train/train_'+imgNumber+'.nii')
epi_img_data = image.get_data()
imgShape = epi_img_data.shape
imgHalfX = imgShape[0]/2
imgHalfY = imgShape[1]/2
imgHalfZ = imgShape[2]/2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def show_slices(slices,rect):
	""" Function to display row of image slices """
	fig, axes = plt.subplots(2, len(slices))
	orientation_x = [0,2,2]
	orientation_y = [1,1,0]
	for i, slice in enumerate(slices):
		axes[0][i].imshow(slice.T, cmap="gray", origin="lower")
		
		x = orientation_x[i]
		y = orientation_y[i]
		axes[0][i].add_patch(
			patches.Rectangle(
				(rect[x], rect[y]),   # (x,y)
				rect[3+x],		  # width
				rect[3+y],		  # height
				edgecolor="red",
				fill = False
			)
		)
		
		slice_part = slice[rect[x]:rect[x]+rect[3+x],rect[y]:rect[y]+rect[3+y]]
		print slice_part.shape
		axes[1][i].imshow(slice_part.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[imgHalfX, :, :, 0]
slice_1 = epi_img_data[:, imgHalfY, :, 0]
slice_2 = epi_img_data[:, :, imgHalfZ, 0]
#      [corner closest to 0, dimensions]
#      [x,y,z,xd,yd,zd]
rect = [50,45,100,100,25,25]

show_slices([slice_0, slice_1, slice_2],rect)

plt.suptitle("Center slices for EPI image")  
plt.show()

