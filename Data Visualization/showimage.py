import nibabel as nib
import sys


imgNumber = str(sys.argv[1])
image = nib.load('../data/set_train/train_'+imgNumber+'.nii')
epi_img_data = image.get_data()
imgShape = epi_img_data.shape
imgHalfX = 108 #imgShape[0]/2
imgHalfY = imgShape[1]/2
imgHalfZ = 54 #imgShape[2]/2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def show_slices(slices,rect):
	""" Function to display row of image slices """
	fig, axes = plt.subplots(2, len(slices))
	for i, slice in enumerate(slices):
		axes[0][i].imshow(slice.T, cmap="gray", origin="lower")
		indices = [0,1,2]
		x,y = indices[:i]+indices[i+1:]
		x = 2*x
		y = 2*y
		print x,y
		axes[0][i].add_patch(
			patches.Rectangle(
				(rect[x], rect[y]),   # (x,y)
				rect[x+1]-rect[x],		  # width
				rect[y+1]-rect[y],		  # height
				edgecolor="red",
				fill = False
			)
		)
		
		slice_part = slice[rect[x]:rect[x+1],rect[y]:rect[y+1]]
		print slice_part.shape
		axes[1][i].imshow(slice_part.T, cmap="gray", origin="lower")


#      [x_start,x_stop,y_start,y_stop,z_start,z_stop]
rect = [90,125,75,125,45,90]	

#Empyrical estimate: 
# X range: 90-125
# Y range: 75-125
# Z range: 45-90

slice_0 = epi_img_data[imgHalfX, :, :, 0]
slice_1 = epi_img_data[:, imgHalfY, :, 0]
slice_2 = epi_img_data[:, :, imgHalfZ, 0]

show_slices([slice_0, slice_1, slice_2],rect)

plt.suptitle("Center slices for EPI image")  
plt.show()

