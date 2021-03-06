import os
import inspect
import nibabel as nib
import numpy as np
import scipy as sp
import scipy.ndimage.interpolation as interpolation
import glob
import pickle
import math
from printProgress import printProgress

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
# from skimage.measure import compare_ssim as ssim

# The directory to store the precomputed features
featuresDir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# imgDir must be a string and maxValue must be an integer
# nBins is the number of "bins" for the histogram
def extractHistograms(imgDir, maxValue = 4000, nBins = -1, nPartitions = 1):
	if nBins == -1: nBins=maxValue

	# The number of different intensities per point of the histogram
	binSize = math.ceil((maxValue*1.)/nBins)
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"histograms_"+str(nBins)+"-"+str(maxValue)+"-"+str(nPartitions)+"_"+imgDir.replace(os.sep,"-")+".feature")
	print "Looking for file",outputFileName
	if os.path.isfile(outputFileName):
		print "Found file, loading..."
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		print "Done"
		return histograms
	print "Could not find file"

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		histograms = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(histograms))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"
	
	img_shape = nib.load(allImageSrc[0]).get_data().shape
	n_voxels=img_shape[0]*img_shape[1]*img_shape[2]
	n_iter = n_voxels*n_samples
	iter_count = len(histograms)*n_voxels
	printProgress(iter_count, n_iter)
	for i in range(len(histograms),n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		# Count occurances of each intensity below the maxValue
		single_brain = np.array([[[[0]*nBins]*nPartitions]*nPartitions]*nPartitions)
		for x in range(imgData.shape[0]):
			for y in range(imgData.shape[1]):
				for z in range(imgData.shape[2]):
					val = imgData[x][y][z][0]
					partX = int((x*nPartitions)/imgData.shape[0])
					partY = int((y*nPartitions)/imgData.shape[1])
					partZ = int((z*nPartitions)/imgData.shape[2])
					if val < maxValue and val > 0:
						c = int(val/binSize)
						single_brain[partX][partY][partZ][c] += 1
					iter_count += 1
					printProgress(iter_count, n_iter, decimals = 5)
		histograms.append(single_brain.flatten().tolist())
		
		output = open(outputFileName+".part","wb")
		pickle.dump(histograms,output)
		output.close()

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return histograms


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(reduce(lambda x, y: x*y, imageA.shape))
	return err

def extractFlipSim(imgDir,nPartitions=8,exponent=50):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"flipsim_"+str(nPartitions)+"-"+str(exponent)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		data = pickle.load(save)
		save.close()
		return data

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	data = []
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = np.asarray(img.get_data()[:,:,:,0]);
		imgData_flipped = np.flip(imgData,0)
		part_size_x = int(round(float(imgData.shape[0])/nPartitions))
		part_size_y = int(round(float(imgData.shape[1])/nPartitions))
		part_size_z = int(round(float(imgData.shape[2])/nPartitions))
		partsSim = []
		for x in range(nPartitions):
			for y in range(nPartitions):
				for z in range(nPartitions):
					x_start = x*part_size_x
					if(x == nPartitions-1): x_stop=imgData.shape[0]
					else: x_stop = (x+1)*part_size_x

					y_start = y*part_size_y
					if(y == nPartitions-1): y_stop=imgData.shape[1]
					else: y_stop = (y+1)*part_size_y

					z_start = z*part_size_z
					if(z == nPartitions-1): z_stop=imgData.shape[2]
					else: z_stop = (z+1)*part_size_z

					imgPart = imgData[x_start:x_stop,y_start:y_stop,z_start:z_stop]
					imgPart_flipped = imgData_flipped[x_start:x_stop,y_start:y_stop,z_start:z_stop]

					#ssim_val = ssim(imgPart,imgPart_flipped)
					mse_val = mse(imgPart,imgPart_flipped)
					#similarity = [ssim_val**exponent,mse_val]
					partsSim.append(mse_val)
		data.append(partsSim)
		printProgress(i+1, n_samples)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(data,output)
	output.close()
	print "Done"

	return data

# Test function to see if there is a significant
# difference in the amount of voxels between
# male and female brains
def extractBrainSize(imgDir): 
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"brainsize_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		brainsizes = pickle.load(save)
		save.close()
		return brainsizes

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	brainsizes = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		voxelCount = 0
		for x in range(imgData.shape[0]):
			for y in range(imgData.shape[1]):
				for z in range(imgData.shape[2]):
					val = imgData[x][y][z][0]
					if val > 0:
						voxelCount += 1	
		brainsizes.append(voxelCount)
		printProgress(i+1, n_samples)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(brainsizes,output)
	output.close()
	print "Done"

	return brainsizes

def extractCompleteBrain(imgDir):
	imgPath = os.path.join(imgDir,"*")
	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"complete_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		data = pickle.load(save)
		save.close()
		return data
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	data = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		single_brain = imgData.flatten().tolist()
		data.append(single_brain)
		printProgress(i+1, n_samples)
	print "\n!!!!!NOT Storing the features !!!!"
	#output = open(outputFileName,"wb")
	#pickle.dump(data,output)
	#output.close()
	#print "Done"
	return data

def extractBrainPart(imgDir,n_divisions=3,x_part=0,y_part=0,z_part=0):
	imgPath = os.path.join(imgDir,"*")
	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"brainpart_"+str(n_divisions)+"_"+str(x_part)+"_"+str(y_part)+"_"+str(z_part)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		data = pickle.load(save)
		save.close()
		return data
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	data = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		# Resize to a <scale> proportion of original size
		imgData_resized = sp.ndimage.interpolation.zoom(imgData_original,scale)
		imgData_flipped = np.flip(imgData_resized,0)
		
		data.append([similarity])
		printProgress(i+1, n_samples)
	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(data,output)
	output.close()
	print "Done"
	return data
'''
# This was an attempt at a more sophisticated feature using agglomerative clustering to define "colors"
# and then taking a histogram of those color. This did not prove to give better results.
def extractHierarchicalClusters(imgDirFullPath, n_clusters=10, ignoreCache=False, scale=0.10):
	imgPath = os.path.join(imgDirFullPath,"*")

	outputFileName = os.path.join(featuresDir,"hierarchicalclusters_"+str(n_clusters)+"_"+str(scale)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	
	if os.path.isfile(outputFileName) and not ignoreCache:
		save = open(outputFileName,'rb')
		clusters = pickle.load(save)
		save.close()
		return clusters
	
	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	clusters = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*n_clusters
		total_intensities = [0]*n_clusters
		img = nib.load(allImageSrc[i])
		imgData_original = np.asarray(img.get_data()[:,:,:,0])
		# Resize to 10% of original size for faster processing
		imgData_resized = sp.ndimage.interpolation.zoom(imgData_original,scale)
		imgData = np.reshape(imgData_resized,(-1,1))

		connectivity = grid_to_graph(*imgData_resized.shape)
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
		labels = ward.fit_predict(imgData).flatten().tolist()

		for j, lab in enumerate(labels):
			intensity = imgData[j][0]
			total_intensities[lab] += intensity
			hist[lab] += 1

		avg_intensity = np.asarray(total_intensities)*1./np.asarray(hist)
		avg_intensity = avg_intensity.flatten().tolist()
		avg_intensity, hist = zip(*sorted(zip(avg_intensity,hist)))
		
		clusters.append(hist)
		printProgress(i+1, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(clusters,output)
	output.close()
	print "Done"
	return clusters

# The AgglomerativeClusters approach was suffering from the severely reduced "resolution" of the images
# and this was an attempt to improve on that by only looking at one slice of the image instead of reducing the
# "resolution". This too was unsuccessful.
def extractHierarchicalClustersSingleSlice(imgDirFullPath, n_clusters=10, ignoreCache=False):
	imgPath = os.path.join(imgDirFullPath,"*")

	outputFileName = os.path.join(featuresDir,"hierarchicalclusterssingleslice_"+str(n_clusters)+"_"+imgDirFullPath.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName) and not ignoreCache:
		save = open(outputFileName,'rb')
		clusters = pickle.load(save)
		save.close()
		return clusters

	# Fetch all directory listings of set_train
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	clusters = []
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		hist = [0]*n_clusters
		total_intensities = [0]*n_clusters
		img = nib.load(allImageSrc[i])
		imgData_original = np.asarray(img.get_data()[:,:,:,0])
		brainSlice = imgData_original[:,:,imgData_original.shape[2]/2]
		imgData = np.reshape(brainSlice,(-1,1))

		connectivity = grid_to_graph(*brainSlice.shape)
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
		labels = ward.fit_predict(imgData).flatten().tolist()

		for j, lab in enumerate(labels):
			intensity = imgData[j][0]
			total_intensities[lab] += intensity
			hist[lab] += 1
		avg_intensity = np.asarray(total_intensities)*1./np.asarray(hist)
		avg_intensity = avg_intensity.flatten().tolist()
		avg_intensity, hist = zip(*sorted(zip(avg_intensity,hist)))

		clusters.append(hist)
		printProgress(i+1, n_samples)
	print "Done"
	print "\nStoring the features in "+outputFileName

	output = open(outputFileName,"wb")
	pickle.dump(clusters,output)
	output.close()
	print "Done"
	return clusters
'''
# imgDir must be a string and maxValue must be an integer
# nBins is the number of "bins" for the histogram
def extractZoneAverages(imgDir, nPartitions = 1):
	imgPath = os.path.join(imgDir,"*")

	allZoneAverages = []

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"zoneavg_"+str(nPartitions)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		imgDataDisected = imgData[20:148, 35:163, 30:158, 0]
		# Size should be same for all dimensions, imgData should
		# have same dimensions for x, y, z all such that they can be
		# divided by nPartitions
		size = imgDataDisected.shape[0]/nPartitions
		'''
		zoneAverages = []
		for j in range(0,nPartitions):
			for k in range(0, nPartitions):
				for l in range(0, nPartitions):
					xStart = j*size
					yStart = k*size
					zStart = l*size

					totalSum = 0
					totalVoxels = size*size*size
					for x in range(xStart, xStart+size):
						for y in range(yStart, yStart+size):
							for z in range(zStart, zStart+size):
								totalSum += imgDataDisected[x][y][z]

					mean = totalSum/totalVoxels
					zoneAverages.append(mean)	
					#print "brain " +str(i)+" zone " + str(j) + ", " + str(k) + ", " + str(l) + " with mean " +str(mean)
		'''
		zoneAverages = np.array([[[0]*nPartitions]*nPartitions]*nPartitions)
		totalVoxels = size*size*size
		for x in range(imgDataDisected.shape[0]):
			for y in range(imgDataDisected.shape[1]):
				for z in range(imgDataDisected.shape[2]):
					val = imgDataDisected[x][y][z]
					partX = int((x*nPartitions)/imgDataDisected.shape[0])
					partY = int((y*nPartitions)/imgDataDisected.shape[1])
					partZ = int((z*nPartitions)/imgDataDisected.shape[2])
					if val > 0:
						zoneAverages[partX][partY][partZ] += val
		for j in range(zoneAverages.shape[0]):
			for k in range(zoneAverages.shape[1]):
				for l in range(zoneAverages.shape[1]):
					zoneAverages[j][k][l] = float(zoneAverages[j][k][l])/totalVoxels
		allZoneAverages.append(zoneAverages.flatten().tolist())
		printProgress(i+1, n_samples)		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allZoneAverages,output)
	output.close()
	print "Done"

	return allZoneAverages

def extractBlackzones(imgDir, nPartitions=1):
	imgPath = os.path.join(imgDir,"*")

	allBlackZones = []

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"blackzones_"+str(nPartitions)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		imgDataDisected = imgData[20:148, 35:163, 30:158, 0]

		blackzones = np.asarray([[[0]*nPartitions]*nPartitions]*nPartitions)
		# Size should be same for all dimensions, imgData should
		# have same dimensions for x, y, z all such that they can be
		# divided by nPartitions
		for x in range(imgDataDisected.shape[0]):
			for y in range(imgDataDisected.shape[1]):
				for z in range(imgDataDisected.shape[2]):
					val = imgDataDisected[x][y][z]
					partX = int((x*nPartitions)/imgDataDisected.shape[0])
					partY = int((y*nPartitions)/imgDataDisected.shape[1])
					partZ = int((z*nPartitions)/imgDataDisected.shape[2])
					if val < 450 and val > 0:
						blackzones[partX][partY][partZ] += 1

		allBlackZones.append(blackzones.flatten().tolist())
		printProgress(i+1, n_samples)		
		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allBlackZones,output)
	output.close()
	print "Done"
	return allBlackZones

def extractThreeColors(imgDir, darkColor, grayColor, whiteColor, nPartitions=1):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"threeColors_"+str(nPartitions)+"_"+str(darkColor)+"_"+str(grayColor)+"_"+str(whiteColor)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	

	allPercentages = []
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		imgDataDisected = imgData[:, :, :, 0]

		percentages = []
		voxelsCounted = 0.0
		blackCounted = 0		
		grayCounted = 0
		whiteCounted = 0
		for j in range(0, imgData.shape[0]):
			for k in range(0, imgData.shape[1]):
				for l in range(0, imgData.shape[2]):
					value = imgDataDisected[j][k][l]
					if value > 0:
						voxelsCounted += 1
						if value <= darkColor:
							blackCounted += 1
						if value <= grayColor and value > darkColor:
							grayCounted+= 1
						if value <= whiteColor and value > grayColor:
							whiteCounted += 1

		if(voxelsCounted > 0):
			percentages.append(blackCounted/voxelsCounted)
			percentages.append(grayCounted/voxelsCounted)
			percentages.append(whiteCounted/voxelsCounted)

		allPercentages.append(percentages)
		printProgress(i+1, n_samples)		
		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allPercentages,output)
	output.close()
	print "Done"
	return allPercentages

def extractColorPercentage(imgDir, upperDark, upperGray, firstColor, secondColor):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"2ColorPercentage_"+str(upperDark)+"_"+str(upperGray)+"_"+str(firstColor)+"_"+str(secondColor)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	

	allPercentages = []
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		imgDataSlice = imgData[:, :, imgData.shape[2]/2, 0]

		COUNTS = []
		voxelsCounted = 0.0
		blackCounted = 0		
		grayCounted = 0
		whiteCounted = 0
		for j in range(0, imgData.shape[0]):
			for k in range(0, imgData.shape[1]):
					value = imgDataSlice[j][k]
					if value > 0:
						voxelsCounted += 1
						if value <= upperDark:
							blackCounted += 1
						if value <= upperGray and value > upperDark:
							grayCounted+= 1
						if value > upperGray:
							whiteCounted += 1

		if(voxelsCounted > 0):
			COUNTS.append(blackCounted)
			COUNTS.append(grayCounted)
			COUNTS.append(whiteCounted)

		allPercentages.append(COUNTS[firstColor]/COUNTS[secondColor])
		printProgress(i+1, n_samples)		
		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allPercentages,output)
	output.close()
	print "Done"
	return allPercentages

def extractColorPercentage(imgDir, upperDark, upperGray):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"2ColorPercentage_"+str(upperDark)+"_"+str(upperGray)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	

	allPercentages = []
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();

		imgDataSlice1 = imgData[:, :, imgData.shape[2]/2 + imgData.shape[2]/4, 0]
		imgDataSlice2 = imgData[:, :, imgData.shape[2]/2, 0]
		imgDataSlice3 = imgData[:, :, imgData.shape[2]/2 - imgData.shape[2]/4, 0]

		COUNTS = []
		ratios = []
		voxelsCounted = 0.0
		blackCounted = 1		
		grayCounted = 1
		whiteCounted = 1
		for j in range(0, imgData.shape[0]):
			for k in range(0, imgData.shape[1]):
					values = []
					values.append(imgDataSlice1[j][k])
					values.append(imgDataSlice2[j][k])
					values.append(imgDataSlice3[j][k])
					for value in values:
						if value > 0:
							voxelsCounted += 1
							if value <= upperDark:
								blackCounted += 1
							if value <= upperGray and value > upperDark:
								grayCounted+= 1
							if value > upperGray:
								whiteCounted += 1

		if(voxelsCounted > 0):
			#COUNTS.append(blackCounted)
			COUNTS.append(grayCounted)
			COUNTS.append(whiteCounted)

		#ratios.append((1.0*COUNTS[0])/COUNTS[1])
		#ratios.append((1.0*COUNTS[1])/COUNTS[2])
		#ratios.append((1.0*COUNTS[0])/COUNTS[2])

		allPercentages.append(COUNTS)
		#printProgress(i+1, n_samples)		
		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allPercentages,output)
	output.close()
	print "Done"
	return allPercentages

# Written for 2D for faster computing while testing
def extractColoredZone(imgDir, minColor, maxColor, nPartitions=1):
	imgPath = os.path.join(imgDir,"*")

	allColoredZones = []

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"coloredzones2d_"+str(nPartitions)+"_"+str(minColor)+"_"+str(maxColor)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		imgDataDisected = imgData[20:148, 35:163, 88, 0]

		colZones = np.asarray([[[0]*nPartitions]*nPartitions]*nPartitions)
		# Size should be same for all dimensions, imgData should
		# have same dimensions for x, y, z all such that they can be
		# divided by nPartitions
		for x in range(imgDataDisected.shape[0]):
			for y in range(imgDataDisected.shape[1]):
				#for z in range(imgDataDisected.shape[2]):
					val = imgDataDisected[x][y]#[z]
					partX = int((x*nPartitions)/imgDataDisected.shape[0])
					partY = int((y*nPartitions)/imgDataDisected.shape[1])
					#partZ = int((z*nPartitions)/imgDataDisected.shape[2])
					if val <= maxColor and val >= minColor:
						colZones[partX][partY] += 1

		allColoredZones.append(colZones.flatten().tolist())
		printProgress(i+1, n_samples)		
		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allColoredZones,output)
	output.close()
	print "Done"
	return allColoredZones


# Written for 3D
def extractColoredZone3D(imgDir, minColor, maxColor, nPartitions=1):
	imgPath = os.path.join(imgDir,"*")

	allColoredZones = []

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"coloredzones_"+str(nPartitions)+"_"+str(minColor)+"_"+str(maxColor)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);
	print "Found "+str(n_samples)+" images!"
	print "Preparing the data"
	printProgress(0, n_samples)
	for i in range(0,n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		imgDataDisected = imgData[:, :, :, 0]

		colZones = np.asarray([[[0]*nPartitions]*nPartitions]*nPartitions)
		# Size should be same for all dimensions, imgData should
		# have same dimensions for x, y, z all such that they can be
		# divided by nPartitions
		for x in range(imgDataDisected.shape[0]):
			for y in range(imgDataDisected.shape[1]):
				for z in range(imgDataDisected.shape[2]):
					val = imgDataDisected[x][y][z]
					partX = int((x*nPartitions)/imgDataDisected.shape[0])
					partY = int((y*nPartitions)/imgDataDisected.shape[1])
					partZ = int((z*nPartitions)/imgDataDisected.shape[2])
					if val <= maxColor and val >= minColor:
						colZones[partX][partY][partZ] += 1

		allColoredZones.append(colZones.flatten().tolist())
		printProgress(i+1, n_samples)		
		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allColoredZones,output)
	output.close()
	print "Done"
	return allColoredZones
	
def extractGrayWhiteRatio(imgDir, nPartitions=1):
	imgPath = os.path.join(imgDir,"*")

	allColoredZones = []

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"grayWhiteRatio_"+str(nPartitions)+"_"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		zoneAverages = pickle.load(save)
		save.close()
		return zoneAverages

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	n_samples = len(allImageSrc);

	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		allColoredZones = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(allColoredZones))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	img_shape = nib.load(allImageSrc[0]).get_data().shape
	n_voxels=img_shape[0]*img_shape[1]*img_shape[2]
	n_iter = n_voxels*n_samples
	iter_count = len(allColoredZones)*n_voxels

	minGray = 450
	maxGray = 800
	minWhite = 900
	maxWhite = 2500
	printProgress(iter_count, n_iter, decimals=5)
	for i in range(len(allColoredZones),n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		imgDataDisected = imgData[:, :, :, 0]

		grayInZone = np.asarray([[[1.0]*nPartitions]*nPartitions]*nPartitions)
		whiteInZone = np.asarray([[[1.0]*nPartitions]*nPartitions]*nPartitions)
		# Size should be same for all dimensions, imgData should
		# have same dimensions for x, y, z all such that they can be
		# divided by nPartitions
		for x in range(imgDataDisected.shape[0]):
			for y in range(imgDataDisected.shape[1]):
				for z in range(imgDataDisected.shape[2]):
					iter_count += 1
					val = imgDataDisected[x][y][z]
					partX = int((x*nPartitions)/imgDataDisected.shape[0])
					partY = int((y*nPartitions)/imgDataDisected.shape[1])
					partZ = int((z*nPartitions)/imgDataDisected.shape[2])
					
					if val <= maxGray and val >= minGray: # Gray
						grayInZone[partX][partY][partZ] += 1.0
					if val <= maxWhite and val >= minWhite: #White
						whiteInZone[partX][partY][partZ] += 1.0
					printProgress(iter_count,n_iter,decimals=5)

		zoneRatio = grayInZone/whiteInZone
		allColoredZones.append(zoneRatio.flatten().tolist())

		output = open(outputFileName+".part","wb")
		pickle.dump(allColoredZones,output)
		output.close()

		

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(allColoredZones,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
		
	return allColoredZones

def extractHippocampi(imgDir):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"hippocampi_raw"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		hippocampi = pickle.load(save)
		save.close()
		return hippocampi

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	hippocampi = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		hippocampi = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(hippocampi))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	img_shape = nib.load(allImageSrc[0]).get_data().shape
	imgHalfY = img_shape[1]/2
	printProgress(len(hippocampi), n_samples, decimals = 3)
	for i in range(len(hippocampi),n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		hippocampus = imgData[100:125, imgHalfY, 45:70, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		hippocampi.append(hippocampus.flatten())
		#output = open(outputFileName+".part","wb")
		#pickle.dump(hippocampi,output)
		#output.close()
		printProgress(i+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(hippocampi,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return hippocampi
	
def extractHippocampusMeans(imgDir):
	hippocampi = extractHippocampi(imgDir)
	return np.mean(hippocampi,axis=1).reshape(-1,1)
	
def extractHippocampusVars(imgDir):
	hippocampi = extractHippocampi(imgDir)
	return np.var(hippocampi,axis=1).reshape(-1,1)
	
	
def extractHippocampusMedians(imgDir):
	hippocampi = extractHippocampi(imgDir)
	return np.median(hippocampi,axis=1).reshape(-1,1)

def extractHippocampusHistograms(imgDir,maxValue=4000,bins=45):
	hippocampi = extractHippocampi(imgDir)
	histograms = []
	for h in hippocampi:
		histograms.append(np.histogram(h,bins=bins, range=(1,maxValue))[0])
	return histograms

# =========================================
#       3D HIPPOCAMPUS AMYGDALA STUFF
# =========================================

def extractLargeHippocampi3D(imgDir):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"large_hippocampi3d_raw"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		hippocampi = pickle.load(save)
		save.close()
		return hippocampi

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	hippocampi = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		hippocampi = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(hippocampi))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	# Hippocampus bounding box, as estimated empirically from images: 
	# X range: 90-125 (35)
	# Y range: 75-125 (50)
	# Z range: 45-90  (45)
	
	img_shape = nib.load(allImageSrc[0]).get_data().shape
	imgHalfY = img_shape[1]/2
	printProgress(len(hippocampi), n_samples, decimals = 3)
	for i in range(len(hippocampi),n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		hippocampusRear = imgData[102:125, 79:100, 45:81, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		hippocampusFront = imgData[102:125, 100:114, 45:81, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		hippocampus = np.concatenate((hippocampusRear.flatten(),hippocampusFront.flatten()))
		hippocampi.append(hippocampus)
		output = open(outputFileName+".part","wb")
		pickle.dump(hippocampi,output)
		output.close()
		printProgress(i+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(hippocampi,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return hippocampi

def extractSmallHippocampi3D(imgDir):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"small_hippocampi3d_raw"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		hippocampi = pickle.load(save)
		save.close()
		return hippocampi

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	hippocampi = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		hippocampi = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(hippocampi))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	# Hippocampus bounding box, as estimated empirically from images: 
	# X range: 90-125 (35)
	# Y range: 75-125 (50)
	# Z range: 45-90  (45)
	
	img_shape = nib.load(allImageSrc[0]).get_data().shape
	imgHalfY = img_shape[1]/2
	printProgress(len(hippocampi), n_samples, decimals = 3)
	for i in range(len(hippocampi),n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		hippocampus =imgData[102:125, 100:114, 45:64, 0]
		hippocampi.append(hippocampus.flatten())
		output = open(outputFileName+".part","wb")
		pickle.dump(hippocampi,output)
		output.close()
		printProgress(i+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(hippocampi,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return hippocampi

def extractAmygdala3D(imgDir):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"amygdala3d_raw"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		hippocampi = pickle.load(save)
		save.close()
		return hippocampi

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	hippocampi = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		hippocampi = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(hippocampi))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	# Hippocampus bounding box, as estimated empirically from images: 
	# X range: 90-125 (35)
	# Y range: 75-125 (50)
	# Z range: 45-90  (45)
	
	img_shape = nib.load(allImageSrc[0]).get_data().shape
	imgHalfY = img_shape[1]/2
	printProgress(len(hippocampi), n_samples, decimals = 3)
	for i in range(len(hippocampi),n_samples):
		img = nib.load(allImageSrc[i])
		imgData = img.get_data();
		hippocampus =imgData[102:120, 109:120, 43:64, 0]
		hippocampi.append(hippocampus.flatten())
		output = open(outputFileName+".part","wb")
		pickle.dump(hippocampi,output)
		output.close()
		printProgress(i+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(hippocampi,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return hippocampi	
	
def extractLargeHippocampusMeans3D(imgDir):
	hippocampi = extractLargeHippocampi3D(imgDir)
	return np.mean(hippocampi,axis=1).reshape(-1,1)

def extractSmallHippocampusMeans3D(imgDir):
	hippocampi = extractSmallHippocampi3D(imgDir)
	return np.mean(hippocampi,axis=1).reshape(-1,1)

def extractAmygdalaMeans3D(imgDir):
	hippocampi = extractAmygdala3D(imgDir)
	return np.mean(hippocampi,axis=1).reshape(-1,1)

	
def extractLargeHippocampusVars3D(imgDir):
	hippocampi = extractLargeHippocampi3D(imgDir)
	return np.var(hippocampi,axis=1).reshape(-1,1)
	

def extractSmallHippocampusVars3D(imgDir):
	hippocampi = extractSmallHippocampi3D(imgDir)
	return np.var(hippocampi,axis=1).reshape(-1,1)
		
def extractAmygdalaVars3D(imgDir):
	hippocampi = extractAmygdala3D(imgDir)
	return np.var(hippocampi,axis=1).reshape(-1,1)
		

def extractLargeHippocampusMedians3D(imgDir):
	hippocampi = extractLargeHippocampi3D(imgDir)
	return np.median(hippocampi,axis=1).reshape(-1,1)

def extractSmallHippocampusMedians3D(imgDir):
	hippocampi = extractSmallHippocampi3D(imgDir)
	return np.median(hippocampi,axis=1).reshape(-1,1)


def extractAmygdalaMedians3D(imgDir):
	hippocampi = extractAmygdala3D(imgDir)
	return np.median(hippocampi,axis=1).reshape(-1,1)



def extractLargeHippocampusHistograms3D(imgDir,maxValue=4000,bins=45):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"large_hippocampi3d_histo-"+str(maxValue)+"-"+str(bins)+"-"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		histograms = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(histograms))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	# Hippocampus bounding box, as estimated empirically from images: 
	# X range: 90-125 (35)
	# Y range: 75-125 (50)
	# Z range: 45-90  (45)
	rect = [102,125,79,114,45,81]
	
	img_shape = nib.load(allImageSrc[0]).get_data().shape
	printProgress(len(histograms), n_samples, decimals = 3)
	for l in range(len(histograms),n_samples):
		img = nib.load(allImageSrc[l])
		imgData = img.get_data();
		#hippocampusRear = imgData[90:125, 75:100, 45:90, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		#hippocampusFront = imgData[90:125, 100:125, 45:90, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		concatHistos = []
		
		for i in range(3):
			x_step = int(round((rect[1]-rect[0])/3.0))
			x_start = rect[0]+x_step*i
			x_stop = min(rect[1],rect[0]+x_step*(i+1))
			for j in range(3):
				y_step = int(round((rect[3]-rect[2])/3.0))
				y_start = rect[2]+y_step*j
				y_stop = min(rect[3],rect[2]+y_step*(j+1))
				for k in range(3):							
					z_step = int(round((rect[5]-rect[4])/3.0))
					z_start = rect[4]+z_step*k
					z_stop = min(rect[5],rect[4]+z_step*(k+1))
					imgPart = imgData[x_start:x_stop, y_start:y_stop, z_start:z_stop, 0]
					histo = np.histogram(imgPart, bins=bins, range=(1,maxValue))[0].tolist()
					concatHistos += histo
	
		histograms.append(concatHistos)
		output = open(outputFileName+".part","wb")
		pickle.dump(histograms,output)
		output.close()
		printProgress(l+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return histograms


def extractSmallHippocampusHistograms3D(imgDir,maxValue=4000,bins=45):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"small_hippocampi3d_histo-"+str(maxValue)+"-"+str(bins)+"-"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		histograms = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(histograms))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	# Hippocampus bounding box, as estimated empirically from images: 
	# X range: 90-125 (35)
	# Y range: 75-125 (50)
	# Z range: 45-90  (45)
	rect = [102,125,100,114,45,64]

	img_shape = nib.load(allImageSrc[0]).get_data().shape
	printProgress(len(histograms), n_samples, decimals = 3)
	for l in range(len(histograms),n_samples):
		img = nib.load(allImageSrc[l])
		imgData = img.get_data();
		#hippocampusRear = imgData[90:125, 75:100, 45:90, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		#hippocampusFront = imgData[90:125, 100:125, 45:90, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		concatHistos = []
		for i in range(3):
			x_step = int(round((rect[1]-rect[0])/3.0))
			x_start = rect[0]+x_step*i
			x_stop = min(rect[1],rect[0]+x_step*(i+1))
			for j in range(3):
				y_step = int(round((rect[3]-rect[2])/3.0))
				y_start = rect[2]+y_step*j
				y_stop = min(rect[3],rect[2]+y_step*(j+1))
				for k in range(3):							
					z_step = int(round((rect[5]-rect[4])/3.0))
					z_start = rect[4]+z_step*k
					z_stop = min(rect[5],rect[4]+z_step*(k+1))
					imgPart = imgData[x_start:x_stop, y_start:y_stop, z_start:z_stop, 0]
					histo = np.histogram(imgPart, bins=bins, range=(1,maxValue))[0].tolist()
					concatHistos += histo
	
		histograms.append(concatHistos)
		output = open(outputFileName+".part","wb")
		pickle.dump(histograms,output)
		output.close()
		printProgress(l+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return histograms


def extractAmygdalaHistograms3D(imgDir,maxValue=4000,bins=45):
	imgPath = os.path.join(imgDir,"*")

	# This is the cache for the feature, used to make sure we do the heavy computations more often than necessary
	outputFileName = os.path.join(featuresDir,"amygdala3d_histo-"+str(maxValue)+"-"+str(bins)+"-"+imgDir.replace(os.sep,"-")+".feature")
	if os.path.isfile(outputFileName):
		save = open(outputFileName,'rb')
		histograms = pickle.load(save)
		save.close()
		return histograms

	# Fetch all directory listings of set_train and sort them on the image number
	allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
	histograms = []
	n_samples = len(allImageSrc);
	if os.path.isfile(outputFileName+".part"):
		save = open(outputFileName+".part",'rb')
		histograms = pickle.load(save)
		save.close()
		print "Found "+str(n_samples)+" images, "+str(len(histograms))+" already processed. Resuming..."
	else:
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"

	# Hippocampus bounding box, as estimated empirically from images: 
	# X range: 90-125 (35)
	# Y range: 75-125 (50)
	# Z range: 45-90  (45)
	rect = [102,120,109,120,43,64]

	img_shape = nib.load(allImageSrc[0]).get_data().shape
	printProgress(len(histograms), n_samples, decimals = 3)
	for l in range(len(histograms),n_samples):
		img = nib.load(allImageSrc[l])
		imgData = img.get_data();
		#hippocampusRear = imgData[90:125, 75:100, 45:90, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		#hippocampusFront = imgData[90:125, 100:125, 45:90, 0] #MAGIC NUMBERS, DO NOT MEDDLE!!!
		concatHistos = []
		for i in range(3):
			x_step = int(round((rect[1]-rect[0])/3.0))
			x_start = rect[0]+x_step*i
			x_stop = min(rect[1],rect[0]+x_step*(i+1))
			for j in range(3):
				y_step = int(round((rect[3]-rect[2])/3.0))
				y_start = rect[2]+y_step*j
				y_stop = min(rect[3],rect[2]+y_step*(j+1))
				for k in range(3):							
					z_step = int(round((rect[5]-rect[4])/3.0))
					z_start = rect[4]+z_step*k
					z_stop = min(rect[5],rect[4]+z_step*(k+1))
					imgPart = imgData[x_start:x_stop, y_start:y_stop, z_start:z_stop, 0]
					histo = np.histogram(imgPart, bins=bins, range=(1,maxValue))[0].tolist()
					concatHistos += histo
	
		histograms.append(concatHistos)
		output = open(outputFileName+".part","wb")
		pickle.dump(histograms,output)
		output.close()
		printProgress(l+1, n_samples, decimals = 3)

	print "\nStoring the features in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(histograms,output)
	output.close()
	print "Done"

	if os.path.isfile(outputFileName+".part"):
		os.remove(outputFileName+".part")
	
	return histograms



def extractImgNumber(imgPath):
	imgName = imgPath.split(os.sep)[-1]
	imgNum = int(imgName.split('_')[-1][:-4])
	return imgNum
