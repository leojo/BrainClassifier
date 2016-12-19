import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

imgShape = nib.load('../data/set_train/train_1.nii').shape

imgQuarterX = imgShape[0]/2
imgTopQuarterZ = imgShape[2]*3/4

def extractImgNumber(imgPath):
	imgName = imgPath.split(os.sep)[-1]
	imgNum = int(imgName.split('_')[-1][:-4])
	return imgNum


# Fetch all directory listings of set_train
allImageSrc = sorted(glob.glob("../data/set_train/*"), key=extractImgNumber)


# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets, dtype='int_')

# For now, try fitting the data with linear regression using
# only one dimension of the image data (i.e. by fixing 2 two variables
# and keeping 1 degree of freedom) 
#
# For this we'll be choosing 1/4th part of x-axis and 3/4th of y-axis, simply
# to catch the most interesting area of the brain
all_samples = []
for i in range(0,len(allImageSrc)):
	img = nib.load(allImageSrc[i])
	imgData = img.get_data();
	# brain seen from top, center
	'''brain_slice = imgData[52:120, 65:150, 55:110, 0]
	black_pixels = 0
	for j in range(0,len(brain_slice)):
		for k in range(0, len(brain_slice[j])):
			for l in range(0, len(brain_slice[j][k])):
				if(brain_slice[j][k][l] < 450): black_pixels = black_pixels+1'''

	brain_slice = imgData[52:120, 65:150, 88, 0]
	black_pixels = 0
	for j in range(0,len(brain_slice)):
		for k in range(0, len(brain_slice[j])):
			if(brain_slice[j][k] < 450): black_pixels = black_pixels+1
	
	print "Percentage done: "+str((i*100)/len(allImageSrc))+" %"
	# Our 'feature' is how many black pixels we have:
	all_samples.append([black_pixels])


indicesOfHealthy = []
blacksOfHealthy = []
indicesOfUnhealthy = []
blacksOfUnhealthy = []
for i in range(0,len(targets)):
	if(targets[i] == 0):
		indicesOfUnhealthy.append(i)
		blacksOfUnhealthy.append(all_samples[i])
	else:
		indicesOfHealthy.append(i)
		blacksOfHealthy.append(all_samples[i])


fig, ax = plt.subplots()
ax.scatter(indicesOfHealthy, blacksOfHealthy, color="blue")
ax.scatter(indicesOfUnhealthy, blacksOfUnhealthy, color="red")
plt.show()

'''
total_error = 0
for i in range(0, len(all_samples)):
	trainData = all_samples[:i] + all_samples[i+1:]
	testData = all_samples[i]

	trainTargets = np.concatenate((targets[:i],targets[i+1:]))
	testTarget = targets[i]


	#reg = linear_model.LinearRegression()
	#reg = linear_model.Lasso(alpha = 0.1)
	reg = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
	reg.fit (trainData, trainTargets);
	reg.fit (trainData, trainTargets);

	testPrediction = reg.predict(testData)[0][0];
	print "Prediction for datapoint "+str(i)+":"
	print testPrediction
	print "Actual value"
	print testTarget[0]
	print ""
	total_error += abs(testPrediction-testTarget[0])

avg_error_1 = total_error/len(all_samples)
print "Average error (method 1) "+str(avg_error_1)'''

