import nibabel as nib
import matplotlib.pyplot as plt
import glob
import csv
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pprint, pickle


# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets, dtype='int_')
lastTarget = targets[-1]

# Get the preprocessed features:
pkl_file = open('../Features/histogram_norm.pkl', 'rb')
histograms = pickle.load(pkl_file)
pkl_file.close()

total_error=0
N = 30 #number of cross validation tests to run

# plotting vectors
xVector = np.arange(1,31)
yTargets = []
yPredictions = []
for i in range(0, N):
	#trainData = np.array(histograms[:i]+histograms[i+1:])
	#testData = np.array([histograms[i]])
	trainData = np.concatenate((histograms[:i],histograms[i+1:]))
	testData = histograms[i]



	trainTargets = np.concatenate((targets[:i],targets[i+1:]))
	testTarget = targets[i]


	preferredModel = linear_model.Ridge(alpha = 0.05)
	#preferredModel = linear_model.LinearRegression()
	#preferredModel = linear_model.Lasso(alpha = 0.95)
	#reg = make_pipeline(PolynomialFeatures(2), preferredModel)
	reg = preferredModel
	reg.fit (trainData, trainTargets);

	testPrediction = reg.predict(testData.reshape(1,-1));
	yTargets.append(testTarget[0])
	yPredictions.append(testPrediction[0][0])
	print "Prediction for datapoint "+str(i)+":"
	print testPrediction
	print "Actual value"
	print testTarget[0]
	print ""
	total_error += abs(testPrediction-testTarget[0])

print len(xVector)
print xVector
print len(yTargets)
print yTargets
print len(yPredictions)
print yPredictions
plt.scatter(list(xVector), list(yTargets))
plt.plot(list(xVector), list(yPredictions))
plt.show()

avg_error_1 = total_error/N
print "Average error (method 1) "+str(avg_error_1)

