import csv
import sys
import numpy as np
from sklearn import pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

from Features.extract_features import *
from Features.scoring import hammingLoss
from Features.scoring import partialHammingLoss


# IMPORT SEM THARF AD TAKA TIL I THEGAR VID VITUM HVADA STUFF VID VILJUM NOTA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest

'''
Template for predict_final.py

Goal:

1. Extract features from MRI
2. For each label (age, health, sex), create a classifier from
	i. A subset of the features above, where the most variance is found
	ii. With a voting classifier from several models that prove to be good for said class.
3. Cross validate all classifiers
4. Create a classifier that predicts all labels using the three classifiers
5. Generate an output file with predictions of each label for all test images
'''


# ==========================================
#				  HELPER FUNCTIONS
# ==========================================

def getBestFeatures(classLabel):
	bestFeatures = []
	correspondingPipelines = []

	minVariance = 0
	for i in range(0, bestFeatures.size()):
		var = calculateVariance(allFeatures[i], classLabel)
		if var > minVariance:
			bestFeatures.append(allFeatures[i])
			correspondingPipelines.append(allPipelines[i])

	return [bestFeatures, correspondingPipelines]

def calculateVariance(feature, classLabel): 
	return


# Create a voting classifier from several classifiers for EACH feature, then
# create a voting classifier from these feature-specific classifiers
def createVotingClassifier(classLabel, features, pipelines, names, classifiers):
	voterWeights = []
	voters = []
	# Create a model for each kind of feature and preprocessing pipeline for that feature:
	for feature, preproc in zip(features, pipelines):
		
		print "Shape of feature:",np.asarray(feature).shape
		print "Shape of targets:",targets[:,classLabel].shape
		# We want the model to be a voter combined from several classifiers:
		weights = []
		for name, classifier in zip(names,classifiers):
			print(name)
			pl = pipeline.make_pipeline(
				preproc,
				classifier)

			scorer = make_scorer(partialHammingLoss,greater_is_better=False)
			scores = cross_val_score(pl, feature, targets[:,classLabel], cv=10, scoring=scorer, n_jobs=1)
			print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),name)
			weights.append(1.0/(-scores.mean()))

		model = pipeline.make_pipeline(
				preproc,
				VotingClassifier(zip(names,classifiers), voting='soft', weights=weights ,n_jobs=1)
				)

		print "\nCalculating score of model:"
		scorer = make_scorer(partialHammingLoss,greater_is_better=False)
		scores = cross_val_score(model, feature, targets[:,classLabel], cv=10, scoring=scorer, n_jobs=1)
		print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),"VotingClassifier")
		model.fit(feature,targets[:,classLabel])
		voterWeights.append(1.0/(-scores.mean()))
		voters.append(model)
	return np.array(zip(voters,voterWeights)) # Since the voters in this "model" require different feature-preprocessing it must be used manually
	
def createPredictions(testFeatures, sexWeightedVoters, ageWeightedVoters, healthWeightedVoters):
	numTestSamples = np.asarray(testFeatures[0]).shape[0]
	predictions = []
	for weightedVoters in [sexWeightedVoters,ageWeightedVoters,healthWeightedVoters]:
		totalWeight = np.sum(weightedVoters[:,1])
		totalPrediction = np.array([0]*numTestSamples).astype(np.float)
		for feature,voter,weight in zip(testFeatures,*zip(*weightedVoters)):
			prediction0, prediction1 = zip(*voter.predict_proba(feature))
			print "Shape of prediction:",np.array(prediction1).shape
			totalPrediction += np.array(prediction1).astype(np.float)*float(weight)/float(totalWeight)
		totalPrediction = np.round(totalPrediction)
		predictions.append(totalPrediction.tolist())
	
	id = 0
	resultFileName = 'submission'
	if len(sys.argv) == 2:
		resultFileName = sys.argv[1]
	if resultFileName[-4:] != ".csv":
		resultFileName += ".csv"
	with open(resultFileName, 'w') as csvfile:
		resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
		resultWriter.writerow(['ID','Sample','Label','Predicted'])
		for sample_no, sample in enumerate(zip(*predictions)):
			for label, prediction in zip(['gender','age','health'],[bool(sample[0]),bool(sample[1]),bool(sample[2])]):
				row=[id,sample_no,label,prediction]
				resultWriter.writerow(row)
				id+=1
		csvfile.close()

	

# ==========================================
#				   TARGETS
# ==========================================

# Get the targets
with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets).astype(np.int)
targets_sex, targets_age, targets_health = zip(*targets) 

# ==========================================
#				 TRAIN FEATURES
# ==========================================
# TO BE ADDED: Gray white ratio; black zones from center of brain

allFeatures = []

### histo = extractHistograms('data/set_train', 4000, 45, 9)
flipzones = extractFlipSim('data/set_train')
#blackzone = extractColoredZone('data/set_train', )
grayzone = extractColoredZone('data/set_train', 450, 800, 8)
hippocMedian = extractHippocampusMedians('data/set_train')
hippocMean = extractHippocampusMeans('data/set_train')
hippocVar = extractHippocampusVars('data/set_train')
hippocHisto = extractHippocampusHistograms('data/set_train')

### allFeatures.append(histo)
allFeatures.append(flipzones)
#allFeatures.append(blackzone)
#allFeatures.append(whitegray)
allFeatures.append(grayzone)
allFeatures.append(hippocVar)
allFeatures.append(hippocMean)
allFeatures.append(hippocMedian)
allFeatures.append(hippocHisto)

# ==========================================
#				  TEST FEATURES
# ==========================================
# TO BE ADDED: Gray white ratio; black zones from center of brain

allTestFeatures = []

### testHisto = extractHistograms('data/set_test', 4000, 45, 9)
testFlipzones = extractFlipSim('data/set_test')
print "Shape of testFlipzones:",np.asarray(testFlipzones).shape
#blackzone = extractColoredZone('data/set_train', )
testGrayzone = extractColoredZone('data/set_test', 450, 800, 8)
print "Shape of testGrayzone:",np.asarray(testGrayzone).shape
testHippocMedian = extractHippocampusMedians('data/set_test')
print "Shape of testHippocMedian:",np.asarray(testHippocMedian).shape
testHippocMean = extractHippocampusMeans('data/set_test')
print "Shape of testHippocMean:",np.asarray(testHippocMean).shape
testHippocVar = extractHippocampusVars('data/set_test')
print "Shape of testHippocVar:",np.asarray(testHippocVar).shape
testHippocHisto = extractHippocampusHistograms('data/set_test')
print "Shape of testHippocHisto:",np.asarray(testHippocHisto).shape

### allTestFeatures.append(testHisto)
allTestFeatures.append(testFlipzones)
#allTestFeatures.append(blackzone)
#allTestFeatures.append(whitegray)
allTestFeatures.append(testGrayzone)
allTestFeatures.append(testHippocVar)
allTestFeatures.append(testHippocMean)
allTestFeatures.append(testHippocMedian)
allTestFeatures.append(testHippocHisto)


# ==========================================
# 				 PIPELINES
# ==========================================

allPipelines = []

### histoPipeline = pipeline.make_pipeline(PCA(n_components=1400), StandardScaler())
flipzonePipeline  = pipeline.make_pipeline(PCA(n_components=10))
#blackzonePipeline 
grayzonePipeline  = pipeline.make_pipeline(PCA(n_components=100))
hippocVariancePipeline  = pipeline.make_pipeline(PCA(n_components=1))
hippocMeanPipeline  = pipeline.make_pipeline(PCA(n_components=1))
hippocMedianPipeline  = pipeline.make_pipeline(PCA(n_components=1))
hippocHistoPipeline  = pipeline.make_pipeline(PCA(n_components=10))


### allPipelines.append(histoPipeline)
allPipelines.append(flipzonePipeline)
#allPipelines.append(blackzonePipeline)
allPipelines.append(grayzonePipeline)
allPipelines.append(hippocVariancePipeline)
allPipelines.append(hippocMeanPipeline)
allPipelines.append(hippocMedianPipeline)
allPipelines.append(hippocHistoPipeline)



# ==========================================
#				    SEX
# ==========================================

print "SEX!!!!!!!!!!!!!!!!"
# ------------- CLASSIFIERS
sexClassifierNames = ["RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",]
sexClassifiers = [
	RandomForestClassifier(max_depth=30,n_estimators=200),
	#SVC(kernel="linear", C=1.0, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	MLPClassifier(alpha=1),
]

# This call will also print cross val score for each classifier and a score for the voting classifiers
sexClassifier = createVotingClassifier(0, allFeatures, allPipelines, sexClassifierNames, sexClassifiers)



# ==========================================
#				    AGE
# ==========================================
print "AGE!!!!!!!!!!!!!!!!"

# ------------- CLASSIFIERS
ageClassifierNames = ["RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",]
ageClassifiers = [
	RandomForestClassifier(max_depth=30,n_estimators=200),
	#SVC(kernel="linear", C=1.0, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	MLPClassifier(alpha=1),
]

# This call will also print cross val score for each classifier and a score for the voting classifier
ageClassifier = createVotingClassifier(1, allFeatures, allPipelines, ageClassifierNames, ageClassifiers)


# ==========================================
#				    HEALTH
# ==========================================
print "HEALTH!!!!!!!!!!!!!!!!"

# ------------- CLASSIFIERS
healthClassifierNames = ["RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",]
healthClassifiers = [
	RandomForestClassifier(max_depth=30,n_estimators=200),
	#SVC(kernel="linear", C=1.0, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	MLPClassifier(alpha=1),
]

# This call will also print cross val score for each classifier and a score for the voting classifier
healthClassifier = createVotingClassifier(2, allFeatures, allPipelines, healthClassifierNames, healthClassifiers)



# ==========================================
#				PREDICTIONS
# ==========================================

createPredictions(allTestFeatures, sexClassifier, ageClassifier, healthClassifier)
