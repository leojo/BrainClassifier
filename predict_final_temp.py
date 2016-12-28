import csv
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
#				   TARGETS
# ==========================================

# Get the targets
with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets).astype(np.int)
targets_age, targets_health, targets_sex = zip(*targets) 

# ==========================================
#				  FEATURES
# ==========================================

allFeatures = []

histo 
flipzones
blackzone
whiteGray
hippocVariance
hippocMean
hippocMedian
hippocHisto


allFeatures.append(histo)
allFeatures.append(flipzones)
allFeatures.append(blackzone)
allFeatures.append(whitegray)
allFeatures.append(hippocVariance)
allFeatures.append(hippocMean)
allFeatures.append(hippocMedian)
allFeatures.append(hippocHisto)

# ==========================================
# 				 PIPELINES
# ==========================================

allPipelines = []

histoPipeline 
flipzonePipeline 
blackzonePipeline 
whiteGrayPipeline 
hippocVariancePipeline 
hippocMeanPipeline 
hippocMedianPipeline 
hippocHistoPipeline 


allPipelines.append(histoPipeline)
allPipelines.append(flipzonesPipeline)
allPipelines.append(blackzonePipeline)
allPipelines.append(whitegrayPipeline)
allPipelines.append(hippocVariancePipeline)
allPipelines.append(hippocMeanPipeline)
allPipelines.append(hippocMedianPipeline)
allPipelines.append(hippocHistoPipeline)


# ==========================================
#				    AGE
# ==========================================

ageFeatures, agePipelines = getBestFeatures(0)


# ------------- CLASSIFIERS
ageClassifierNames = ["RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",]
ageClassifiers = [
	RandomForestClassifier(max_depth=30,n_estimators=200),
	SVC(kernel="linear", C=1.0, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	MLPClassifier(alpha=1),
]

# This call will also print cross val score for each classifier and a score for the voting classifier
ageClassifier = createVotingClassifier(0, ageFeatures, agePipelines, ageClassifierNames, ageClassifiers)


# ==========================================
#				    HEALTH
# ==========================================

healthFeatures, agePipelines = getBestFeatures(1)


# ------------- CLASSIFIERS
ageClassifierNames = ["RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",]
ageClassifiers = [
	RandomForestClassifier(max_depth=30,n_estimators=200),
	SVC(kernel="linear", C=1.0, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	MLPClassifier(alpha=1),
]

# This call will also print cross val score for each classifier and a score for the voting classifier
ageClassifier = createVotingClassifier(1, ageFeatures, agePipelines, ageClassifierNames, ageClassifiers)



# ==========================================
#				    SEX
# ==========================================

ageFeatures, agePipelines = getBestFeatures(2)


# ------------- CLASSIFIERS
ageClassifierNames = ["RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",]
ageClassifiers = [
	RandomForestClassifier(max_depth=30,n_estimators=200),
	SVC(kernel="linear", C=1.0, probability=True),
	GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	MLPClassifier(alpha=1),
]

# This call will also print cross val score for each classifier and a score for the voting classifiers
ageClassifier = createVotingClassifier(2, ageFeatures, agePipelines, ageClassifierNames, ageClassifiers)



# ==========================================
#				PREDICTIONS
# ==========================================

createOutputFile(ageClassifier, healthClassifier, sexClassifier)

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
	# Create a model for each kind of feature and preprocessing pipeline for that feature:
	for feature, preproc in zip(features, pipelines):
		
		# We want the model to be a voter combined from several classifiers:
		weights = []
		for name, classifier in zip(names,classifiers):
			print(name)
			pl = pipeline.make_pipeline(
				preproc,
				classifier)

			scorer = make_scorer(hammingLoss,greater_is_better=False)
			scores = cross_val_score(pl, feature, targets[classLabel], cv=10, scoring=scorer, n_jobs=1)
			print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),name)
			weights.append(1.0/(-scores.mean()))

		model = pipeline.make_pipeline(
				preproc,
				VotingClassifier(zip(names,classifiers), voting='soft', weights=weights ,n_jobs=1)
				)

		print "\nCalculating score of model:"
		score = cross_val_score(model, data, targets, cv=10, scoring='neg_log_loss', n_jobs=1)
		print "score: %0.2f (+/- %0.2f) [%s]" % (-score.mean(), score.std(),"VotingClassifier")
		model.fit(data,targets)
		voterWeights.append(1.0/(-scores.mean()))
		voters.append(model)