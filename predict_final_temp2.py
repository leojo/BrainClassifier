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
from sklearn.linear_model import LogisticRegression

from Features.extract_features import *
from Features.scoring import hammingLoss
from Features.scoring import partialHammingLoss
from multi_label_voting_classifier import MultiLabelVotingClassifier


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

import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_decomposition import CCA


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

# ===================================================================================================
#				  					        HELPER FUNCTIONS
# ===================================================================================================


def plot_hyperplane(clf, min_x, max_x, linestyle, label, col):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label, color=col)


def plot_subfigure(X, Y, subplot, title, transform, classif):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    #classif = OneVsRestClassifier(SVC(kernel='linear'))
    print Y.shape
    print X.shape
    classif.fit(X, Y)
    plt.subplot(1, 1, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    zero_not_class = np.where(1-Y[:, 0])
    one_class = np.where(Y[:, 1])
    one_not_class = np.where(1-Y[:, 1])
    two_class = np.where(Y[:, 2])
    two_not_class = np.where(1-Y[:, 2])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray')
    plt.scatter(X[zero_class, 0], X[zero_class, 1], edgecolors='black',
               facecolors='black', linewidths=2, label='Class 1')
    plt.scatter(X[zero_not_class, 0], X[zero_not_class, 1], edgecolors='orange',
               facecolors='orange', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=500, edgecolors='green',
               facecolors='none', linewidths=2, label='Class 2')
    plt.scatter(X[two_class, 0], X[two_class, 1], s=340, edgecolors='red', marker='<',
               facecolors='none', linewidths=2, label='Class 3')
    plt.scatter(X[two_not_class, 0], X[two_not_class, 1], s=300, edgecolors='blue', marker='>',
               facecolors='none', linewidths=2, label='Class 3')


    #plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k-.', 'Boundary\nfor class 1', 'orange')
    #plot_hyperplane(classif.estimators_[1], min_x, max_x, 'ko', 'Boundary\nfor class 2', 'green')
    #plot_hyperplane(classif.estimators_[2], min_x, max_x, 'k>', 'Boundary\nfor class 3', 'blue')


    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")

def makePlot(X, Y, classif, plotNum):

    plt.figure(figsize=(8, 6))

    #plot_subfigure(X, Y, plotNum, "Without unlabeled samples + CCA", "cca", classif)
    plot_subfigure(X, Y, plotNum, "Without unlabeled samples + PCA", "pca", classif)
    
    plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
    plt.show()

def makeAllPlots(allX, Y, classif):
	for i in range(0, len(allX)):
		makePlot(allX[i], Y, classif, 1)

# Create a voting classifier from several classifiers for EACH feature, then
# create a voting classifier from these feature-specific classifiers
def createVotingClassifier(features, pipelines, names, classifiers):
	outputFileName = "votingClassifier1.classifier"
	if os.path.isfile(outputFileName):
		print "Found classifier save:",outputFileName
		save = open(outputFileName,'rb')
		save_data = pickle.load(save)
		votingClassifier = MultiLabelVotingClassifier(save_data[0],voting="hard", weights=save_data[1], separateFeatures = True)
		save.close()
		return votingClassifier

	voterWeights = []
	voters = []
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
			scores = cross_val_score(pl, feature, targets, cv=10, scoring=scorer, n_jobs=1)
			print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),name)
			weights.append(1.0/(-scores.mean()))

		model = pipeline.make_pipeline(
				preproc,
				MultiLabelVotingClassifier(classifiers, voting='soft', weights=weights)
				)

		print "\nCalculating score of model:"
		scorer = make_scorer(hammingLoss,greater_is_better=False)
		scores = cross_val_score(model, feature, targets, cv=10, scoring=scorer, n_jobs=1)
		print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),"VotingClassifier")
		voterWeights.append(1.0/(-scores.mean()))
		voters.append(model)
	print "!!!!!!!!!!!!!!!!!!! Creating final classifier !!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	save_data = [voters,voterWeights]
	votingClassifier = MultiLabelVotingClassifier(save_data[0],voting="hard", weights=save_data[1], separateFeatures = True)
	#scorer = make_scorer(hammingLoss,greater_is_better=False)
	#scores = cross_val_score(votingClassifier, features, targets, cv=10, scoring=scorer, n_jobs=1)
	#print "!!!!!!!!!!!!!!!!!!! Score of final classifier !!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
	#print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),"VotingClassifier")
	print "\nStoring the classifier in "+outputFileName
	output = open(outputFileName,"wb")
	pickle.dump(save_data,output)
	output.close()
	print "Done"
	return votingClassifier

	
def createPredictions(testFeatures, sexModel, ageModel, healthModel):
	numTestSamples = np.asarray(testFeatures[0]).shape[0]
	sexPredictions = sexModel[0].predict(testFeatures[sexModel[1]])
	agePredictions = ageModel[0].predict(testFeatures[ageModel[1]])
	healthPredictions = healthModel[0].predict(testFeatures[healthModel[1]])
	predictions = zip(sexPredictions,agePredictions,healthPredictions)
	print predictions

	id = 0
	resultFileName = 'submission'
	if len(sys.argv) == 2:
		resultFileName = sys.argv[1]
	if resultFileName[-4:] != ".csv":
		resultFileName += ".csv"
	with open(resultFileName, 'w') as csvfile:
		resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
		resultWriter.writerow(['ID','Sample','Label','Predicted'])
		for sample_no, sample in enumerate(predictions):
			for label, prediction in zip(['gender','age','health'],[bool(sample[0]),bool(sample[1]),bool(sample[2])]):
				row=[id,sample_no,label,prediction]
				resultWriter.writerow(row)
				id+=1
		csvfile.close()

# models array of model, model = [classifiers, classifier weights]
# models[i] is the model = classif, classifweights for feature testFeatures[i]
def createWeightedPrediction(testFeatures, models):
	numTestSamples = np.asarray(testFeatures[0]).shape[0]
	
	totalWeight = 0.0
	weightedPrediction = np.zeros(numTestSamples)
	for i in range(0, len(testFeatures)):
		feature = testFeatures[i]
		model = models[i]
		classifier = model[0]
		clWeight = model[1]

		bothPredictions = np.array(classifier.predict_proba(testFeatures[i]))
		print bothPredictions

		probaPredictions = bothPredictions[:,1]
		print probaPredictions
		weightedPrediction += np.array(probaPredictions)*clWeight
		totalWeight += clWeight

	return np.round(weightedPrediction/totalWeight)


# models array of model, model = [classifiers, classifier weights]
# models[i] is the model = classif, classifweights for feature testFeatures[i]
def createWeightedPredictions(testFeatures, sexModels, ageModels, healthModels):
	sexPredictions = createWeightedPrediction(testFeatures, sexModels)
	agePredictions = createWeightedPrediction(testFeatures, ageModels)
	healthPredictions = createWeightedPrediction(testFeatures, healthModels)

	weightedPredictions = zip(sexPredictions,agePredictions,healthPredictions)

	id = 0
	resultFileName = 'submission'
	if len(sys.argv) == 2:
		resultFileName = sys.argv[1]
	if resultFileName[-4:] != ".csv":
		resultFileName += ".csv"
	with open(resultFileName, 'w') as csvfile:
		resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
		resultWriter.writerow(['ID','Sample','Label','Predicted'])
		for sample_no, sample in enumerate(weightedPredictions):
			for label, prediction in zip(['gender','age','health'],[bool(sample[0]),bool(sample[1]),bool(sample[2])]):
				row=[id,sample_no,label,prediction]
				resultWriter.writerow(row)
				id+=1
		csvfile.close()




# Returns the best voting classifier, I.E. the classifier created from the feature that worked
# best for this label
def createSingleLabelVC(classLabel, features, pipelines, names, classifiers):
	bestVoter = None
	bestVoterScore = 1
	voterWeights = []
	voters = []
	# Create a model for each kind of feature and preprocessing pipeline for that feature:
	for i, feature, preproc in zip(range(len(features)), features, pipelines):
		
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

		#voterWeights.append(1.0/(-scores.mean()))
		#voters.append(model)
	#return np.array(zip(voters,voterWeights)) # Since the voters in this "model" require different feature-preprocessing it must be used manually

	# NOTE: Later on it would be better if we didn't return best, but a combination of the others, like before
		if(-scores.mean() < bestVoterScore):
			bestVoterScore = -scores.mean()
			model.fit(feature,targets[:,classLabel])
			bestVoter = [model, i]

	return bestVoter

# Returns the best voting classifier, I.E. the classifier created from the feature that worked
# best for this label
def createSingleLabelVCS(classLabel, features, pipelines, names, classifiers):

	voterWeights = []
	voters = []
	# Create a model for each kind of feature and preprocessing pipeline for that feature:
	for i, feature, preproc in zip(range(len(features)), features, pipelines):
		
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

# ===================================================================================================
#				  					        MAIN FUNCTIONALITY
# ===================================================================================================

if __name__=="__main__":

	shouldPlot = False

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

	#histo = extractHistograms('data/set_train', 4000, 45, 9)
	flipzones = extractFlipSim('data/set_train')
	blackzone = extractBlackzones('data/set_train',nPartitions=3)
	grayzone = extractColoredZone3D('data/set_train', 450, 800, 8)
	grayWhiteRatio = extractGrayWhiteRatio('data/set_train', 8)
	hippocMedian = extractHippocampusMedians('data/set_train')
	hippocMean = extractHippocampusMeans('data/set_train')
	hippocVar = extractHippocampusVars('data/set_train')
	hippocHisto = extractHippocampusHistograms('data/set_train')

	#allFeatures.append(histo)
	allFeatures.append(flipzones)
	allFeatures.append(blackzone)
	allFeatures.append(grayzone)
	allFeatures.append(grayWhiteRatio)
	allFeatures.append(hippocMedian)
	allFeatures.append(hippocMean)
	allFeatures.append(hippocVar)
	allFeatures.append(hippocHisto)

	# ==========================================
	#				  TEST FEATURES
	# ==========================================
	# TO BE ADDED: Gray white ratio; black zones from center of brain

	allTestFeatures = []

	#testHisto = extractHistograms('data/set_test', 4000, 45, 9)
	testFlipzones = extractFlipSim('data/set_test')
	testBlackzone = extractBlackzones('data/set_test',nPartitions=3)
	testGrayzone = extractColoredZone3D('data/set_test', 450, 800, 8)
	testGrayWhiteRatio  = extractGrayWhiteRatio('data/set_test', 8)
	testHippocMedian = extractHippocampusMedians('data/set_test')
	testHippocMean = extractHippocampusMeans('data/set_test')
	testHippocVar = extractHippocampusVars('data/set_test')
	testHippocHisto = extractHippocampusHistograms('data/set_test')

	#allTestFeatures.append(testHisto)
	allTestFeatures.append(testFlipzones)
	allTestFeatures.append(testBlackzone)
	allTestFeatures.append(testGrayzone)
	allTestFeatures.append(testGrayWhiteRatio)
	allTestFeatures.append(testHippocMedian)
	allTestFeatures.append(testHippocMean)
	allTestFeatures.append(testHippocVar)
	allTestFeatures.append(testHippocHisto)


	# ==========================================
	# 				 PIPELINES
	# ==========================================

	allPipelines = []

	#histoPipeline = pipeline.make_pipeline(PCA(n_components=1400), StandardScaler())
	flipzonePipeline  = pipeline.make_pipeline(PCA(n_components=10))
	blackzonePipeline = pipeline.make_pipeline(PCA(n_components=10))
	grayzonePipeline  = pipeline.make_pipeline(PCA(n_components=10))
	grayWhiteRatioPipeline = pipeline.make_pipeline(PCA(n_components=10))
	hippocVariancePipeline  = pipeline.make_pipeline(PCA(n_components=1))
	hippocMeanPipeline  = pipeline.make_pipeline(PCA(n_components=1))
	hippocMedianPipeline  = pipeline.make_pipeline(PCA(n_components=1))
	hippocHistoPipeline  = pipeline.make_pipeline(PCA(n_components=10))


	#allPipelines.append(histoPipeline)
	allPipelines.append(flipzonePipeline)
	allPipelines.append(blackzonePipeline)
	allPipelines.append(grayzonePipeline)
	allPipelines.append(grayWhiteRatioPipeline)
	allPipelines.append(hippocMedianPipeline)
	allPipelines.append(hippocMeanPipeline)
	allPipelines.append(hippocVariancePipeline)
	allPipelines.append(hippocHistoPipeline)

	# ==========================================
	# 				 CLASSIFIERS
	# ==========================================

	allClassifiers = []
	classifier_names = []

	gaussBased = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
	logBased = LogisticRegression()
	rforestBased = RandomForestClassifier(max_depth=30,n_estimators=200)
	mlpBased = MLPClassifier(alpha=1)

	allClassifiers.append(gaussBased)
	classifier_names.append("Gaussian Process base")
	allClassifiers.append(logBased)
	classifier_names.append("Logistic Regression base")
	#allClassifiers.append(rforestBased)
	#classifier_names.append("Random Forest base")
	allClassifiers.append(mlpBased)
	classifier_names.append("MLP base")

	#votingClassifier = createVotingClassifier(allFeatures,allPipelines,classifier_names,allClassifiers)
	#votingClassifier.fit(allFeatures,targets)

	# ==========================================
	# 				   SEX 
	# ==========================================

	#sexModel = createSingleLabelVC(0, allFeatures, allPipelines, classifier_names, allClassifiers)
	sexModels = createSingleLabelVCS(0, allFeatures, allPipelines, classifier_names, allClassifiers)
	
	
	# ==========================================
	# 				   AGE 
	# ==========================================


	#ageModel = createSingleLabelVC(1, allFeatures, allPipelines, classifier_names, allClassifiers)
	ageModels = createSingleLabelVCS(1, allFeatures, allPipelines, classifier_names, allClassifiers)


	# ==========================================
	# 				  HEALTH 
	# ==========================================


	#healthModel = createSingleLabelVC(2, allFeatures, allPipelines, classifier_names, allClassifiers)
	healthModels = createSingleLabelVCS(2, allFeatures, allPipelines, classifier_names, allClassifiers)

	# ==========================================
	#				PREDICTIONS
	# ==========================================
	if(shouldPlot): makeAllPlots(allFeatures, targets,OneVsRestClassifier(LogisticRegression()))

	createWeightedPredictions(allTestFeatures, sexModels, ageModels, healthModels)

