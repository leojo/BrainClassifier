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
from MultiLabelVotingClassifier import MultiLabelVotingClassifier


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

# ===================================================================================================
#				  					        HELPER FUNCTIONS
# ===================================================================================================

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

	
def createPredictions(testFeatures, weightedVotingClassifier):
	numTestSamples = np.asarray(testFeatures[0]).shape[0]
	predictions = weightedVotingClassifier.predict(testFeatures)

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

def plotImages(datasets, classifiers, names):
	figure = plt.figure(figsize=(27, 9))
	i = 1
	# iterate over datasets
	for ds_cnt, ds in enumerate(datasets):
	    # preprocess dataset, split into training and test part
	    X, y = ds
	    X = StandardScaler().fit_transform(X)
	    X_train, X_test, y_train, y_test = \
	        train_test_split(X, y, test_size=.4, random_state=42)

	    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                         np.arange(y_min, y_max, h))

	    # just plot the dataset first
	    cm = plt.cm.RdBu
	    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
	    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
	    if ds_cnt == 0:
	        ax.set_title("Input data")
	    ### Plot the training points
	    ### ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
	    ### and testing points
	    ### ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
	    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
	    # and testing points
	    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.6)
	    ax.set_xlim(xx.min(), xx.max())
	    ax.set_ylim(yy.min(), yy.max())
	    ax.set_xticks(())
	    ax.set_yticks(())
	    i += 1

	    # iterate over classifiers
	    for name, clf in zip(names, classifiers):
	        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
	        score = cross_val_score(clf, X, y, cv=10, scoring='neg_log_loss', n_jobs=-1)#clf.score(X_test, y_test)
	        clf.fit(X, y)

	        # Plot the decision boundary. For that, we will assign a color to each
	        # point in the mesh [x_min, x_max]x[y_min, y_max].
	        if hasattr(clf, "decision_function"):
	            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	        else:
	            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

	        # Put the result into a color plot
	        Z = Z.reshape(xx.shape)
	        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

	        # Plot also the training points
	        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
	        # and testing points
	        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
	                   alpha=0.6)

	        ax.set_xlim(xx.min(), xx.max())
	        ax.set_ylim(yy.min(), yy.max())
	        ax.set_xticks(())
	        ax.set_yticks(())
	        if ds_cnt == 0:
	            ax.set_title(name)
	        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % -score.mean()).lstrip('0'),
	                size=15, horizontalalignment='right')
	        i += 1

	plt.tight_layout()
	plt.show()

	# ===================================================================================================
#				  					        MAIN FUNCTIONALITY
# ===================================================================================================

if __name__=="__main__":

	shouldPlot = True

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

	histo = extractHistograms('data/set_train', 4000, 45, 9)
	flipzones = extractFlipSim('data/set_train')
	blackzone = extractBlackzones('data/set_train',nPartitions=3)
	grayzone = extractColoredZone3D('data/set_train', 450, 800, 8)
	hippocMedian = extractHippocampusMedians('data/set_train')
	hippocMean = extractHippocampusMeans('data/set_train')
	hippocVar = extractHippocampusVars('data/set_train')
	hippocHisto = extractHippocampusHistograms('data/set_train')

	allFeatures.append(histo)
	allFeatures.append(flipzones)
	allFeatures.append(blackzone)
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

	testHisto = extractHistograms('data/set_test', 4000, 45, 9)
	testFlipzones = extractFlipSim('data/set_test')
	blackzone = extractBlackzones('data/set_train',nPartitions=3)
	testGrayzone = extractColoredZone3D('data/set_test', 450, 800, 8)
	testHippocMedian = extractHippocampusMedians('data/set_test')
	testHippocMean = extractHippocampusMeans('data/set_test')
	testHippocVar = extractHippocampusVars('data/set_test')
	testHippocHisto = extractHippocampusHistograms('data/set_test')

	allTestFeatures.append(testHisto)
	allTestFeatures.append(testFlipzones)
	allTestFeatures.append(blackzone)
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

	histoPipeline = pipeline.make_pipeline(PCA(n_components=1400), StandardScaler())
	flipzonePipeline  = pipeline.make_pipeline(PCA(n_components=10))
	blackzonePipeline = pipeline.make_pipeline(PCA(n_components=10))
	grayzonePipeline  = pipeline.make_pipeline(PCA(n_components=10))
	hippocVariancePipeline  = pipeline.make_pipeline(PCA(n_components=1))
	hippocMeanPipeline  = pipeline.make_pipeline(PCA(n_components=1))
	hippocMedianPipeline  = pipeline.make_pipeline(PCA(n_components=1))
	hippocHistoPipeline  = pipeline.make_pipeline(PCA(n_components=10))


	allPipelines.append(histoPipeline)
	allPipelines.append(flipzonePipeline)
	allPipelines.append(blackzonePipeline)
	allPipelines.append(grayzonePipeline)
	allPipelines.append(hippocVariancePipeline)
	allPipelines.append(hippocMeanPipeline)
	allPipelines.append(hippocMedianPipeline)
	allPipelines.append(hippocHistoPipeline)

	# ==========================================
	# 				 CLASSIFIERS
	# ==========================================

	allClassifiers = []
	classifier_names = []

	gaussBased = OneVsRestClassifier(GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True))
	logBased = OneVsRestClassifier(LogisticRegression())
	rforestBased = OneVsRestClassifier(RandomForestClassifier(max_depth=30,n_estimators=200))
	mlpBased = OneVsRestClassifier(MLPClassifier(alpha=1))

	allClassifiers.append(gaussBased)
	classifier_names.append("Gaussian Process base")
	allClassifiers.append(logBased)
	classifier_names.append("Logistic Regression base")
	allClassifiers.append(rforestBased)
	classifier_names.append("Random Forest base")
	allClassifiers.append(mlpBased)
	classifier_names.append("MLP base")

	votingClassifier = createVotingClassifier(allFeatures,allPipelines,classifier_names,allClassifiers)
	votingClassifier.fit(allFeatures,targets)


	# ==========================================
	#				PREDICTIONS
	# ==========================================
	if(shouldPlot): plotImages(allFeatures, allClassifiers, classifier_names)

	createPredictions(allTestFeatures, votingClassifier)

