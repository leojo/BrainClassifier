# =================================================================	#
#						IMPORTS										#
# =================================================================	#

import csv
import sys
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


from sklearn.ensemble import VotingClassifier

from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

from Features.extract_features import *
from Features.scoring import *

# =================================================================	#
#						HELPER FUNCTIONS							#
# =================================================================	#

def createPredictions(sexFeatures,ageFeatures,healthFeatures, sexEst, ageEst, healthEst):
	sexPredictions = sexEst.predict(sexFeatures)
	agePredictions = ageEst.predict(ageFeatures)
	agePredictedFeature = np.array(agePredictions).reshape(-1,1) # The predicted probability of age label = 1
	healthFeatures = np.concatenate((healthFeatures,agePredictedFeature),axis=1) # adding age prediction as feature
	healthPredictions = healthEst.predict(healthFeatures)
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


# =================================================================	#
#						LOAD LABELS									#
# =================================================================	#
with open('data/targets.csv', 'rb') as f:
	reader = csv.reader(f)
	targets = list(reader)

targets = np.array(targets).astype(np.int)
targets_sex, targets_age, targets_health = zip(*targets) 


# =================================================================	#
#						TRAIN FEATURES								#
# =================================================================	#
# Features for sex:		grayzone+hippoVar
# Features for age:		blackzone+grayzone+hippoHisto
# Features for health:	hippoMedian+blackzone+flipzones+hippoHisto

sexFeatures = []	
ageFeatures = []	
healthFeatures = []	

#histo = extractHistograms('data/set_train', 4000, 45, 9)
flipzones = extractFlipSim('data/set_train')
blackzone = extractBlackzones('data/set_train',nPartitions=3)
grayzone = extractColoredZone3D('data/set_train', 450, 800, 8)
grayWhiteRatio = extractGrayWhiteRatio('data/set_train', 8)
largeHippocMedian = extractLargeHippocampusMedians3D('data/set_train')
largeHippocMean = extractLargeHippocampusMeans3D('data/set_train')
largeHippocVar = extractLargeHippocampusVars3D('data/set_train')
largeHippocHisto = extractLargeHippocampusHistograms3D('data/set_train')
smallHippocMedian = extractSmallHippocampusMedians3D('data/set_train')
smallHippocMean = extractSmallHippocampusMeans3D('data/set_train')
smallHippocVar = extractSmallHippocampusVars3D('data/set_train')
smallHippocHisto = extractSmallHippocampusHistograms3D('data/set_train')
amygdalaMedian = extractAmygdalaMedians3D('data/set_train')
amygdalaMean = extractAmygdalaMeans3D('data/set_train')
amygdalaVar = extractAmygdalaVars3D('data/set_train')
amygdalaHisto = extractAmygdalaHistograms3D('data/set_train')


sexFeatures.append(blackzone)
sexFeatures.append(grayzone)
sexFeatures.append(grayWhiteRatio)
sexFeatures.append(amygdalaMedian)
sexFeatures.append(amygdalaMean)
sexFeatures.append(amygdalaVar)
sexFeatures.append(amygdalaHisto)


ageFeatures.append(blackzone)
ageFeatures.append(grayzone)
ageFeatures.append(grayWhiteRatio)
ageFeatures.append(largeHippocHisto)


healthFeatures.append(flipzones)
healthFeatures.append(grayzone)
healthFeatures.append(grayWhiteRatio)
healthFeatures.append(blackzone)
healthFeatures.append(largeHippocMedian)
healthFeatures.append(largeHippocHisto)
healthFeatures.append(np.array(targets_age).reshape(-1,1)) #Age group as feature for health

# =================================================================	#
#						TEST FEATURES								#
# =================================================================	#

sexFeatures_t = []	
ageFeatures_t = []	
healthFeatures_t = []	

#testHisto = extractHistograms('data/set_test', 4000, 45, 9)
testFlipzones = extractFlipSim('data/set_test')
testBlackzone = extractBlackzones('data/set_test',nPartitions=3)
testGrayzone = extractColoredZone3D('data/set_test', 450, 800, 8)
testGrayWhiteRatio = extractGrayWhiteRatio('data/set_test', 8)
testLargeHippocMedian = extractLargeHippocampusMedians3D('data/set_test')
testLargeHippocMean = extractLargeHippocampusMeans3D('data/set_test')
testLargeHippocVar = extractLargeHippocampusVars3D('data/set_test')
testLargeHippocHisto = extractLargeHippocampusHistograms3D('data/set_test')
testSmallHippocMedian = extractSmallHippocampusMedians3D('data/set_test')
testSmallHippocMean = extractSmallHippocampusMeans3D('data/set_test')
testSmallHippocVar = extractSmallHippocampusVars3D('data/set_test')
testSmallHippocHisto = extractSmallHippocampusHistograms3D('data/set_test')
testAmygdalaMedian = extractAmygdalaMedians3D('data/set_test')
testAmygdalaMean = extractAmygdalaMeans3D('data/set_test')
testAmygdalaVar = extractAmygdalaVars3D('data/set_test')
testAmygdalaHisto = extractAmygdalaHistograms3D('data/set_test')


sexFeatures_t.append(testBlackzone)
sexFeatures_t.append(testGrayzone)
sexFeatures_t.append(testGrayWhiteRatio)
sexFeatures_t.append(testAmygdalaMedian)
sexFeatures_t.append(testAmygdalaMean)
sexFeatures_t.append(testAmygdalaVar)
sexFeatures_t.append(testAmygdalaHisto)

ageFeatures_t.append(testBlackzone)
ageFeatures_t.append(testGrayzone)
ageFeatures_t.append(testGrayWhiteRatio)
ageFeatures_t.append(testLargeHippocHisto)


healthFeatures_t.append(testFlipzones)
healthFeatures_t.append(testGrayzone)
healthFeatures_t.append(testGrayWhiteRatio)
healthFeatures_t.append(testBlackzone)
healthFeatures_t.append(testLargeHippocMedian)
healthFeatures_t.append(testLargeHippocHisto)
####### remember to add age predictions as feature during final predictions

# =================================================================	#
#					CONCATENATE FEATURES							#
# =================================================================	#

first = True
Csex = None
Csex_t = None
for f, f_t in zip(sexFeatures,sexFeatures_t):
	if first:
		Csex = f
		Csex_t = f_t
		first = False
	else:
		Csex = np.concatenate((Csex,f), axis = 1)
		Csex_t = np.concatenate((Csex_t,f_t), axis=1)
		
print "Gender feature shapes:",Csex.shape," (Test",Csex_t.shape
		
first = True
Cage = None
Cage_t = None
for f, f_t in zip(ageFeatures,ageFeatures_t):
	if first:
		Cage = f
		Cage_t = f_t
		first = False
	else:
		Cage = np.concatenate((Cage,f), axis = 1)
		Cage_t = np.concatenate((Cage_t,f_t), axis=1)

print "Age feature shapes:",Cage.shape," (Test",Cage_t.shape

first = True
Chealth = None
Chealth_t = None
for f, f_t in zip(healthFeatures,healthFeatures_t):
	if first:
		Chealth = f
		Chealth_t = f_t
		first = False
	else:
		Chealth = np.concatenate((Chealth,f), axis = 1)
		Chealth_t = np.concatenate((Chealth_t,f_t), axis=1)

Chealth = np.concatenate((Chealth,healthFeatures[-1]), axis = 1)

print "Health feature shapes:",Chealth.shape," (Test",Chealth_t.shape

# =================================================================	#
#						   ESTIMATORS				 				#
# =================================================================	#

estA = 	make_pipeline(
			VarianceThreshold(),
			PCA(n_components=100),
			VotingClassifier(estimators = [
				("LogisticRegression", LogisticRegression()),
				("Gaussian", GaussianProcessClassifier(0.5 * RBF(1.0), warm_start=True)),
				("Naive Bayes",  GaussianNB()),
				("NeuralNet", MLPClassifier(alpha=1.0)),
				("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10))
				], voting = "hard")
			)
		
estB = 	make_pipeline(
			VarianceThreshold(),
			SelectKBest(k=250),
			VotingClassifier(estimators = [
				("LogisticRegression", LogisticRegression()),
				("Naive Bayes",  GaussianNB()),
				("Gaussian", GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
				("NeuralNet", MLPClassifier(alpha=1)),
				("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10))
				], voting = "hard", weights=[1, 2, 3, 3, 3])
			)

estC = make_pipeline(
			VarianceThreshold(),
			SelectKBest(k=250),
			RandomOverSampler(),
			VotingClassifier(estimators = [
				("LogisticRegression", LogisticRegression()),
				("GaussianProcess", GaussianProcessClassifier(0.7* RBF(0.5), warm_start=True)),
				("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10)),
				("NeuralNet", MLPClassifier(alpha=1)),
				("Naive Bayes", GaussianNB())
				], voting = "hard")
			)


#estA = clone(est)
#estB = clone(est)
#estC = clone(est)

scorer = make_scorer(partialHammingLoss,greater_is_better=False)

print "Calculating scores..."
scoreA = cross_val_score(estA, Csex, targets_sex, cv=10, scoring=scorer)
scoreB = cross_val_score(estB, Cage, targets_age, cv=10, scoring=scorer)
scoreC = cross_val_score(estC, Chealth, targets_health, cv=10, scoring=scorer)
print "Done\n"
print "Scores:"
print "%0.2f (+/- %0.2f) [%s]" % (-scoreA.mean(), scoreA.std(), 'Gender')
print "%0.2f (+/- %0.2f) [%s]" % (-scoreB.mean(), scoreB.std(), 'Age')
print "%0.2f (+/- %0.2f) [%s]" % (-scoreC.mean(), scoreC.std(), 'Health')
print "%0.2f (+/- %0.2f) [%s]" % (-(scoreA.mean()+scoreB.mean()+scoreC.mean())/3.0, (scoreA.std()+scoreB.std()+scoreC.std())/3.0, 'Combined')


# =================================================================	#
#						GENERATE PREDICTIONS						#
# =================================================================	#

print "\nCreating predictions"
estA.fit(Csex,targets_sex)
estB.fit(Cage,targets_age)
estC.fit(Chealth,targets_health)
createPredictions(Csex_t,Cage_t,Chealth_t,estA,estB,estC)