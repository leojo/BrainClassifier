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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


from sklearn.ensemble import VotingClassifier

from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

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
	healthPredictions = healthEst.predict(healthFeatures)
	healthPredictions[np.where(agePredictions)] = 1
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
targets_sex = np.array(targets_sex)
targets_age = np.array(targets_age)
old = np.where(1-targets_age)
young = np.where(targets_age)
#### YOUNG PEOPLE ALL HAVE HEALTHY BRAINS ACCORDING TO OUR DATA (i.e. age=1 => health=1)
targets_health_old = np.array(targets_health)[old]


# =================================================================	#
#						TRAIN FEATURES								#
# =================================================================	#
# Features for sex:		grayzone+hippoVar
# Features for age:		blackzone+grayzone+hippoHisto
# Features for health:	hippoMedian+blackzone+flipzones+hippoHisto

#histo = extractHistograms('data/set_train', 4000, 45, 9)
grayWhiteRatio = extractGrayWhiteRatio('data/set_train', 8)
flipzones = extractFlipSim('data/set_train')
blackzone = extractBlackzones('data/set_train',nPartitions=3)
grayzone = extractColoredZone3D('data/set_train', 450, 800, 8)
hippocMedian = extractSmallHippocampusMedians3D('data/set_train')
hippocMean = extractSmallHippocampusMeans3D('data/set_train')
hippocVar = extractSmallHippocampusVars3D('data/set_train')
hippocHisto = extractSmallHippocampusHistograms3D('data/set_train')


sexFeatures = []	
ageFeatures = []	
healthFeatures = []	

sexFeatures.append(grayzone)
sexFeatures.append(grayWhiteRatio)
sexFeatures.append(hippocVar)
	
ageFeatures.append(blackzone)
ageFeatures.append(grayzone)
ageFeatures.append(hippocHisto)

healthFeatures.append(flipzones)
healthFeatures.append(blackzone)
healthFeatures.append(hippocMedian)
healthFeatures.append(hippocHisto)
#healthFeatures.append(np.array(targets_age).reshape(-1,1)) #Age group as feature for health


allFeatures = []
#allFeatures.append(histo)
allFeatures.append(grayWhiteRatio)
allFeatures.append(flipzones)
allFeatures.append(blackzone)
allFeatures.append(grayzone)
allFeatures.append(hippocMedian)
allFeatures.append(hippocMean)
allFeatures.append(hippocVar)
allFeatures.append(hippocHisto)

# =================================================================	#
#						TEST FEATURES								#
# =================================================================	#

#testHisto = extractHistograms('data/set_test', 4000, 45, 9)
testGrayWhiteRatio = extractGrayWhiteRatio('data/set_test', 8)		
testFlipzones = extractFlipSim('data/set_test')
testBlackzone = extractBlackzones('data/set_test',nPartitions=3)
testGrayzone = extractColoredZone3D('data/set_test', 450, 800, 8)
testHippocMedian = extractSmallHippocampusMedians3D('data/set_test')
testHippocMean = extractSmallHippocampusMeans3D('data/set_test')
testHippocVar = extractSmallHippocampusVars3D('data/set_test')
testHippocHisto = extractSmallHippocampusHistograms3D('data/set_test')

sexFeatures_t = []	
ageFeatures_t = []	
healthFeatures_t = []	

sexFeatures_t.append(testGrayzone)
sexFeatures_t.append(testGrayWhiteRatio)
sexFeatures_t.append(testHippocVar)

ageFeatures_t.append(testBlackzone)
ageFeatures_t.append(testGrayzone)
ageFeatures_t.append(testHippocHisto)

healthFeatures_t.append(testFlipzones)
healthFeatures_t.append(testBlackzone)
healthFeatures_t.append(testHippocMedian)
healthFeatures_t.append(testHippocHisto)
####### remember to add age predictions as feature during final predictions

allFeatures_t = []	
#allFeatures_t.append(histo)
allFeatures_t.append(testGrayWhiteRatio)
allFeatures_t.append(testFlipzones)
allFeatures_t.append(testBlackzone)
allFeatures_t.append(testGrayzone)
allFeatures_t.append(testHippocMedian)
allFeatures_t.append(testHippocMean)
allFeatures_t.append(testHippocVar)
allFeatures_t.append(testHippocHisto)


# =================================================================	#
#					CONCATENATE FEATURES							#
# =================================================================	#

first = True
C = None
C_t = None
for f, f_t in zip(allFeatures,allFeatures_t):
	if first:
		C = f
		C_t = f_t
		first = False
	else:
		C = np.concatenate((C,f), axis = 1)
		C_t = np.concatenate((C_t,f_t), axis=1)
		
print "Feature shape:",C.shape," (Test",C_t.shape,")"

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
		
print "Gender feature shapes:",Csex.shape," (Test",Csex_t.shape,")"
		
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

print "Age feature shapes:",Cage.shape," (Test",Cage_t.shape,")"

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
#print np.where(np.array(targets_age))
#print np.where(1-np.array(targets_age))
Chealth_o = Chealth[old]
print "Health feature shapes:",Chealth.shape," (Test",Chealth_t.shape,")"
print "Old:",Chealth_o.shape

# =================================================================	#
#						ESTIMATORS									#
# =================================================================	#
		
estA = make_pipeline(
			VarianceThreshold(),
			SelectKBest(k=250),
			VotingClassifier(estimators = [
				("SVC", SVC(kernel="linear")),
				("LogisticRegression", LogisticRegression()),
				("Gaussian", GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
				("NeuralNet", MLPClassifier(alpha=1)),
				("RandomForest", RandomForestClassifier(max_depth=30,n_estimators=200))
				], voting = "hard")
			)
estB = make_pipeline(
			VarianceThreshold(),
			SelectKBest(k=250),
			VotingClassifier(estimators = [
				("LogisticRegression", LogisticRegression()),
				("GaussianProcess", GaussianProcessClassifier(0.323 * RBF(0.4), warm_start=True)),
				("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10)),
				("NeuralNet", MLPClassifier(alpha=1)),
				("Naive Bayes", GaussianNB())
				], voting = "soft", weights = [1,2,3,3,3])
			)
estC = make_pipeline(
			VarianceThreshold(),
			SelectKBest(k=250),
			VotingClassifier(estimators = [
				("LogisticRegression", LogisticRegression()),
				#("Poly SVC", SVC(kernel="poly")),
				#("GaussianProcess", GaussianProcessClassifier(0.323 * RBF(0.4), warm_start=True)),
				("GaussianProcess", GaussianProcessClassifier(1.0 * RBF(0.4), warm_start=True)),
				("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10)),
				("NeuralNet", MLPClassifier(alpha=1)),
				("Naive Bayes", GaussianNB())
				], voting = "hard")
			)

scorer = make_scorer(partialHammingLoss,greater_is_better=False)

print "Calculating scores..."
N = 20
scoreA = cross_val_score(estA, C, targets_sex, cv=N, scoring=scorer)
scoreB = cross_val_score(estB, C, targets_age, cv=N, scoring=scorer)
scoreC = cross_val_score(estC, C[old], targets_health_old, cv=2*N, scoring=scorer)
n_young = len(young)
n_old = len(old)
scoreC = (n_old*scoreC)/float(n_old+n_young) ## ATTEMPT TO ESTIMATE ACTUAL SCORE OF HEALTH PREDICTION
print "Done\n"
print "Scores:"
print "%0.5f (+/- %0.5f) [%s]" % (-scoreA.mean(), scoreA.std(), 'Gender')
print "%0.5f (+/- %0.5f) [%s]" % (-scoreB.mean(), scoreB.std(), 'Age')
print "%0.5f (+/- %0.5f) [%s]" % (-scoreC.mean(), scoreC.std(), 'Health')
print "%0.5f (+/- %0.5f) [%s]" % (-(scoreA.mean()+scoreB.mean()+scoreC.mean())/3.0, (scoreA.std()+scoreB.std()+scoreC.std())/3.0, 'Combined')

# =================================================================	#
#						GENERATE PREDICTIONS						#
# =================================================================	#

print "\nCreating predictions"
estA.fit(C,targets_sex)
estB.fit(C,targets_age)
estC.fit(C[old],targets_health_old)
createPredictions(C_t,C_t,C_t,estA,estB,estC)