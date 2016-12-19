import csv
import numpy as np
from sklearn import pipeline
from sklearn.svm import SVC
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

from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array([float(x[0]) for x in targets])

hists = np.asarray(extractHistograms("../data/set_train",4500,45,9))
hists_pl = pipeline.make_pipeline(
		SelectKBest(k=20000),
		PCA(n_components=1400),
		StandardScaler())
hists_method = (hists,hists_pl)
flip = np.asarray(extractFlipSim("../data/set_train"))
flip_pl = pipeline.make_pipeline(
		PCA(n_components=10),
		StandardScaler())
flip_method = (flip,flip_pl)
gray = np.asarray(extractColoredZone("../data/set_train", 450, 800, 8))
gray_pl = pipeline.make_pipeline(
		PCA(n_components=2),
		StandardScaler())
gray_method = (gray,gray_pl)

#methods = [hists_method,flip_method,gray_method]
methods = [flip_method,gray_method]


print "Training model"


voters=[]
voterWeights=[]
for data, prep in methods:
	names = [
		"Decision Tree","RandomForestClassifier","AdaBoostClassifier (linearSVM)","AdaBoostClassifier (polySVM)",
		"Poly SVM","Linear SVM", "Gaussian Process", "Neural Net","Naive Bayes","QDA"
	]

	classifiers = [
		DecisionTreeClassifier(max_depth=30),
		RandomForestClassifier(max_depth=30,n_estimators=200),
		AdaBoostClassifier(base_estimator=SVC(kernel='linear',probability=True), n_estimators=200),
		AdaBoostClassifier(base_estimator=SVC(kernel='poly',probability=True), n_estimators=200),
		SVC(kernel="poly", C=1.0, probability=True),
		SVC(kernel="linear", C=0.025, probability=True),
		GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
		MLPClassifier(alpha=1),
		GaussianNB(),
		QuadraticDiscriminantAnalysis()
	]

	weights = []
	for name, classifier in zip(names,classifiers):
		print(name)
		pl = pipeline.make_pipeline(
			prep,
			classifier)
		scores = cross_val_score(pl, data, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
		print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),name)
		weights.append(1.0/(-scores.mean()))

	model = pipeline.make_pipeline(
			prep,
			VotingClassifier(zip(names,classifiers), voting='soft', weights=weights ,n_jobs=-1)
			)

	print "\nCalculating score of model:"
	score = cross_val_score(model, data, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-score.mean(), score.std(),"VotingClassifier")
	model.fit(data,targets)
	voterWeights.append(1.0/(-scores.mean()))
	voters.append(model)

print "Testing model"
hists_test = np.array(extractHistograms("../data/set_test",4500,45,9))
flip_test = np.asarray(extractFlipSim("../data/set_test"))
gray_test = np.asarray(extractColoredZone("../data/set_test", 450, 800, 8))
#test_data = [hists_test,flip_test,gray_test]
test_data = [flip_test,gray_test]

predictions = np.asarray([])
for model,weight,data in zip(voters,voterWeights,test_data):
	predictions_single = model.predict_proba(data)
	predictions_0, predictions_1 = zip(*predictions_single)
	if(len(predictions) == 0): predictions = np.asarray(predictions_1)*weight
	else: predictions += np.asarray(predictions_1)*weight
predictions /= float(np.sum(voterWeights))

with open('trippleVotingClassifier.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(predictions[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()

