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
with open('src/data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array([float(x[0]) for x in targets])


flip = np.asarray(extractFlipSim("src/data/set_train"))
flip_pl = pipeline.make_pipeline(
		PCA(n_components=10),
		StandardScaler())
flip_method = (flip,flip_pl)

print "Got flips"
gray = np.asarray(extractColoredZone("src/data/set_train", 450, 800, 8))
gray_pl = pipeline.make_pipeline(
		PCA(n_components=3),
		StandardScaler())
gray_method = (gray,gray_pl)


print "Got gray zones"
methods = [flip_method,gray_method]

print "Training model"


voters=[]
voterWeights=[]
for data, prep in methods:
	names = [
		"RandomForestClassifier", "linear SVC", "Gaussian Process", "Neural Net",
	]

	classifiers = [
		RandomForestClassifier(max_depth=30,n_estimators=200),
		SVC(kernel="linear", C=1.0, probability=True),
		GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
		MLPClassifier(alpha=1),
	]

	weights = []
	for name, classifier in zip(names,classifiers):
		print(name)
		pl = pipeline.make_pipeline(
			prep,
			classifier)
		scores = cross_val_score(pl, data, targets, cv=10, scoring='neg_log_loss', n_jobs=1)
		print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),name)
		weights.append(1.0/(-scores.mean()))

	model = pipeline.make_pipeline(
			prep,
			VotingClassifier(zip(names,classifiers), voting='soft', weights=weights ,n_jobs=1)
			)

	print "\nCalculating score of model:"
	score = cross_val_score(model, data, targets, cv=10, scoring='neg_log_loss', n_jobs=1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-score.mean(), score.std(),"VotingClassifier")
	model.fit(data,targets)
	voterWeights.append(1.0/(-scores.mean()))
	voters.append(model)

print "Testing model"
flip_test = np.asarray(extractFlipSim("src/data/set_test"))
gray_test = np.asarray(extractColoredZone("src/data/set_test", 450, 800, 8))

test_data = [flip_test,gray_test]

predictions = np.asarray([])
for model,weight,data in zip(voters,voterWeights,test_data):
	predictions_single = model.predict_proba(data)
	predictions_0, predictions_1 = zip(*predictions_single)
	if(len(predictions) == 0): predictions = np.asarray(predictions_1)*weight
	else: predictions += np.asarray(predictions_1)*weight
predictions /= float(np.sum(voterWeights))

with open('final_sub.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(predictions[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()

