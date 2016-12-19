import csv
import numpy as np
from random import shuffle
from sklearn import pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.neural_network import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.asarray([int(x[0]) for x in targets])

histo = np.asarray(extractHistograms("../data/set_train",1500,45,16))
color = np.asarray(extractColoredZone3D("../data/set_train", 450, 800, 8))

print "We have extracted the data."

#print "Shape of data:"
#print np.array(data).shape

print "Estimating error:"

color_models = {
	#"SVC (poly)" : SVC(kernel='poly', probability=True),
	#"RandomForestClassifier" : RandomForestClassifier(max_depth=20,n_estimators=700,max_features=200),
	#"SVC (rbf)" : SVC(kernel='rbf', gamma=2, probability=True),
	"Gaussian" : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	#"Neural net" : MLPClassifier(hidden_layer_sizes=(700,),activation='logistic'),
	"AdaBoost" : AdaBoostClassifier(base_estimator=SVC(kernel='poly',probability=True), n_estimators=700),
	#"Nearest Neighbors":KNeighborsClassifier(3),
    "Linear SVM":SVC(kernel="poly", C=0.025, probability=True),
    "Poly SVM":SVC(kernel="poly", C=1.0, probability=True),
    "RBF SVM":SVC(gamma=2, C=1, probability=True),
    "Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    #"Decision Tree":DecisionTreeClassifier(max_depth=5),
    #"Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net":MLPClassifier(alpha=1),
    "AdaBoost":AdaBoostClassifier(),
    #"Naive Bayes":GaussianNB(),
    #"QDA":QuadraticDiscriminantAnalysis()
}

histo_models = {
	"Poly SVM":SVC(kernel="poly", C=1.0, probability=True),
	"Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
}

voter_color = VotingClassifier(color_models.items(), voting='soft',n_jobs=-1)
voter_histo = VotingClassifier(histo_models.items(), voting='soft',n_jobs=-1)

voting_model_color = {"Voting Classifier color": voter_color}
voting_model_histo = {"Voting Classifier histo": voter_histo}

color_models.update(voting_model_color)
histo_models.update(voting_model_histo)

for key, model in sorted(color_models.items()):
	print(key)
	pl = pipeline.make_pipeline(
		PCA(n_components=2),
		StandardScaler(),
		model)
	scores = cross_val_score(pl, color, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)

for key, model in sorted(histo_models.items()):
	print(key)
	pl = pipeline.make_pipeline(
		PCA(n_components=2),
		StandardScaler(),
		model)
	scores = cross_val_score(pl, histo, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)

print "Train voters"
voter_color.fit(color,targets)
voter_histo.fit(histo,targets)

print "Combining voters"
testData_color = np.array(extractColoredZone3D("../data/set_test", 450, 800, 8))
testData_histo = np.array(extractHistograms('../data/set_test',1500,45,9))
predictions_color = voter_color.predict_proba(testData_color)
predictions_histo = voter_histo.predict_proba(testData_histo)
predictions_color_0, predictions_color_1 = zip(*predictions_color)
predictions_histo_0, predictions_histo_1 = zip(*predictions_histo)
predictions_1 = (np.asarray(predictions_histo_1)+np.asarray(predictions_color_1))/2
print predictions_1
with open('combinedVoters.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions_1)):
		id = str(i+1)
		p = str(predictions_1[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()


#hist_targ = zip(data,targets)
#shuffle(hist_targ)
#n_test_data = 30
#n_tests = 100
#errors = {}
#for i in range(n_tests):
#	X_shuffled, y_shuffled = zip(*hist_targ)
#	X_train = X_shuffled[:-n_test_data]
#	y_train = y_shuffled[:-n_test_data]
#	X_test = X_shuffled[-n_test_data:]
#	y_test = y_shuffled[-n_test_data:]
#	for key, model in models.items():
#		model.fit(X_train,y_train)
#		prediction = np.array(model.predict(X_test))
#		truth = np.array(y_test)
#		difference = np.absolute(prediction-truth)
#		average_error = sum(difference)
#		if key in errors:
#			errors[key] += (average_error*1.)/n_tests
#		else:
#			errors[key] = (average_error*1.)/n_tests
#
#for key, error in sorted(errors.items()):
#	print "Average number of errors: %f [%s]"%(error,key)
