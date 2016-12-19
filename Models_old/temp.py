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

#data = np.asarray(extractHistograms("../data/set_train",4500,45,9))
data = np.asarray(extractColoredZone("../data/set_train", 450, 800, 8))
print "Shape of data:"
print np.array(data).shape

print "Estimating error:"

models = {
	#"SVC (poly)" : SVC(kernel='poly', probability=True),
	#"RandomForestClassifier" : RandomForestClassifier(max_depth=20,n_estimators=700,max_features=200),
	#"SVC (rbf)" : SVC(kernel='rbf', gamma=2, probability=True),
	#"Gaussian" : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
	#"Neural net" : MLPClassifier(hidden_layer_sizes=(700,),activation='logistic'),
	#"AdaBoost" : AdaBoostClassifier(base_estimator=SVC(kernel='poly',probability=True), n_estimators=700)

	#"Nearest Neighbors":KNeighborsClassifier(3),
    #"Linear SVM":SVC(kernel="linear", C=0.025, probability=True),
    #"RBF SVM":SVC(gamma=2, C=1, probability=True),
    "Gaussian Process":GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    #"Decision Tree":DecisionTreeClassifier(max_depth=5),
    #"Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net":MLPClassifier(alpha=1),
    #"AdaBoost":AdaBoostClassifier(),
    #"Naive Bayes":GaussianNB(),
    #"QDA":QuadraticDiscriminantAnalysis()
}

voter = VotingClassifier(models.items(), voting='soft',n_jobs=-1)
voting_model = {"Voting Classifier": voter}

models.update(voting_model)

for key, model in sorted(models.items()):
	print(key)
	pl = pipeline.make_pipeline(
		PCA(n_components=2),
		StandardScaler(),
		model)
	scores = cross_val_score(pl, data, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)

test_data = np.asarray(extractColoredZone("../data/set_test", 450, 800, 8))
model = pipeline.make_pipeline(
		PCA(n_components=2),
		StandardScaler(),
		voter
		)

print "Fitting model"
model.fit(data,targets)
print "predicting"
predictions = model.predict_proba(test_data)
predictions_0, predictions_1 = zip(*predictions)
print predictions_1
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
