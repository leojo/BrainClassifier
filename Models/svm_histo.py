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
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.asarray([int(x[0]) for x in targets])

data = np.asarray(extractHistograms("../data/set_train", 4500, 45, 16))
print "Shape of data:"
print np.array(data).shape

print "Estimating error:"

models = {
	"AdaBoost (scaler - select 1400 - PCA 100)" : pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=1400),
		PCA(n_components=100),
		AdaBoostClassifier()
	),

	"SVM (poly - select 1400 - PCA 100)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=1400),
		PCA(n_components=100),
		svm.SVC(kernel='poly', degree=3, C=1.0, probability=True)
	),
	"SVM (poly - select 1400 - PCA 500)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=1400),
		PCA(n_components=500),
		svm.SVC(kernel='poly', degree=3, C=1.0, probability=True)
	),
	"SVM (poly - select 1400 - PCA 1000)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=1400),
		PCA(n_components=1000),
		svm.SVC(kernel='poly', degree=3, C=1.0, probability=True)
	),
	"SVM (rbf - select 1400 - PCA 100)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=5000),
		PCA(n_components=100),
		svm.SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
	),
	"SVM (rbf - select 1400 - PCA 500)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=5000),
		PCA(n_components=500),
		svm.SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
	),
	"SVM (rbf - select 1400 - PCA 1000)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=5000),
		PCA(n_components=1000),
		svm.SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
	),
	"SVM (rbf - select 1400 - PCA 2000)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=5000),
		PCA(n_components=2000),
		svm.SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
	),
}

for key, model in sorted(models.items()):
	scores = cross_val_score(model, data, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)


