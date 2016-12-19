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

data = np.asarray(extractZoneAverages("../data/set_train", 8))
print "Shape of data:"
print np.array(data).shape

print "Estimating error:"

models = {
	"AdaBoost (scaler - select 300 - PCA 100)" : pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=100),
		PCA(n_components=2),
		AdaBoostClassifier()
	),

	"SVM (select 300 - PCA 100)": pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=100),
		PCA(n_components=2),
		svm.SVC(kernel='poly', degree=2, C=1.0, probability=True)
	)
}

for key, model in sorted(models.items()):
	scores = cross_val_score(model, data, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)

