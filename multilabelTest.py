import csv
import numpy as np
from sklearn.pipeline import make_pipeline 
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from Features.extract_features import *
from Features.scoring import hammingLoss

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)


Y = np.array(targets).astype(np.int)
print "Shape of Y:",Y.shape

X = np.array(extractFlipSim("../data/set_train"))
print "Shape of X:",X.shape

classif = OneVsRestClassifier(
		make_pipeline(
			PCA(n_components=10),
			SVC(kernel='linear'),
		)
	)
classif.fit(X,Y)

number_of_test_samples = 10
number_of_tests  = 50

loss = []
for test in range(number_of_tests):
	data = zip(X,Y)
	np.random.shuffle(data)
	train_data = data[:-10]
	test_data = data[-10:]
	X_train, Y_train = zip(*train_data)
	X_test, Y_test = zip(*test_data)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	Y_train = np.array(Y_train)
	Y_test = np.array(Y_test)
	classif = OneVsRestClassifier(SVC(kernel='linear'))
	classif.fit(X_train,Y_train)
	Y_pred = classif.predict(X_test)
	loss.append(hammingLoss(Y_pred,Y_test))
	
mean = np.mean(loss)
std = np.std(loss)
print "Average loss:",mean,"+-"+str(std)

