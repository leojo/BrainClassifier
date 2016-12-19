import csv
import numpy as np
from sklearn import pipeline
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.model_selection import cross_val_score

from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array([float(x[0]) for x in targets])

histograms = np.array(extractHistograms("../data/set_train",4500,45,9))

print "Training model"
model = pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=1400),
		PCA(n_components=100),
		SVC(kernel='poly',probability=True)
	)
model.fit(histograms,targets)

print "Testing model"
testData = np.array(extractHistograms('../data/set_test',4500,45,9))
predictions = model.predict_proba(testData)
predictions_0, predictions_1 = zip(*predictions)
print predictions_1
with open('svc.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions_1)):
		id = str(i+1)
		p = str(predictions_1[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()