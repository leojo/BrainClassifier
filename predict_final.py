import csv
import numpy as np
from sklearn import pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

from Features.extract_features import *
from Features.scoring import hammingLoss

# Get the targets
with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets).astype(np.int)


flip = np.asarray(extractFlipSim("data/set_train"))
flip_pl = pipeline.make_pipeline(
		PCA(n_components=10),
		StandardScaler())
flip_method = (flip,flip_pl)

print "Got flips"
gray = np.asarray(extractColoredZone("data/set_train", 450, 800, 8))
gray_pl = pipeline.make_pipeline(
		PCA(n_components=3),
		StandardScaler())
gray_method = (gray,gray_pl)


print "Got gray zones"
methods = [flip_method,gray_method]

print "Evaluating model"
model = OneVsRestClassifier(SVC(kernel='linear'))
scorer = make_scorer(hammingLoss,greater_is_better=False)
scores = cross_val_score(model, flip, targets, cv=10, scoring=scorer, n_jobs=1)
print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),"OneVsRestClassifier")
exit()
print "Training model"


print "Making predictions"
flip_test = np.asarray(extractFlipSim("data/set_test"))
gray_test = np.asarray(extractColoredZone("data/set_test", 450, 800, 8))

test_data = [flip_test,gray_test]


with open('final_sub.csv', 'w') as csvfile:
	resultWriter = csv.writer(csvfile, delimiter=',', quotechar='|')
	resultWriter.writerow(['ID','Prediction'])
	for i in range(0,len(predictions)):
		id = str(i+1)
		p = str(predictions[i])
		row = [id,p]
		resultWriter.writerow(row)
	csvfile.close()

