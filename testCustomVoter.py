import csv
import numpy as np

from Features.extract_features import *

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from multi_label_voting_classifier import MultiLabelVotingClassifier

with open('data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.array(targets).astype(np.int)

data1 = extractHippocampusHistograms("data/set_train")
data2 = extractHippocampusMeans("data/set_train")
data3 = extractHippocampusVars("data/set_train")

y = targets[:-10]
y_test = targets[-10:]

XX = data1[:-10]
XX_test = data1[-10:]

X = [data1[:-10],data2[:-10],data3[:-10]]
X_test = [data1[-10:],data2[-10:],data3[-10:]]

names = ["LogisticRegression","SVC","DecisionTreeClassifier"]
clf1 = OneVsRestClassifier(LogisticRegression())
clf2 = OneVsRestClassifier(SVC(probability=True,kernel="linear"))
clf3 = OneVsRestClassifier(DecisionTreeClassifier())
clfs = [clf1,clf2,clf3]

clf1.fit(X[0],y)

votingClassifierHard = MultiLabelVotingClassifier(zip(names,clfs))
votingClassifierSoft = MultiLabelVotingClassifier(zip(names,clfs),voting='soft')

votingClassifier2Hard = MultiLabelVotingClassifier(zip(names,clfs))
votingClassifier2Soft = MultiLabelVotingClassifier(zip(names,clfs),voting='soft')

votingClassifierHard.fit(X,y)
votingClassifierSoft.fit(X,y)

votingClassifier2Hard.fit(XX,y)
votingClassifier2Soft.fit(XX,y)

print "votingClassifierHard:"
print "predict:",votingClassifierHard.predict(X_test)
print"\nvotingClassifierSoft:"
print "predict:",votingClassifierSoft.predict(X_test)
print "predict_proba:",votingClassifierSoft.predict_proba(X_test)
print "\nvotingClassifier2Hard:"
print "predict:",votingClassifier2Hard.predict(XX_test)
print"\nvotingClassifier2Soft:"
print "predict:",votingClassifier2Soft.predict(XX_test)
print "predict_proba:",votingClassifier2Soft.predict_proba(XX_test)
