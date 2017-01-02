import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Features.extract_features import *
import csv
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import make_scorer
from Features.extract_features import *
from Features.scoring import *

# Get the targets
with open('data/targets.csv', 'rb') as f:
	reader = csv.reader(f)
	targets = list(reader)

targets = np.array(targets).astype(np.int)
targets_sex, targets_age, targets_health = zip(*targets) 

# =================================================================	#
#						TRAIN FEATURES								#
# =================================================================	#
# Features for sex:		grayzone+hippoVar
# Features for age:		blackzone+grayzone+hippoHisto
# Features for health:	hippoMedian+blackzone+flipzones+hippoHisto

sexFeatures = []	
ageFeatures = []	
healthFeatures = []	

#histo = extractHistograms('data/set_train', 4000, 45, 9)
flipzones = extractFlipSim('data/set_train')
blackzone = extractBlackzones('data/set_train',nPartitions=3)
grayzone = extractColoredZone3D('data/set_train', 450, 800, 8)
grayWhiteRatio = extractGrayWhiteRatio('data/set_train', 8)
largeHippocMedian = extractLargeHippocampusMedians3D('data/set_train')
largeHippocMean = extractLargeHippocampusMeans3D('data/set_train')
largeHippocVar = extractLargeHippocampusVars3D('data/set_train')
largeHippocHisto = extractLargeHippocampusHistograms3D('data/set_train')
smallHippocMedian = extractSmallHippocampusMedians3D('data/set_train')
smallHippocMean = extractSmallHippocampusMeans3D('data/set_train')
smallHippocVar = extractSmallHippocampusVars3D('data/set_train')
smallHippocHisto = extractSmallHippocampusHistograms3D('data/set_train')
amygdalaMedian = extractAmygdalaMedians3D('data/set_train')
amygdalaMean = extractAmygdalaMeans3D('data/set_train')
amygdalaVar = extractAmygdalaVars3D('data/set_train')
amygdalaHisto = extractAmygdalaHistograms3D('data/set_train')
print "amygdalaMedians shape:",np.asarray(amygdalaMedian).shape
print "amygdalaMeans shape:",np.asarray(amygdalaMean).shape
print "amygdalaVars shape:",np.asarray(amygdalaVar).shape
print "amygdalaHisto shape:",np.asarray(amygdalaHisto).shape
    


sexFeatures.append(grayzone)
sexFeatures.append(np.array(amygdalaMedian))
sexFeatures.append(np.array(amygdalaMean))
sexFeatures.append(np.array(amygdalaVar))
#sexFeatures.append(np.array(amygdalaHisto))
#sexFeatures.append(hippocVar)


ageFeatures.append(np.array(smallHippocHisto))

healthFeatures.append(flipzones)
healthFeatures.append(blackzone)
healthFeatures.append(largeHippocMedian)
healthFeatures.append(largeHippocHisto)
healthFeatures.append(np.array(targets_age).reshape(-1,1)) #Age group as feature for health


# =================================================================	#
#					CONCATENATE FEATURES							#
# =================================================================	#

first = True
Csex = None
for f in sexFeatures:
	if first:
		Csex = f
		first = False
	else:
		Csex = np.concatenate((Csex,f), axis = 1)
		
print "Gender feature shapes:",Csex.shape
		
first = True
Cage = None
for f in ageFeatures:
	if first:
		Cage = f
		first = False
	else:
		Cage = np.concatenate((Cage,f), axis = 1)

print "Age feature shapes:",Cage.shape

first = True
Chealth = None
for f in healthFeatures:
	if first:
		Chealth = f
		first = False
	else:
		Chealth = np.concatenate((Chealth,f), axis = 1)

print "Health feature shapes:",Chealth.shape



h = .02  # step size in the mesh

names = ["Logistic Regression", "Linear SVM", "SGD Classifier", "Gaussian Process",
         "Poly SVM", "Sigmoid SVM", "Random Forest", "Neural Net", "AdaBoost",
"Naive Bayes", "QDA"]



classifiers = [
    LogisticRegression(tol=0.0001, C=1.0, class_weight={0: 10}),
    SVC(kernel="linear", C=0.025, probability=True, class_weight={0: 10}),
    SGDClassifier(loss="log", alpha=0.0001, shuffle=True, class_weight={0: 10}),
    GaussianProcessClassifier(0.323 * RBF(0.4), warm_start=True),
    SVC(kernel="poly", C=1.0, probability=True), 
    SVC(kernel="sigmoid", C=1.0, probability=True), 
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(base_estimator=SVC(kernel="linear", C=0.025, probability=True)),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    #LinearDiscriminantAnalysis()
]




pcaSex = PCA(n_components=2)
pcaSexData = pcaSex.fit_transform(Csex)
sexDataset = (pcaSexData, targets_sex)


pcaAge = PCA(n_components=2)
pcaAgeData = pcaAge.fit_transform(Cage)
ageDataset = (pcaSexData, targets_age)


pcaHealth = PCA(n_components=2)
pcaHealthData = pcaHealth.fit_transform(Chealth)
healthDataset = (pcaHealthData, targets_health)


datasets = [sexDataset, ageDataset, healthDataset]


figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    '''X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
'''
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    '''# Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)'''
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # and testing points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        scorer = make_scorer(partialHammingLoss,greater_is_better=False)
        score = cross_val_score(clf, X, y, cv=10, scoring=scorer)#clf.score(X_test, y_test)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
        # and testing points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % -score.mean()).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()