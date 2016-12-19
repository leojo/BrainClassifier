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

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(base_estimator=SVC(kernel="linear", C=0.025, probability=True)),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


'''
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
'''
'''datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
<<<<<<< HEAD
            ]
'''
         
'''
#histograms
histoData = np.asarray(extractHistograms("../data/set_train", 1500, 45, 16))
=======
            ]'''

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)
targets = np.asarray([int(x[0]) for x in targets])           

'''
#histograms
histoData = np.asarray(extractHistograms("../data/set_train", 4500, 45, 16))
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6
pcaHisto = PCA(n_components=2)
pcaHistoData = pcaHisto.fit_transform(histoData)
myDatasetHisto = (pcaHistoData, targets)


# zones
zoneData = np.asarray(extractZoneAverages("../data/set_train", 8))
pcaZone = PCA(n_components=2)
pcaZoneData = pcaZone.fit_transform(zoneData)
myDatasetZone = (pcaZoneData, targets)

# zones 
<<<<<<< HEAD
=======
<<<<<<< HEAD

=======
'''
'''
blackData = np.asarray(extractBlackzones("../data/set_train", 8))
pcaBlack = PCA(n_components=2)
pcaBlackData = pcaBlack.fit_transform(blackData)
myDatasetBlack = (pcaBlackData, targets)

blackData2 = np.asarray(extractBlackzones("../data/set_train", 8))
blackData2 = blackData2*blackData2
pcaBlack2 = PCA(n_components=2)
pcaBlackData2 = pcaBlack2.fit_transform(blackData2)
myDatasetBlack2 = (pcaBlackData2, targets)

blackData3 = np.asarray(extractBlackzones("../data/set_train", 8))
print blackData3
blackData3 = np.log10((blackData3+np.asarray([[1]*len(blackData3[0])]*len(blackData3))))
print blackData3
pcaBlack3 = PCA(n_components=2)
pcaBlackData3 = pcaBlack3.fit_transform(blackData3)
myDatasetBlack3 = (pcaBlackData3, targets)

# zones

=======
myDatasetBlack3 = (pcaBlackData3, targets)'''
'''
# zones
<<<<<<< HEAD
fullData = np.asarray(extractFlipSim("../data/set_train"),scale = 0.1)
pcaFull = PCA(n_components=2)
pcaFullData = pcaFull.fit_transform(fullData)
myDatasetFull = (pcaFullData, targets)
''''''
fourFifthsData = np.asarray(extractFlipSim("../data/set_train", scale = 0.2))
pcaFourFifths = PCA(n_components=2)
pcaFourFifthsData = pcaFourFifths.fit_transform(fourFifthsData)
myDatasetFourFifths = (pcaFourFifthsData, targets)

threeFifthsData = np.asarray(extractFlipSim("../data/set_train", scale = 0.15))
pcaThreeFifths = PCA(n_components=2)
pcaThreeFifthsData = pcaThreeFifths.fit_transform(threeFifthsData)
myDatasetThreeFifths = (pcaThreeFifthsData, targets)

twoFifthsData = np.asarray(extractFlipSim("../data/set_train", scale = 0.1))
pcaTwoFifths = PCA(n_components=2)
pcaTwoFifthsData = pcaTwoFifths.fit_transform(twoFifthsData)
myDatasetTwoFifths = (pcaTwoFifthsData, targets)

oneFifthData = np.asarray(extractFlipSim("../data/set_train", scale = 0.05))
pcaOneFifth = PCA(n_components=2)
pcaOneFifthData = pcaOneFifth.fit_transform(oneFifthData)
myDatasetOneFifth = (pcaOneFifthData, targets)
print oneFifthData.shape
datasets = [myDatasetFull]#, myDatasetFourFifths, myDatasetThreeFifths, myDatasetTwoFifths, myDatasetOneFifth]
=======
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6
blackData = np.asarray(extractColoredZone("../data/set_train", 1, 450, 8))
pcaBlack = PCA(n_components=2)
pcaBlackData = pcaBlack.fit_transform(blackData)
myDatasetBlack = (pcaBlackData, targets)

grayData = np.asarray(extractColoredZone("../data/set_train", 450, 800, 8))
pcaGray = PCA(n_components=2)
pcaGrayData = pcaGray.fit_transform(grayData)
myDatasetGray = (pcaGrayData, targets)

<<<<<<< HEAD
grayData2 = np.asarray(extractColoredZone("../data/set_train", 450, 800, 8))
grayData2 = np.log10((grayData2+np.asarray([[1]*len(grayData2[0])]*len(grayData2))))
pcaGray2 = PCA(n_components=2)
pcaGrayData2 = pcaGray2.fit_transform(grayData2)
myDatasetGray2 = (pcaGrayData2, targets)

=======
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6
whiteData = np.asarray(extractColoredZone("../data/set_train", 800, 1500, 8))
pcaWhite = PCA(n_components=2)
pcaWhiteData = pcaWhite.fit_transform(whiteData)
myDatasetWhite = (pcaWhiteData, targets)

evenWhiterData = np.asarray(extractColoredZone("../data/set_train", 1500, 4000, 8))
pcaEvenWhiter = PCA(n_components=2)
pcaWhiterData = pcaEvenWhiter.fit_transform(evenWhiterData)
myDatasetWhiter = (pcaWhiterData, targets)

<<<<<<< HEAD


blackData = np.asarray(extractColoredZone("../data/set_train", 200, 750, 16))
#bestSelectorBlack = SelectKBest(k=20)
pcaBlack = PCA(n_components=2)
#pcaBlackData = bestSelectorBlack.fit_transform(blackData, targets)
pcaBlackData = pcaBlack.fit_transform(blackData)
myDatasetBlack = (pcaBlackData, targets)

grayData = np.asarray(extractColoredZone("../data/set_train", 400, 800, 8))
bestSelectorGray = SelectKBest(k=100)
pcaGray = PCA(n_components=2)
pcaGrayData = bestSelectorGray.fit_transform(grayData, targets)
pcaGrayData = pcaGray.fit_transform(pcaGrayData)
myDatasetGray = (pcaGrayData, targets)

whiteData = np.asarray(extractColoredZone("../data/set_train", 350, 900, 8))
bestSelectorWhite = SelectKBest(k=100)
pcaWhite = PCA(n_components=2)
pcaWhiteData = bestSelectorWhite.fit_transform(whiteData, targets)
pcaWhiteData = pcaWhite.fit_transform(pcaWhiteData)
myDatasetWhite = (pcaWhiteData, targets)

evenWhiterData = np.asarray(extractColoredZone("../data/set_train", 200, 750, 8))
bestSelectorWhiter = SelectKBest(k=100)
pcaEvenWhiter = PCA(n_components=2)
pcaWhiterData = bestSelectorWhiter.fit_transform(evenWhiterData, targets)
pcaWhiterData = pcaEvenWhiter.fit_transform(pcaWhiterData)
myDatasetWhiter = (pcaWhiterData, targets)


threeColorData = np.asarray(extractThreeColors("../data/set_train", 8, 450, 650, 1500))
pcaThreeColor = PCA(n_components=2)
pcaThreeColorData = pcaThreeColor.fit_transform(threeColorData)
myDatasetThreeColors = (pcaThreeColorData, targets)


blackgray = np.asarray(extractColorPercentage("../data/set_train", 350, 650))
pca1 = PCA(n_components=2)
pca1Data = pca1.fit_transform(blackgray)
myDatasetBlackGray = (pca1Data, targets)

blackgray2 = np.asarray(extractColorPercentage("../data/set_train", 450, 1100))
pca2 = PCA(n_components=2)
pca2Data = pca2.fit_transform(blackgray2)
myDatasetBlackGray2 = (pca2Data, targets)

blackgray3 = np.asarray(extractColorPercentage("../data/set_train", 650, 900))
pca3 = PCA(n_components=2)
pca3Data = pca3.fit_transform(blackgray3)
myDatasetBlackGray3 = (pca3Data, targets)

graywhite = np.asarray(extractColorPercentage("../data/set_train", 450, 1100, 1, 2))
pca2 = PCA(n_components=2)
pca2Data = pca2.fit_transform(graywhite)
myDatasetGrayWhite = (pca2Data, targets)

blackwhite = np.asarray(extractColorPercentage("../data/set_train", 450, 1100, 0, 2))
pca3 = PCA(n_components=2)
pca3Data = pca3.fit_transform(blackwhite)
myDatasetBlackWhite = (pca3Data, targets)
'''

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)
targets = np.asarray([int(x[0]) for x in targets])  

h = .02  # step size in the mesh

names = ["Logistic Regression", "Linear SVM", "SGD Classifier", "Gaussian Process",
         "Poly SVM", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

# What algorithms are we most interested in?
# 
# Some form of
#
#       SVC
#       Gaussian Process Classifier
#       Maybe AdaBoost?

# These are the algorithms that have been discussed in class and these are the algorithms

classifiers = [
    #LogisticRegression(tol=0.0001, C=1.0, class_weight={0: 10}),
    SVC(kernel="linear", C=0.025, probability=True, class_weight={0: 10}),
    #SGDClassifier(loss="log", alpha=0.0001, shuffle=True, class_weight={0: 10}),
    GaussianProcessClassifier(0.323 * RBF(0.4), warm_start=True),
    SVC(kernel="poly", C=1.0, probability=True), 
    SVC(kernel="sigmoid", C=1.0, probability=True), 
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1),
    #AdaBoostClassifier(base_estimator=SVC(kernel="linear", C=0.025, probability=True)),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
    #LinearDiscriminantAnalysis()]
    ]
'''
blackgray = np.asarray(extractColorPercentage("../data/set_train", 350, 650))
pca1 = PCA(n_components=2)
pca1Data = pca1.fit_transform(blackgray)
myDatasetBlackGray = (pca1Data, targets)

grayData = np.asarray(extractColoredZone("../data/set_train", 450, 800, 8))
pcaGray = PCA(n_components=2)
pcaGrayData = pcaGray.fit_transform(grayData)
myDatasetGray = (pcaGrayData, targets)

histoData = np.asarray(extractHistograms("../data/set_train", 1500, 45, 16))
pcaHisto = PCA(n_components=2)
pcaHistoData = pcaHisto.fit_transform(histoData)
myDatasetHisto = (pcaHistoData, targets)

zoneData = np.asarray(extractZoneAverages("../data/set_train", 8))
pcaZone = PCA(n_components=2)
pcaZoneData = pcaZone.fit_transform(zoneData)
myDatasetZone = (pcaZoneData, targets)
'''
grayData = np.asarray(extractColoredZone("../data/set_train", 450, 800, 8))
pcaGray = PCA(n_components=2)
pcaGrayData = pcaGray.fit_transform(grayData)
myDatasetGray = (pcaGrayData, targets)

grayData3D = np.asarray(extractColoredZone3D("../data/set_train", 450, 800, 8))
pcaGray3D = PCA(n_components=2)
pcaGrayData3D = pcaGray.fit_transform(grayData3D)
myDatasetGray3D = (pcaGrayData3D, targets)


histoData = np.asarray(extractHistograms("../data/set_train", 1500, 45, 16))
pcaHisto = PCA(n_components=2)
pcaHistoData = pcaHisto.fit_transform(histoData)
myDatasetHisto = (pcaHistoData, targets)
datasets = [myDatasetGray, myDatasetHisto, myDatasetGray3D]


'''
=======
combinedData = np.concatenate((blackData, grayData, whiteData, evenWhiterData),axis=1)
pcaCombined = PCA(n_components=2)
pcaCombinedData = pcaCombined.fit_transform(combinedData)
myDatasetCombined = (pcaCombinedData, targets)
print combinedData.shape
datasets = [myDatasetBlack, myDatasetGray, myDatasetWhite, myDatasetWhiter, myDatasetCombined]
>>>>>>> 2f29850d321756c0d3cb3f82f4bfef7b922fa2a6
>>>>>>> 975d874a910e58bf95b3c239225b5f4dc5223c25
'''
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
        score = cross_val_score(clf, X, y, cv=10, scoring='neg_log_loss', n_jobs=-1)#clf.score(X_test, y_test)
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
