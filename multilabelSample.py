print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label, col):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label, color=col)


def plot_subfigure(X, Y, subplot, title, transform, classif):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    #classif = OneVsRestClassifier(SVC(kernel='linear'))
    print Y.shape
    print X.shape
    classif.fit(X, Y)
    plt.subplot(2, 8, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    two_class = np.where(Y[:, 2])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray')
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='cyan',
               facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
               facecolors='none', linewidths=2, label='Class 2')
    plt.scatter(X[two_class, 0], X[two_class, 1], s=340, edgecolors='magenta',
               facecolors='none', linewidths=2, label='Class 3')


    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1', 'cyan')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k+',
                    'Boundary\nfor class 2', 'orange')
    plot_hyperplane(classif.estimators_[2], min_x, max_x, 'k*',
                    'Boundary\nfor class 3', 'magenta')



    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")

def makePlot(X, Y, classif):

    plt.figure(figsize=(8, 6))

    plot_subfigure(X, Y, 1, "Without unlabeled samples + CCA", "cca", classif)
    plot_subfigure(X, Y, 2, "Without unlabeled samples + PCA", "pca", classif)
    
    plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
    plt.show()



X, Y = make_multilabel_classification(n_classes=3, n_labels=3,
                                      allow_unlabeled=True,
                                      random_state=1)
makePlot(X,Y,  OneVsRestClassifier(SVC(kernel='linear')))