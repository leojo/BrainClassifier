from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class MultiLabelVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Votingclaressor for scikit-learn estimators.

    Parameters
    ----------

    classifs : `iterable`
      A list of multilabel classifiers
    weights : `list` (default: `None`)
      If `None`, an unweighted average will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the weighted averaged of
        the predicted class labels using the given weights will be used.

    """
    def __init__(self, classifs, voting="hard", weights=None, separateFeatures = False):
        self.classifs = classifs
        self.voting = voting
        self.weights = weights
        self.multiFeature = separateFeatures

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_voters, n_samples, n_features] or [n_samples,n_features]
            Training data
        y : list or numpy array, shape = [n_samples,n_labels]
            Class labels

        """
        if self.multiFeature:
            for i,cla in enumerate(self.classifs):
                XX = np.asarray(np.asarray(X[i]).tolist())
                cla.fit(XX,y)
        else:
            for cla in self.classifs:
                cla.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_voters, n_samples, n_features] or [n_samples,n_features]

        Returns
        ----------

        y : list or numpy array, shape = [n_samples,n_labels]
            Weighted averages of the predictions of the estimators or predicted class probabilities

        """

        if self.multiFeature:
            predictions = np.asarray([cla.predict(XX) for cla,XX in zip(self.classifs,X)])
        else:
            predictions = np.asarray([cla.predict(X) for cla in self.classifs])

        if self.weights:
            y = sum([p*w for p,w in zip(predictions,self.weights)])/sum(self.weights)
        else:
            y = (sum(predictions)*1.)/len(predictions)

        if self.voting == "hard":
            y = np.round(y).astype(np.int)

        return y