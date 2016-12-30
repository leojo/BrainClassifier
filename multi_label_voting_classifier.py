"""
Soft Voting/Majority Rule classifier.

This module contains a Soft Voting/Majority Rule classifier for
classification estimators.

"""

# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#		  Gilles Louppe <g.louppe@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted


def _parallel_fit_estimator(estimator, X, y, sample_weight):
	"""Private function used to fit an estimator within a job."""
	if sample_weight is not None:
		estimator.fit(X, y, sample_weight)
	else:
		estimator.fit(X, y)
	return estimator


class MultiLabelVotingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
	"""Soft Voting/Majority Rule classifier for unfitted estimators.

	.. versionadded:: 0.17

	Read more in the :ref:`User Guide <voting_classifier>`.

	Parameters
	----------
	estimators : list of (string, estimator) tuples
		Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
		of those original estimators that will be stored in the class attribute
		`self.estimators_`.

	voting : str, {'hard', 'soft'} (default='hard')
		If 'hard', uses predicted class labels for majority rule voting.
		Else if 'soft', predicts the class label based on the argmax of
		the sums of the predicted probabilities, which is recommended for
		an ensemble of well-calibrated classifiers.

	weights : array-like, shape = [n_classifiers], optional (default=`None`)
		Sequence of weights (`float` or `int`) to weight the occurrences of
		predicted class labels (`hard` voting) or class probabilities
		before averaging (`soft` voting). Uses uniform weights if `None`.

	n_jobs : int, optional (default=1)
		The number of jobs to run in parallel for ``fit``.
		If -1, then the number of jobs is set to the number of cores.

	Attributes
	----------
	estimators_ : list of classifiers
		The collection of fitted sub-estimators.

	classes_ : array-like, shape = [n_predictions]
		The classes labels.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.linear_model import LogisticRegression
	>>> from sklearn.naive_bayes import GaussianNB
	>>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
	>>> clf1 = LogisticRegression(random_state=1)
	>>> clf2 = RandomForestClassifier(random_state=1)
	>>> clf3 = GaussianNB()
	>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	>>> y = np.array([1, 1, 1, 2, 2, 2])
	>>> eclf1 = VotingClassifier(estimators=[
	...		 ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
	>>> eclf1 = eclf1.fit(X, y)
	>>> print(eclf1.predict(X))
	[1 1 1 2 2 2]
	>>> eclf2 = VotingClassifier(estimators=[
	...		 ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
	...		 voting='soft')
	>>> eclf2 = eclf2.fit(X, y)
	>>> print(eclf2.predict(X))
	[1 1 1 2 2 2]
	>>> eclf3 = VotingClassifier(estimators=[
	...		('lr', clf1), ('rf', clf2), ('gnb', clf3)],
	...		voting='soft', weights=[2,1,1])
	>>> eclf3 = eclf3.fit(X, y)
	>>> print(eclf3.predict(X))
	[1 1 1 2 2 2]
	>>>
	"""

	def __init__(self, estimators, voting='hard', weights=None, n_jobs=1):
		self.estimators = estimators
		self.named_estimators = dict(estimators)
		self.voting = voting
		self.weights = weights
		self.n_jobs = n_jobs

	def fit(self, X, y, sample_weight=None):
		""" Fit the estimators.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features] or 
			list of {array-like, sparse matrix} with shape = [n_samples, n_features], of length len(self.estimators).
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.

		y : array-like, shape = [n_samples] or [n_samples,n_labels]
			Target values.

		sample_weight : array-like, shape = [n_samples] or None
			Sample weights. If None, then samples are equally weighted.
			Note that this is supported only if all underlying estimators
			support sample weights.

		Returns
		-------
		self : object
		"""
		if len(X) == 0:
			raise ValueError("X must contain at least one entry; got (X=%r)" % X)
		if self.voting not in ('soft', 'hard'):
			raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
							 % self.voting)

		if self.estimators is None or len(self.estimators) == 0:
			raise AttributeError('Invalid `estimators` attribute, `estimators`'
								 ' should be a list of (string, estimator)'
								 ' tuples')

		if self.weights and len(self.weights) != len(self.estimators):
			raise ValueError('Number of classifiers and weights must be equal'
							 '; got %d weights, %d estimators'
							 % (len(self.weights), len(self.estimators)))

		if sample_weight is not None:
			for name, step in self.estimators:
				if not has_fit_parameter(step, 'sample_weight'):
					raise ValueError('Underlying estimator \'%s\' does not support'
									 ' sample weights.' % name)
		
		if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
			self.multilabel_ = True
			self.le_ = MultiLabelBinarizer()
			self.le_.fit([range(y.shape[1])])
		else:
			self.multilabel_ = False
			self.le_ = LabelEncoder()
			self.le_.fit(y)
			
		self.classes_ = self.le_.classes_
		self.estimators_ = []

		transformed_y = self.le_.transform(y)
		# Check whether we have X.shape = [n_samples,n_features] or X = [[n_samples,n_features_1],...,[n_samples,n_features_k]]
		measure = X[0]
		if not isinstance(measure, np.ndarray):
			measure = np.array(measure)
		
		self.multiple_features_ = len(measure.shape) == 2
		
		if self.multiple_features_ and len(X) != len(self.estimators):
			raise ValueError("For voters requiring different data, X must be a list of"
							"data arrays, with the same length as the number of voters. Got X of length %s" % len(X))
		
		if self.multiple_features_ and not isinstance(X,list):
			raise ValueError("For voters requiring different data, X must be a list of"
							"data arrays, with the same length as the number of voters. Got %s " % type(X))
		
		if self.multiple_features_:
			self.estimators_ = Parallel(n_jobs=self.n_jobs)(
					delayed(_parallel_fit_estimator)(clone(clf), XX, transformed_y,
						sample_weight)
						for XX, _, clf in zip(X,*zip(*self.estimators)))
		
		else:
			self.estimators_ = Parallel(n_jobs=self.n_jobs)(
					delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
						sample_weight)
						for _, clf in self.estimators)

		return self

	def predict(self, X):
		""" Predict class labels for X.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		----------
		maj : array-like, shape = [n_samples]
			Predicted class labels.
		"""

		check_is_fitted(self, 'estimators_')
		
		if self.voting == 'soft':
			if self.multilabel_:
				maj = np.round(self.predict_proba(X)).astype(np.int)
			else:
				maj = np.argmax(self.predict_proba(X), axis=1)

		else:  # 'hard' voting
			if self.multilabel_:
				predictions = self._predict(X).T
				maj = np.apply_along_axis(lambda x:
										  np.argmax(np.bincount(x,
													weights=self.weights)),
										  axis=0,
										  arr=predictions.astype('int'))
			else:
				predictions = self._predict(X)
				maj = np.apply_along_axis(lambda x:
										  np.argmax(np.bincount(x,
													weights=self.weights)),
										  axis=1,
										  arr=predictions.astype('int'))

				maj = self.le_.inverse_transform(maj)

		return maj

	def _collect_probas(self, X):
		"""Collect results from clf.predict calls. """
		if self.multiple_features_:
			if not isinstance(X,list):
				raise ValueError("For voters requiring different data, X must be a list of"
								"data arrays, with the same length as the number of voters. Got %s " % type(X))
			return np.asarray([clf.predict_proba(np.asarray(XX)) for XX,clf in zip(X,self.estimators_)])
		return np.asarray([clf.predict_proba(np.asarray(X)) for clf in self.estimators_])

	def _predict_proba(self, X):
		"""Predict class probabilities for X in 'soft' voting """
		if self.voting == 'hard':
			raise AttributeError("predict_proba is not available when"
								 " voting=%r" % self.voting)
		check_is_fitted(self, 'estimators_')
		
		avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
		return avg

	@property
	def predict_proba(self):
		"""Compute probabilities of possible outcomes for samples in X.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		----------
		avg : array-like, shape = [n_samples, n_classes]
			Weighted average probability for each class per sample.
		"""
		return self._predict_proba

	def transform(self, X):
		"""Return class labels or probabilities for X for each estimator.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		If `voting='soft'`:
		  array-like = [n_classifiers, n_samples, n_classes]
			Class probabilities calculated by each classifier.
		If `voting='hard'`:
		  array-like = [n_samples, n_classifiers]
			Class labels predicted by each classifier.
		"""
		check_is_fitted(self, 'estimators_')
		if self.voting == 'soft':
			return self._collect_probas(X)
		else:
			return self._predict(X)

	def get_params(self, deep=True):
		"""Return estimator parameter names for GridSearch support"""
		if not deep:
			return super(MultiLabelVotingClassifier, self).get_params(deep=False)
		else:
			out = super(MultiLabelVotingClassifier, self).get_params(deep=False)
			out.update(self.named_estimators.copy())
			for name, step in six.iteritems(self.named_estimators):
				for key, value in six.iteritems(step.get_params(deep=True)):
					out['%s__%s' % (name, key)] = value
			return out

	def _predict(self, X):
		"""Collect results from clf.predict calls. """
		if self.multiple_features_:
			if not isinstance(X,list):
				raise ValueError("For voters requiring different data, X must be a list of"
								"data arrays, with the same length as the number of voters. Got %s " % type(X))
			return np.asarray([clf.predict(np.asarray(XX)) for XX,clf in zip(X,self.estimators_)]).T
		return np.asarray([clf.predict(np.asarray(X)) for clf in self.estimators_]).T