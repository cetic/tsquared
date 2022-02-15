import numpy as np
from scipy import stats
from sklearn.utils.validation import check_is_fitted

from tsquared.hotelling_t2 import HotellingT2

class MYTDecomposition:
	"""MYT Decomposition of the Hotelling's T-squared statistic.

	The purpose of the MYT Decomposition of the Hotelling's T-squared statistic
	is to identify the cause of an out-of-control signal, i.e. an outlier, which
	the Hotelling's T-squared test is not able to do. More specifically, such a
	decomposition makes it possible to obtain information on which features
	significantly contribute to the out-of-control signal.

	This implementation does not include the p! entire decompositions, where p
	is the number of features of the data set. It includes only the p
	unconditional T-squared terms

	$T^2_1, T^2_2, \dotsc, T^2_p$

	and the p conditional T-squared terms which condition each feature on the
	remaining p-1 features

	$T^2_{1 \cdot 2, \dotsc, p}, T^2_{2 \cdot 1, 3, \dotsc, p}, \dotsc, T^2_{p \cdot 1, \dotsc, p-1}$.

	For one sample s,

	- a signal on an unconditional term, meaning that the value of this term is
	  greater than the upper control limit (UCL) defined for unconditional
	  terms, implies that the involved feature is outside the operational range
	  specified by the training set. For example, suppose that $T^2_4$ is
	  greater than the UCL. This means that the value of the forth feature in
	  the sample s is outside its allowable range of variation defined by the
	  training set;
	- a signal on an conditional term, meaning that the value of this term is
	  greater than the upper control limit (UCL) defined for conditional terms,
	  implies that something is wrong with the relationship among the features
	  included in the conditional term. For example, a signal on
	  $T^2_{3 \cdot 1, 2, \dotsc, p}$ implies that the relation between the
	  third feature and the remaining ones is counter to the relationship
	  observed in the training set. In other words, the value on the third
	  feature in the sample s is not where it should be relative to the value of
	  the other features.

	Parameters
	----------
	hotelling_t2 : hotelling_t2.HotellingT2
		A hotelling_t2.HotellingT2 object.

	References
	----------
	Robert L. Mason, Nola D. Tracy, John C. Young (1995). Decomposition of T2
	for Multivariate Control Chart Interpretation.
	Journal of Quality Technology.

	Robert L. Mason, John C. Young (2001). Multivariate Statistical Process
	Control with Industrial Applications.
	Society for Industrial and Applied Mathematics.
	ISBN: 9780898714968

	Examples
	--------
	>>> import numpy as np
	>>> from tsquared import HotellingT2
	>>> from tsquared import MYTDecomposition
	>>> X = np.array([[10. , 10.7],
    ...               [10.4,  9.8],
    ...               [ 9.7, 10. ],
    ...               [ 9.7, 10.1],
    ...               [11.7, 11.5],
    ...               [11. , 10.8],
    ...               [ 8.7,  8.8],
    ...               [ 9.5,  9.3],
    ...               [10.1,  9.4],
    ...               [ 9.6,  9.6],
    ...               [10.5, 10.4],
    ...               [ 9.2,  9. ],
    ...               [11.3, 11.6],
    ...               [10.1,  9.8],
    ...               [ 8.5,  9.2]])
    >>> X_test = np.array([[12.3, 12.5],
    ...                    [ 7. ,  7.3],
    ...                    [11. ,  9. ],
    ...                    [ 7.3,  9.1]])
	>>> clf = HotellingT2().fit(X)
	>>> clf.mean_
	array([10., 10.])
	>>> clf.cov_
	array([[0.79857143, 0.67928571],
           [0.67928571, 0.73428571]])
    >>> clf.score_samples(X_test)
    array([ 8.51262745, 11.41034614, 23.14059036, 21.59620748])
    >>> myt_dec = MYTDecomposition(clf)
    >>> myt_dec.unconditional_t2_terms(X_test)
    array([[ 6.62432916,  8.51167315],
           [11.27012522,  9.92801556],
           [ 1.25223614,  1.3618677 ],
           [ 9.12880143,  1.10311284]])
    >>> myt_dec.ucl_unconditional_terms()
    4.906783932447382
    >>> myt_dec.conditional_t2_terms(X_test)
    array([[9.54296667e-04, 1.88829829e+00],
           [1.48233057e+00, 1.40220913e-01],
           [2.17787227e+01, 2.18883542e+01],
           [2.04930946e+01, 1.24674060e+01]])
    >>> myt_dec.ucl_conditional_terms()
    5.361288061175456
	"""

	def __init__(self, hotelling_t2):
		"""
		Construct a MYTDecomposition object.

		Parameters
		----------
		hotelling_t2 : hotelling_t2.HotellingT2
			A hotelling_t2.HotellingT2 object.
		"""

		if not isinstance(hotelling_t2, HotellingT2):
			raise TypeError("The argument `hotelling_t2` must be a"
				" hotelling_t2.HotellingT2 object.")

		self.hotelling_t2 = hotelling_t2

	def unconditional_t2_terms(self, X):
		"""
		Compute unconditional T-squared terms.

		For each sample s in `X`, for each feature j, compute the unconditional
		T-squared term $T_j^2$, equivalent to square the univariate
		t-statistic.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		unconditional_t2_terms : array-like, shape (n_samples, n_features)
			Unconditional T-squared terms.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is
			`self.hotelling_t2.n_features_`.
		"""

		check_is_fitted(self.hotelling_t2)

		X = self.hotelling_t2._check_test_inputs(X)

		return (X - self.hotelling_t2.mean_) ** 2 / \
			np.diag(self.hotelling_t2.cov_)

	def ucl_unconditional_terms(self):
		"""
		Compute the upper control limit (UCL) of unconditional T-squared terms.

		The significance level used is `self.hotelling_t2.alpha`.

		Returns
		-------
		ucl_unconditional_t2_terms : float
			Returns the upper control limit (UCL) of the unconditional T-squared
			terms.
		"""

		check_is_fitted(self.hotelling_t2)

		n_samples = self.hotelling_t2.n_samples_
		critical_val = stats.f.ppf(q=1-self.hotelling_t2.alpha, dfn=1,
			dfd=n_samples-1)
		
		return (n_samples + 1) / n_samples * critical_val

	def conditional_t2_terms(self, X):
		"""
		Compute the p conditional T-squared terms which condition each feature
		on the remaining p-1 features, where p is the number of features.

		For each sample s in `X`, compute the following conditional T-squared
		terms:

		$T^2_{1 \cdot 2, \dotsc, p}, T^2_{2 \cdot 1, 3, \dotsc, p}, \dotsc, T^2_{p \cdot 1, \dotsc, p-1}$,

		where p is the number of features.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		conditional_t2_terms : array-like, shape (n_samples, n_features)
			Conditional T-squared terms which condition each feature on the
			remaining `self.n_features_`-1 features.

		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is
			`self.hotelling_t2.n_features_`.
		"""

		check_is_fitted(self.hotelling_t2)

		X = self.hotelling_t2._check_test_inputs(X)
		
		n_samples, n_features = X.shape
		
		X_centered = X - self.hotelling_t2.mean_ # Zero-centered data.
		
		s_squared = np.empty(n_features)
		x_bar = np.empty((n_features, n_samples))
		for j in range(n_features):
			sxx = np.delete(self.hotelling_t2.cov_[j], j)
			b_j = np.linalg.inv(
				np.delete(np.delete(self.hotelling_t2.cov_, j, axis=1), j,
					axis=0)
			) @ sxx

			s_squared[j] = self.hotelling_t2.cov_[j, j] - sxx @ b_j
			x_bar[j] = self.hotelling_t2.mean_[j] + \
				np.delete(X_centered, j, axis=1) @ b_j
		
		return (X - x_bar.T) ** 2 / s_squared

	def ucl_conditional_terms(self):
		"""
		Compute the upper control limit (UCL) of the p conditional T-squared
		terms which condition each feature on the remaining p-1 features, where
		p is the number of features.

		The significance level used is `self.hotelling_t2.alpha`.

		Returns
		-------
		ucl_unconditional_t2_terms : float
			Returns the upper control limit (UCL) of the conditional T-squared
			terms which condition each feature on the remaining
			`self.n_features_`-1 features.
		"""

		check_is_fitted(self.hotelling_t2)

		n_samples = self.hotelling_t2.n_samples_
		n_cond_vars = self.hotelling_t2.n_features_ - 1 # Number of conditioned
		# variables.
		critical_val = stats.f.ppf(q=1-self.hotelling_t2.alpha, dfn=1,
			dfd=n_samples-n_cond_vars-1)

		return (((n_samples + 1) * (n_samples - 1)) / \
			(n_samples * (n_samples - n_cond_vars - 1))) * critical_val
