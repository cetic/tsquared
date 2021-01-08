from hotelling_t2 import HotellingT2

from scipy import stats

from sklearn.utils.validation import check_is_fitted

class MytDecomposition():
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
	TODO
	CMD + F: TODO (there is a TODO in the conditional_t2_terms method and the
	documentation of this method)


	"""

	def __init__(self, hotelling_t2):
		"""
		Construct a MytDecomposition object.

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
		Compute conditional T-squared terms.

		TODO: change name of this method to correspond to say something like:
		"the p conditional T-squared terms which condition each feature on the
		remaining p-1 features"

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
			Conditional T-squared terms.

		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is
			`self.hotelling_t2.n_features_`.
		"""

		check_is_fitted(self.hotelling_t2)

		X = self.hotelling_t2._check_test_inputs(X)

		# TODO: add variable checking (see the compute_p_conditional_t2_terms
		# function in the Jupyter notebook).
		
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
