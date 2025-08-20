import numpy as np
import math
import itertools
from scipy import stats
from sklearn.utils.validation import check_is_fitted

from tsquared import HotellingT2

class MYTDecomposition:
	"""
	MYT Decomposition of the Hotelling's T-squared statistic.

	The purpose of the MYT Decomposition of the Hotelling's T-squared statistic
	is to identify the cause of an out-of-control signal, i.e. an outlier, which
	the Hotelling's T-squared test is not able to do. More specifically, such a
	decomposition makes it possible to obtain information on which features
	significantly contribute to the out-of-control signal.

	This implementation provides the full MYT decomposition, meaning that
	it is possible to compute the p! complete decompositions, one for each
	permutation (order) of the p features.  

	For a given ordering π = (π₁, …, πₚ), the decomposition has p terms:

		T²_{π₁},
		T²_{π₂ | π₁},
		T²_{π₃ | π₁,π₂},
		…,
		T²_{πₚ | π₁,…,π_{p-1}}.

	Each term measures the contribution of the variable πⱼ, either
	unconditionally (first variable in the order) or conditionally on the
	previous variables in that order.

	Interpretation
	--------------
	For one sample s:

	- A signal on an unconditional term (e.g. T²₄ > UCL) indicates that the
	corresponding feature is outside its normal range of variation, as defined
	by the training set. It corresponds to the square of the univariate
	t-statistic for that feature.

	- A signal on a conditional term (e.g. T²_{3 | 1,2,…,p}) indicates that
	the relationship between the variable in question and the other variables
	is abnormal compared to the relationships observed in the training data.
	In other words, the feature value is inconsistent with the expected
	multivariate structure.

	Notes
	-----
	- The full decomposition involves p! permutations, which grows rapidly with
	p. For large p, it is recommended to restrict to a subset of permutations.
	- This implementation provides methods to compute unconditional terms,
	conditional terms for each variable given the others, and the complete set
	of MYT terms for one or several orders.
	- Upper Control Limits (UCLs) are computed using the F distribution, with
	separate formulas for unconditional and conditional terms.

	Parameters
	----------
	hotelling_t2 : tsquared.HotellingT2
		A fitted HotellingT2 object, providing the mean, covariance,
		and significance level alpha.

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
		hotelling_t2 : tsquared.HotellingT2
			A tsquared.HotellingT2 object.
		"""

		if not isinstance(hotelling_t2, HotellingT2):
			raise TypeError("The argument `hotelling_t2` must be a"
				" tsquared.HotellingT2 object.")

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
			`self.hotelling_t2.n_features_in_`.
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

		n_samples = self.hotelling_t2.n_samples_in_
		critical_val = stats.f.ppf(q=1-self.hotelling_t2.alpha, dfn=1,
			dfd=n_samples-1)

		return (n_samples + 1) / n_samples * critical_val

	def conditional_t2_terms(self, X):
		"""
		Compute the p conditional T-squared terms which condition each feature
		on the remaining p-1 features, where p is the number of features.

		For each sample s in `X`, compute the following conditional T-squared
		terms:

			T²_{1 | 2, ..., p}, T²_{2 | 1, 3, ..., p}, ..., T²_{p | 1, ..., p-1}

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

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is
			`self.hotelling_t2.n_features_in_`.
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

		n_samples = self.hotelling_t2.n_samples_in_
		n_cond_vars = self.hotelling_t2.n_features_in_ - 1 # Number of
		# conditioned variables.
		critical_val = stats.f.ppf(q=1-self.hotelling_t2.alpha, dfn=1,
			dfd=n_samples-n_cond_vars-1)

		return (((n_samples + 1) * (n_samples - 1)) / \
			(n_samples * (n_samples - n_cond_vars - 1))) * critical_val

	def conditional_contribution_subset(self, X, i, S):
		"""
		Compute the vector (n_samples,) of contributions T^2_{i | S}.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		i : int – index of the variable of interest

		S : sequence of indices (can be empty) – variables to condition on

		Returns
		-------
		terms : array-like, shape (n_samples,)
			Conditional T-squared terms T^2_{i | S} for each sample in `X`.
		"""

		check_is_fitted(self.hotelling_t2)
		X = self.hotelling_t2._check_test_inputs(X)
		mean = self.hotelling_t2.mean_
		cov = self.hotelling_t2.cov_

		p = X.shape[1]
		if i < 0 or i >= p:
			raise IndexError("i out of bounds")
		S = tuple(int(k) for k in S)
		if i in S:
			raise ValueError("Index i must not be in S.")

		if len(S) == 0:
			# Terme inconditionnel (z-score^2)
			var = cov[i, i]
			if var <= 0:
				raise np.linalg.LinAlgError("Non positive variance encountered.")
			resid = X[:, i] - mean[i]
			return (resid ** 2) / var

		# Blocs
		mu_i = mean[i]
		mu_S = mean[list(S)]
		Sigma_SS = cov[np.ix_(S, S)]
		Sigma_iS = cov[np.ix_([i], S)]   # (1, |S|)
		Sigma_Si = cov[np.ix_(S, [i])]   # (|S|, 1)

		# Inversion (avec petite régularisation si mal conditionné)
		try:
			Sigma_SS_inv = np.linalg.inv(Sigma_SS)
		except np.linalg.LinAlgError:
			eps = 1e-10
			Sigma_SS_inv = np.linalg.inv(Sigma_SS + eps * np.eye(len(S)))

		beta = Sigma_iS @ Sigma_SS_inv                 # (1, |S|)
		var_i_given_S = cov[i, i] - (Sigma_iS @ Sigma_SS_inv @ Sigma_Si)[0, 0]
		if var_i_given_S <= 0:
			# numérique : clip à 0+ pour éviter divisions négatives
			var_i_given_S = max(var_i_given_S, 0.0)
			if var_i_given_S == 0.0:
				resid = X[:, i] - (mu_i + (X[:, list(S)] - mu_S) @ beta.T).ravel()
				return np.where(np.isclose(resid, 0.0, atol=1e-12), 0.0, np.inf)

		mu_i_given_S = mu_i + (X[:, list(S)] - mu_S) @ beta.T   # (n, 1)
		resid = X[:, i] - mu_i_given_S.ravel()
		return (resid ** 2) / var_i_given_S

	def myt_terms_for_order(self, X, order=None):
		"""
		Computes the full MYT decomposition for a given order of variables.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		order : iterable of int, optional
			An order of the features, as a sequence of integers from 0 to p-1,
			where p is the number of features. If None, the order is assumed to be
			the identity order (0, 1, ..., p-1).

		Returns
		-------
		terms : array (n, p)
			Column t: T^2_{π_t | π_1..π_{t-1}}
		order : tuple
			The order used (permutation).
		"""

		check_is_fitted(self.hotelling_t2)
		X = self.hotelling_t2._check_test_inputs(X)
		p = X.shape[1]

		if order is None:
			order = tuple(range(p))
		else:
			order = tuple(int(k) for k in order)
			if len(order) != p or set(order) != set(range(p)):
				raise ValueError("`order` must be a permutation of 0..p-1 of length p.")

		terms = np.empty((X.shape[0], p))
		S_prefix = []
		for t, i in enumerate(order):
			terms[:, t] = self.conditional_contribution_subset(X, i=i, S=S_prefix)
			S_prefix.append(i)
		return terms, order

	def myt_all_orders(self, X, orders=None, max_permutations=None):
		"""
		Compute the MYT decomposition for all permutations (or a subset).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		orders : iterable of orders (each a sequence of length p). If None,
			all permutations are generated.

		max_permutations : int or None. If p! is large and None, a safeguard is
			applied: if p > 8, raises an error unless `max_permutations` (or
			`orders`) is provided. If given, truncates to the first
			`max_permutations` orders.

		Returns
		-------
		results : dict {order_tuple: array(n_samples, n_features)}
			Dictionary mapping each order (permutation) to its MYT decomposition.
		"""

		check_is_fitted(self.hotelling_t2)
		X = self.hotelling_t2._check_test_inputs(X)
		p = X.shape[1]

		if orders is None:
			total = math.factorial(p)
			if max_permutations is None and p > 8:
				raise ValueError(
					f"p={p} ⇒ p!={total} permutations. Fournissez `orders` explicites ou `max_permutations` pour échantillonner."
				)
			it = itertools.permutations(range(p))
			if max_permutations is not None:
				it = itertools.islice(it, int(max_permutations))
		else:
			it = (tuple(ordr) for ordr in orders)

		results = {}
		for order in it:
			terms, _ = self.myt_terms_for_order(X, order)
			results[order] = terms
		return results

	def ucl_conditional_terms_k(self, k):
		"""
		Compute the upper control limit (UCL) for a conditional T-squared term at 
		step k (conditioned on k-1 variables).

		At step k (1 ≤ k ≤ p), the denominator degrees of freedom (dfd) is n - k,
		and the scaling factor follows the same form as in `ucl_conditional_terms`,
		where `n_cond_vars = k-1`.

		For k = 1 and k = p, the UCL matches those returned by
		`ucl_unconditional_terms` and `ucl_conditional_terms`, respectively.

		Parameters
		----------
		k : int
			Step in the MYT decomposition (1-based index).

		Returns
		-------
		ucl : float
			Upper control limit for the conditional T-squared term at step k.
		"""
		
		check_is_fitted(self.hotelling_t2)
		n_samples = self.hotelling_t2.n_samples_in_
		if k < 1 or k > self.hotelling_t2.n_features_in_:
			raise ValueError("k must be in [1, p]")
		n_cond_vars = k - 1
		critical_val = stats.f.ppf(q=1 - self.hotelling_t2.alpha, dfn=1, dfd=n_samples - n_cond_vars - 1)
		return (((n_samples + 1) * (n_samples - 1)) / (n_samples * (n_samples - n_cond_vars - 1))) * critical_val
