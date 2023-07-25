import numpy as np
import pingouin as pg
from sklearn.utils import check_array

from tsquared import HotellingT2

def clean_samples(X, *, n_iters=0, perc_samples=.5, n_samples_per_iter=1,
	hz_test=True, alpha=0.05):
	"""
	Clean samples using Hotelling's T-squared test. Outliers are removed
	iteratively as long as following criteria are met:

	- the maximum number of iterations `n_iters` is not reached;
	- the number of filtered samples is greater than or equal to a percentage
	  of the initial samples `perc_samples`;
	- the number of samples removed per iteration is greater than or equal to a
	  threshold `n_samples_per_iter`;
	- the null hypothesis of the Henze-Zirkler test can be rejected. In other
	  words, we can reject the fact that the filtered samples of the current
	  iteration are drawn from the a multivariate normal distribution.

	Parameters
	----------
	X : {array-like, sparse matrix}, shape (n_samples, n_features)
		Set of samples, where `n_samples` is the number of samples and
		`n_features` is the number of features. It should be clean and free of
		outliers.

	n_iters : int, greater than or equal to 0, default=0
		Maximum number of iterations. If equal to 0, no limit is set.

	perc_samples : float, between 0 and 1, default=0.5
		Minimum percentage of samples to be kept.

	n_samples_per_iter : int, greater than or equal to 1, default=1
		Minimum number of samples removed per iteration. If this criterion is
		not met, the cleaning process is aborted. The parameter is greater than
		or equal to 1 to avoid an infinite loop.

	hz_test : bool, default=True
		If True, Henze-Zirkler test is used as a stopping criterion.

	alpha : float, between 0 and 1, default=0.05
		The significance level for the Hotelling's T-squared and Henze-Zirkler
		tests.

	Returns
	-------
	X_filtered : array-like, shape (n_samples_filtered, n_features)
		Returns samples filtered of outliers.

	Raises
	------
	ValueError
		If the number of samples of `X`, `n_samples`, is less than or equal
		to the number of features of `X`, `n_features`.

	ValueError
		If the maximum number of iterations `n_iters` is less than 0.

	ValueError
		If the minimum percentage of samples to be kept `perc_samples` is not
		between 0 and 1.

	ValueError
		If the minimum number of samples removed per iteration
		`n_samples_per_iter` is less than 1.

	ValueError
		If the significance level `alpha` is not between 0 and 1.
	"""

	X = check_array(X,
		accept_sparse=True,
		dtype=[np.float64, np.float32],
		force_all_finite=False,
		ensure_2d=True,
		estimator='clean'
	)

	n_samples, n_features = X.shape

	if n_samples <= n_features:
		raise ValueError("The number of samples of X must be strictly greater "
			"than the number of features of X.")

	if n_iters < 0:
		raise ValueError("The maximum number of iterations must be greater "
			"than or equal to 0.")

	if not 0 <= perc_samples <= 1:
		raise ValueError("The minimum percentage of samples to be kept must be "
			"between 0 and 1.")

	if n_samples_per_iter < 1:
		raise ValueError("The minimum number of samples removed per iteration "
			"must be greater than or equal to 1.")

	if not 0 <= alpha <= 1:
		raise ValueError("The significance level alpha must be between 0 and "
			"1.")

	clf = HotellingT2(alpha=alpha)
	clf.set_default_ucl('not indep')
	min_samples_to_keep = np.ceil(n_samples * perc_samples)
	X_clean = X
	X_clean_next = X
	for _ in range(n_iters) if n_iters > 0 else iter(int, 1):
		# Infinite loop if `n_iters` <= 0.

		try:
			X_clean_next = clf.fit_transform(X_clean)
		except ValueError:
			# "it's easier to ask for forgiveness than permission".
			break

		if hz_test:
			try:
				_, pvalue, _ = pg.multivariate_normality(X_clean_next)
			except AssertionError:
				break

			if pvalue < alpha:
				# We can reject null hypothesis, which states that the samples
				# are drawn from the a multivariate normal distribution.
				pass
			else:
				# We fail to reject null hypothesis. Therefore, we do not have
				# sufficient evidence to conclude that the samples are drawn
				# from different distributions.
				break

		n_samples_current = X_clean.shape[0]
		n_samples_next = X_clean_next.shape[0]
		n_samples_removed = n_samples_current - n_samples_next
		if n_samples_removed < n_samples_per_iter:
			break

		if n_samples_next < min_samples_to_keep:
			break

		X_clean = X_clean_next

	return X_clean
