import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from tsquared.threshold import ThresholdCalculator


class HotellingT2(BaseEstimator, OutlierMixin, TransformerMixin):
	"""Hotelling's T-squared test.

    Hotelling's T-squared test is an unsupervised multivariate outlier
    detection method.

    When fitting on a (clean) training set, the real distribution, supposed to
    be a multivariate normal distribution, is estimated. In order to achieve
    this, these parameters are estimated:

    - the empirical mean for each feature;
    - the sample covariance matrix.

    Two methods are available for computing the threshold/upper control limit (UCL):
    - 'statistical': Uses the classical F-distribution based approach (default)
    - 'optimization': Uses convex optimization to find maximum T-squared value
                     under box constraints

    Parameters
    ----------
    alpha : float, between 0 and 1, default=0.05
        The significance level for computing the upper control limit.
    threshold_method : {'statistical', 'optimization'}, default='statistical'
        Method to use for computing the T-squared threshold.
        - 'statistical': Uses F-distribution based UCL (default)
        - 'optimization': Uses optimization-based threshold

    Attributes
    ----------
    mean_ : ndarray, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    cov_ : ndarray, shape (n_features, n_features)
        Sample covariance matrix estimated from the training set.

        Equal to `np.cov(X.T, ddof=1)`.

    ucl_indep_ : float
        Upper control limit (UCL) computed based on the selected threshold_method.
        For 'statistical' method, this represents the UCL when assuming samples
        in test set are independent of the estimated parameters.

    ucl_not_indep_ : float
        Upper control limit (UCL) when assuming samples in test set are not
        independent of the estimated parameters. Only used with 'statistical' method.

    n_features_in_ : int
        Number of features in the training data.

    n_samples_in_ : int
        Number of samples in the training data.

    X_fit_ : {array-like, sparse matrix}, shape (n_samples, n_features)
        A reference to the training set of samples.

    default_ucl : {'auto', 'indep', 'not indep'}, default='indep'
        The upper control limit (UCL) to be used. Only relevant for
        'statistical' threshold method.
    """

	def __init__(self, alpha=0.05, threshold_method='statistical'):
		if threshold_method not in ['statistical', 'optimization']:
			raise ValueError("threshold_method must be either 'statistical' or 'optimization'")

		if not 0 < alpha < 1:
			raise ValueError("The significance level alpha must be between 0 and 1")

		self.alpha = alpha
		self.threshold_method = threshold_method
		self.default_ucl = 'indep'


	def fit(self, X, y=None):
		"""
        Fit Hotelling's T-squared. Specifically, compute the mean vector, the
        covariance matrix on X and the upper control limits.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training set of samples, where `n_samples` is the number of samples
            and `n_features` is the number of features. It should be clean and
            free of outliers.

        y : None
            Not used, present for scikit-learn's API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
		X = self._check_train_inputs(X)
		self.n_samples_in_, self.n_features_in_ = X.shape

		self.mean_ = X.mean(axis=0)
		self.cov_ = np.cov(X.T, ddof=1)
		if self.n_features_in_ == 1:
			self.cov_ = self.cov_.reshape(1, 1)

		if self.threshold_method == 'statistical':
			self.ucl_indep_ = ThresholdCalculator.statistical_ucl(
				self.n_samples_in_,
				self.n_features_in_,
				self.alpha
			)
			self.ucl_not_indep_ = self._ucl_not_indep(
				self.n_samples_in_,
				self.n_features_in_,
				self.alpha
			)
		else:  # 'optimization'
			self.ucl_indep_ = ThresholdCalculator.optimization_based(
				self.mean_,
				self.cov_
			)
			self.ucl_not_indep_ = self.ucl_indep_  # Same threshold for both in optimization method

		self.X_fit_ = X
		return self


	def score_samples(self, X):
		"""
		T-squared score of each sample. The higher the score, the further the
		sample is from the training set distribution. Each score is to be
		compared to the upper control limit (UCL).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where `n_samples` is the number of samples and
			`n_features` is the number of features.

		Returns
		-------
		score_samples : array-like, shape (n_samples,)
			Returns the T-squared score of each sample.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_in_`.
		"""

		check_is_fitted(self)

		X = self._check_test_inputs(X)

		X_centered = X - self.mean_ # Zero-centered data.
		inverse_cov = np.linalg.pinv(self.cov_) # Inverse covariance matrix.
		# Previously np.linalg.inv was used. However, it failed on singular
		# matrix. Explanation on this URL:
		# https://stackoverflow.com/questions/49357417/why-is-numpy-linalg-pinv-preferred-over-numpy-linalg-inv-for-creating-invers/49364727

		t2_scores = np.einsum('ij,ij->i', X_centered @ inverse_cov, X_centered)
		# Equivalent to:
		# ```
		# t2_scores = []
		# for Xi in X:
		#     t2 = (Xi - self.mean_).T @ inverse_cov @ (Xi - self.mean_)
		#     t2_scores.append(t2)
		# t2_scores = np.array(t2_scores)
		# ```
		# Or:
		# ```
		# t2_scores = np.diag(X_centered @ inverse_cov @ X_centered.T)
		# ```
		# Or:
		# ```
		# t2_scores = ((X_centered @ inverse_cov) * X_centered).sum(axis=-1)
		# ```
		# Reference:
		# https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the

		return t2_scores


	def scaled_score_samples(self, X, ucl_baseline=0.1):
		"""
		Scaled T-squared score of each sample `x`. It is between 0 and 1
		denoting how outlier `x` is (i.e. the level of abnormality); 0 meaning
		that `x` is most likely an inlier and 1 meaning that `x` is most likely
		an outlier. Scaled T-squared scores are bounded T-squared scores, which,
		for example, makes plotting of scores more readable.

		The `ucl_baseline` argument is the baseline value for the upper control
		limit (UCL), used to scale T-squared scores. For example, if
		`ucl_baseline` is set to 0.1, any scaled T-squared score less than 0.1
		will be classified as an inlier and, similarly, any scaled T-squared
		score greater than 0.1 will be classified as an outlier.

		Each scaled T-squared score `scaled_s` is computed from the respective
		T-squared score `s` (see the `score_samples` method) as follows:

		```
		scaled_s = s / self.ucl(X) * ucl_baseline
		if scaled_s > 1:
			scaled_s = 1
		```

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where `n_samples` is the number of samples and
			`n_features` is the number of features.

		ucl_baseline : float, default=0.05
			Baseline value, strictly between 0 and 1, for the upper control
			limit (UCL).

		Returns
		-------
		score_samples : array-like, shape (n_samples,)
			Returns the scaled T-squared score of each sample.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_in_`.

		ValueError
			If the UCL baseline `ucl_baseline` is not strictly between 0 and 1.
		"""

		if not (0 < ucl_baseline < 1):
			raise ValueError("The UCL baseline must be strictly between 0 and "
				"1.")

		t2_scores = self.score_samples(X)

		scaled_t2_scores = t2_scores / self.ucl(X) * ucl_baseline
		scaled_t2_scores[scaled_t2_scores > 1] = 1

		return scaled_t2_scores


	def score(self, X):
		"""
		T-squared score of an entire set of samples. The higher the score, the
		further `X` is from the training set distribution. If this score is
		greater than the upper control limit (UCL), then it is likely that `X`
		does not come from the same distribution as the training set.

		Note that the UCL that should be used in this case is not
		`self.ucl_indep_` nor `self.ucl_not_indep_`, but rather:

		`self.n_samples` / (`self.n_samples` + 1) * `self.ucl_indep_`.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where `n_samples` is the number of samples and
			`n_features` is the number of features.

		Returns
		-------
		score_sample : float
			Returns the T-squared score of `X`.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_in_`.
		"""

		check_is_fitted(self)

		X = self._check_test_inputs(X)

		test_mean = X.mean(axis=0)

		t2_score = (test_mean - self.mean_).T @ np.linalg.inv(self.cov_) @ \
			(test_mean - self.mean_)

		return t2_score


	def predict(self, X):
		"""
		Perform classification on samples in `X`.

		Returns -1 for outliers and 1 for inliers.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where `n_samples` is the number of samples and
			`n_features` is the number of features.

		Returns
		-------
		y_pred : array-like, shape (n_samples,)
			Returns -1 for outliers and 1 for inliers.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_in_`.
		"""

		t2_scores = self.score_samples(X)

		return np.where(t2_scores > self.ucl(X), -1, 1)


	def transform(self, X):
		"""
		Filter inliers.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where `n_samples` is the number of samples and
			`n_features` is the number of features.

		Returns
		-------
		X_filtered : array-like, shape (n_samples_filtered, n_features)
			Returns inliers.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_in_`.
		"""

		check_is_fitted(self)

		X = self._check_test_inputs(X)

		t2_scores = self.score_samples(X)

		return X[t2_scores <= self.ucl(X)]


	def set_default_ucl(self, ucl):
		"""
		Set the default upper control limit (UCL) to either `'auto'`, `'indep'`
		or `'not indep'`.

		Parameters
		----------
		ucl : {'auto', 'indep', 'not indep'}
			Set the default upper control limit (UCL).

		Returns
		-------
		self : object
			Returns the instance itself.

		Raises
		------
		ValueError
			If the default upper control limit `ucl` is not either `'auto'`,
			`'indep'` or `'not indep'`.
		"""

		if ucl not in {'auto', 'indep', 'not indep'}:
			raise ValueError("The default upper control limit must be either "
				"'auto', 'indep' or 'not indep'.")

		self.default_ucl = ucl

		return self


	def ucl(self, X_test):
		"""
        Return the value of the upper control limit (UCL) depending on
        self.default_ucl and X_test.

        For 'optimization' threshold method, this always returns ucl_indep_
        regardless of default_ucl setting.

        Parameters
        ----------
        X_test : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test set of samples, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        ucl : float
            Returns the value of the upper control limit (UCL).
        """
		check_is_fitted(self)

		if self.threshold_method == 'optimization':
			return self.ucl_indep_

		if self.default_ucl == 'indep':
			return self.ucl_indep_

		if self.default_ucl == 'not indep':
			return self.ucl_not_indep_

		if self.default_ucl != 'auto':
			raise ValueError("The default upper control limit must be either "
							 "'auto', 'indep' or 'not indep'.")

		X_test = self._check_test_inputs(X_test)

		# Test if `X_test` is not a subset of `self.X_fit_` (may be slow).
		if X_test.shape[0] > self.X_fit_.shape[0] or \
				not np.isin(X_test, self.X_fit_).all():
			return self.ucl_indep_

		return self.ucl_not_indep_


	def _ucl_not_indep(self, n_samples, n_features, alpha=0.05):
		"""
        Compute the upper control limit (UCL) when assuming samples in test set
        are not independent of the estimated parameters.

        Only used with 'statistical' threshold method.
        """
		if not 0 <= alpha <= 1:
			raise ValueError("The significance level alpha must be between 0 "
							 "and 1.")

		critical_val = stats.beta.ppf(q=1 - alpha, a=n_features / 2,
									  b=(n_samples - n_features - 1) / 2)

		return (n_samples - 1) ** 2 / n_samples * critical_val


	def _check_inputs(self, X):
		"""
		Input validation on a sample before fit, predict and transform.

		The input is checked to be a non-empty 2D array containing only finite
		values. If the dtype of the array is object, attempt converting to
		float, raising on failure.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Set of samples to check / convert, where `n_samples` is the number
			of samples and `n_features` is the number of features.

		Returns
		-------
		X_converted : array-like, shape (n_samples, n_features)
			Returns the converted and validated inputs.
		"""

		X = check_array(X,
			accept_sparse=True,
			dtype=[np.float64, np.float32],
			ensure_all_finite=False,
			ensure_2d=True,
			estimator=self
		)

		return X


	def _check_train_inputs(self, X):
		"""
		Input validation on a train sample before fit.

		The input is checked to be a non-empty 2D array containing only finite
		values. If the dtype of the array is object, attempt converting to
		float, raising on failure.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Training set of samples to check / convert, where `n_samples` is the
			number of samples and `n_features` is the number of features.

		Returns
		-------
		X_converted : array-like, shape (n_samples, n_features)
			Returns the converted and validated inputs.

		Raises
		------
		ValueError
			If the number of samples of `X`, `n_samples`, is less than or equal
			to the number of features of `X`, `n_features`.
		"""

		X = self._check_inputs(X)

		n_samples, n_features = X.shape

		if n_samples <= n_features:
			raise ValueError("The number of samples of X must be strictly "
				"greater than the number of features of X.")

		return X


	def _check_test_inputs(self, X):
		"""
		Input validation on a test sample before predict and transform.

		The input is checked to be a non-empty 2D array containing only finite
		values. If the dtype of the array is object, attempt converting to
		float, raising on failure.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples to check / convert, where `n_samples` is the
			number of samples and `n_features` is the number of features.

		Returns
		-------
		X_converted : array-like, shape (n_samples, n_features)
			Returns the converted and validated inputs.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_in_`.
		"""

		X = self._check_inputs(X)

		n_features = X.shape[1]
		if self.n_features_in_ != n_features:
			raise ValueError("The number of features of X must be equal to "
				"the number of features of the training set.")

		return X
