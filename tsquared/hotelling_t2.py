import numpy as np
import pingouin as pg
from scipy import stats
from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

class HotellingT2(BaseEstimator, OutlierMixin, TransformerMixin):
	"""Hotelling's T-squared test.

	Hotelling's T-squared test is an unsupervised multivariate outlier
	detection.

	When fitting on a (clean) training set, the real distribution, supposed to
	be a multivariate normal distribution, is estimated. In order to achieve
	this, these parameters are estimated:

	- the empirical mean for each feature;
	- the sample covariance matrix.

	In addition, two upper control limits (UCLs) are computed. One of these is
	chosen to classify new samples. See the "Attributes" section for an
	explanation of the difference between these two limits: ucl_indep_ and
	ucl_not_indep_. Note that the first one is the UCL used by default, but this
	behavior can be changed by calling the set_default_ucl method.

	When predicting, for each sample x from a test set, a T-squared score is
	computed and compared to the default upper control limit. If this score
	exceeds this limit, then x will be classified as an outlier. Otherwise, x
	will be classified as an inlier.

	Parameters
	----------
	alpha : float, between 0 and 1, optional (default=0.05)
		The significance level for computing the upper control limit.

	Attributes
	----------
	mean_ : ndarray, shape (n_features,)
		Per-feature empirical mean, estimated from the training set.

		Equal to `X.mean(axis=0)`.

	cov_ : ndarray, shape (n_features, n_features)
		Sample covariance matrix estimated from the training set.

		Equal to `np.cov(X.T, ddof=1)`.

	ucl_indep_ : float
		Upper control limit (UCL) when assuming:

		- the parameters of the underlying multivariate normal distribution are
		  unknown and are estimated using a training set;
		- samples in test set are independent of the estimated parameters. In
		  other words, these samples are not used to estimate the parameters.

		For a single sample x, if the T-squared score is greater than the UCL,
		then x will be reported as an outlier. Otherwise, x will be reported as
		an inlier.

	ucl_not_indep_ : float
		Upper control limit (UCL) when assuming:

		- the parameters of the underlying multivariate normal distribution are
		  unknown and are estimated using a training set;
		- samples in test set are not independent of the estimated parameters.
		  In other words, these samples are used to estimate the parameters.

		For a single sample x, if the T-squared score is greater than the UCL,
		then x will be reported as an outlier. Otherwise, x will be reported as
		an inlier.

	n_features_ : int
		Number of features in the training data.

	n_samples_ : int
		Number of samples in the training data.
	
	X_fit_ : {array-like, sparse matrix}, shape (n_samples, n_features)
		A reference to the training set of samples. It is used to infer which
		UCL should be used.

	Other variables
	---------------
	default_ucl : {'auto', 'indep', 'not indep'} (default='indep')
		The upper control limit (UCL) to be used. It affects the methods relying
		on the UCL (such as predict and transform methods).

		default_ucl can take one of the following values:
		- 'indep': the default UCL used will be `self.ucl_indep_`;
		- 'not indep': the default UCL used will be `self.ucl_not_indep_`;
		- 'auto': depending on the test set, the default UCL used will be either
		  `self.ucl_indep_` or `self.ucl_not_indep_`.
		  To determine which UCL should be used, we verify whether the test set
		  is a subset of the training set. If so, `self.ucl_not_indep_` will be
		  used as the default UCL, otherwise `self.ucl_indep_` will be used.

		Note that if 'auto' is selected, the call to methods relying on the UCL
		may be slowed down significantly. For this reason, 'auto' is not the
		default value of default_ucl.

	References
	----------
	Camil Fuchs, Ron S. Kenett (1998). Multivariate Quality Control: Theory and
	Applications. Quality and Reliability.
	Taylor & Francis.
	ISBN: 9780367579326

	Robert L. Mason, John C. Young (2001). Multivariate Statistical Process
	Control with Industrial Applications.
	Society for Industrial and Applied Mathematics.
	ISBN: 9780898714968

	Examples
	--------
	>>> import numpy as np
	>>> from tsquared import HotellingT2
	>>> true_mean = np.array([0, 0])
	>>> true_cov = np.array([[.8, .3],
	...                      [.3, .4]])
	>>> X = np.random.RandomState(0).multivariate_normal(mean=true_mean,
	...                                                  cov=true_cov,
	...                                                  size=500)
	>>> X_test = np.array([[0, 0],
	...                    [3, 3]])
	>>> clf = HotellingT2().fit(X)
	>>> clf.predict(X_test)
	array([ 1, -1])
	>>> clf.score_samples(X_test)
	array([5.16615725e-03, 2.37167895e+01])
	>>> clf.ucl(X_test)
	6.051834565565274
	"""

	def __init__(self, alpha=0.05):
		self.alpha = alpha

		self.default_ucl = 'indep'

	def fit(self, X, y=None):
		"""
		Fit Hotelling's T-squared. Specifically, compute the mean vector, the
		covariance matrix on X and the upper control limits.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Training set of samples, where n_samples is the number of samples
			and n_features is the number of features. It should be clean and
			free of outliers.

		y : None
			Not used, present for scikit-learn's API consistency by convention.

		Returns
		-------
		self : object
			Returns the instance itself.

		Raises
		------
		ValueError
			If the number of samples of `X`, n_samples, is less than or equal
			to the number of features of `X`, n_features.
		"""

		X = self._check_train_inputs(X)

		self.n_samples_, self.n_features_ = X.shape

		self.mean_ = X.mean(axis=0)
		self.cov_ = np.cov(X.T, ddof=1)
		if self.n_features_ == 1:
			self.cov_ = self.cov_.reshape(1, 1)

		self.ucl_indep_ = self._ucl_indep(self.n_samples_, self.n_features_,
			alpha=self.alpha)
		self.ucl_not_indep_ = self._ucl_not_indep(self.n_samples_,
			self.n_features_, alpha=self.alpha)

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
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		score_samples : array-like, shape (n_samples,)
			Returns the T-squared score of each sample.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_`.
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
		Scaled T-squared score of each sample x. It is between 0 and 1 denoting
		how outlier x is (i.e. the level of abnormality); 0 meaning that x is
		most likely an inlier and 1 meaning that x is most likely an outlier.
		Scaled T-squared scores are bounded T-squared scores, which, for
		example, makes plotting of scores more readable.

		The `ucl_baseline` argument is the baseline value for the upper control
		limit (UCL), used to scale T-squared scores. For example, if
		`ucl_baseline` is set to 0.1, any scaled T-squared score less than 0.1
		will be classified as an inlier and, similarly, any scaled T-squared
		score greater than 0.1 will be classified as an outlier.

		Each scaled T-squared score scaled_s is computed from the respective
		T-squared score s (see the score_samples method) as follows:
		```
		scaled_s = s / self.ucl(X) * ucl_baseline
		if scaled_s > 1:
			scaled_s = 1
		```

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		ucl_baseline : float (default=0.05)
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
			features of the training set, that is `self.n_features_`.

		ValueError
			If the UCL baseline `ucl_baseline` is not strictly between 0 and 1.
		"""

		if not (0 < ucl_baseline < 1):
			raise ValueError("The UCL baseline `ucl_baseline` must be strictly"
				" between 0 and 1.")

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
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		score_sample : float
			Returns the T-squared score of `X`.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_`.
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
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		y_pred : array-like, shape (n_samples,)
			Returns -1 for outliers and 1 for inliers.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_`.
		"""

		t2_scores = self.score_samples(X)

		return np.where(t2_scores > self.ucl(X), -1, 1)

	def transform(self, X):
		"""
		Filter inliers.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		X_filtered : array-like, shape (n_samples_filtered, n_features)
			Returns inliers.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_`.
		"""

		check_is_fitted(self)

		X = self._check_test_inputs(X)

		t2_scores = self.score_samples(X)

		return X[t2_scores <= self.ucl(X)]
		
	def cleanfit(self, X, res=1, iter=-1):
		"""
		Recursively remove outliers until conditions are encountered (including
		Henze-Zirkler test), and fit.
		->Merge the several methods in one code
				- minimum number of outliers detected for stopping iterations
				- number of iterations decided by the user
				- max number of iterations based on the data size
				- smart cleaning based on normality coefficient
					- door open for other coefs
		
		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Training set of samples, where n_samples is the number of samples
			and n_features is the number of features.

		n : stop criteria  - minimum number of outliers (default=5)  

		TODO: add res, iter parameters.

		Returns
		-------
		self : object
			Returns the instance itself.

		X_filtered : array-like, shape (n_samples_filtered, n_features)
			Returns inliers.

		n_iterations : number of iterations of cleaning

		TODO: add Xclean2, _iter, hz values.

		Raises
		------
		ValueError
			If the number of samples of `X`, n_samples, is less than or equal
			to the number of features of `X`, n_features.
		"""
		
		# Initialization.
		X = self._check_train_inputs(X)
		self.n_samples_, self.n_features_ = X.shape
		self.ucl_indep_ = self._ucl_indep(self.n_samples_, self.n_features_,
			alpha=self.alpha)
		self.ucl_not_indep_ = self._ucl_not_indep(self.n_samples_,
			self.n_features_, alpha=self.alpha)
		self.X_fit_ = X		
		self.cov_ = np.cov(X.T, ddof=1)
		
		# Cleanfit specific initialization.
			
		_res = self.n_samples_ / 2 # Variable - Initialize to the maximum		
		# allowed points to be removed.
		TOTP = self.n_samples_ # Constant - Initial number of points.
		Xclean2 = X # Initialize second cleaned X for bootstrapping the
		# iteration.
		_iter = 0
		hzprev = 100 # Empirically fixed based on observations on PyOD dataset -
		# hypothesis of normality rejected if too large (generally >300).
		_continue = 1
		if(iter < 0):
			hz, pval, flag = pg.multivariate_normality(Xclean2)
			if(hz < hzprev):
				_continue = 1
				hzprev = hz
			else:
				_continue = 0
		else:
			hz = 1
			det = 1
		self.set_default_ucl('not indep')
		
		# Recursivity.
		while (_res > res) and (_iter != iter) and (Xclean2.shape[0] > TOTP/2) \
			and _continue == 1:

			Xclean = Xclean2
			
			self.fit(Xclean)
			Xclean2 = self.transform(Xclean)
			if(iter > -1): # If iter is given, it discards criteria on HZ
			# coefficient.
				_continue = 1
			else:
				hz, pval, flag = pg.multivariate_normality(Xclean2)
				
				if(hz < hzprev):
					_continue = 1
					hzprev = hz
				else:
					_continue = 0
					
			_res = Xclean.shape[0] - Xclean2.shape[0]
			_iter += 1

		self.set_default_ucl('indep')
		self.fit(Xclean2)

		t2_scores = self.score_samples(X)

		return self, Xclean2, _iter, hz
		
	def set_default_ucl(self, ucl):
		"""
		Set the default upper control limit (UCL) to either 'auto', 'indep' or
		'not indep'.

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
			If the default upper control limit `ucl` is not either 'auto',
			'indep' or 'not indep'.
		"""

		if ucl not in {'auto', 'indep', 'not indep'}:
			raise ValueError("The default upper control limit `ucl` must be"
				" either 'auto', 'indep' or 'not indep'.")

		self.default_ucl = ucl

		return self

	def ucl(self, X_test):
		"""
		Return the value of the upper control limit (UCL) depending on
		`self.default_ucl` and `X_test`.

		Parameters
		----------
		X_test : {array-like, sparse matrix}, shape (n_samples, n_features)
			Test set of samples, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		-------
		ucl : float
			Returns the value of the upper control limit (UCL) depending on
			`self.default_ucl` and `X_test`.

		Raises
		------
		ValueError
			If the default upper control limit `self.default_ucl` is not either
			'auto', 'indep' or 'not indep'.

		ValueError
			If the number of features of `X_test` is not equal to the number of
			features of the training set, that is `self.n_features_`.
		"""

		check_is_fitted(self)

		if self.default_ucl == 'indep':
			return self.ucl_indep_

		if self.default_ucl == 'not indep':
			return self.ucl_not_indep_

		if self.default_ucl != 'auto':
			raise ValueError("The default upper control limit"
				"`self.default_ucl` must be either 'auto', 'indep' or 'not"
				" indep'.")

		X_test = self._check_test_inputs(X_test)

		# Test if `X_test` is not a subset of `self.X_fit_` (may be slow).
		if X_test.shape[0] > self.X_fit_.shape[0] or \
			not np.isin(X_test, self.X_fit_).all():

			return self.ucl_indep_

		return self.ucl_not_indep_

	def _ucl_indep(self, n_samples, n_features, alpha=0.05):
		"""
		Compute the upper control limit (UCL) when assuming:

		- the parameters of the underlying multivariate normal distribution are
		  unknown and are estimated using a training set;
		- samples in test set are independent of the estimated parameters. In
		  other words, these samples are not used to estimate the parameters.

		Parameters
		----------
		n_samples : int
			The number of samples of the training set.

		n_features : int
			The number of features of the training set.

		alpha : float, between 0 and 1, optional (default=0.05)
			The significance level.

		Returns
		-------
		ucl : float
			Returns the upper control limit (UCL) when samples in test set are
			independent of the estimated parameters.

		Raises
		------
		ValueError
			If the significance level `alpha` is not between 0 and 1.
		"""

		if not 0 <= alpha <= 1:
			raise ValueError("The significance level `alpha` must be between 0"
				" and 1.")

		critical_val = stats.f.ppf(q=1-alpha, dfn=n_features,
			dfd=n_samples-n_features)
	
		return n_features * (n_samples + 1) * (n_samples - 1) / n_samples / \
			(n_samples - n_features) * critical_val

	def _ucl_not_indep(self, n_samples, n_features, alpha=0.05):
		"""
		Compute the upper control limit (UCL) when assuming:

		- the parameters of the underlying multivariate normal distribution are
		  unknown and are estimated using a training set;
		- samples in test set are not independent of the estimated parameters.
		  In other words, these samples are used to estimate the parameters.

		Parameters
		----------
		n_samples : int
			The number of samples of the training set.

		n_features : int
			The number of features of the training set.

		alpha : float, between 0 and 1, optional (default=0.05)
			The significance level.

		Returns
		-------
		ucl : float
			Returns the upper control limit (UCL) when samples in test set are
			not independent of the estimated parameters.

		Raises
		------
		ValueError
			If the significance level `alpha` is not between 0 and 1.
		"""

		if not 0 <= alpha <= 1:
			raise ValueError("The significance level `alpha` must be between 0"
				" and 1.")

		critical_val = stats.beta.ppf(q=1-alpha, a=n_features/2,
			b=(n_samples-n_features-1)/2)
	
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
			Set of samples to check / convert, where n_samples is the number of
			samples and n_features is the number of features.

		Returns
		-------
		X_converted : array-like, shape (n_samples, n_features)
			Returns the converted and validated inputs.
		"""

		X = check_array(X,
			accept_sparse=True,
			dtype=[np.float64, np.float32],
			force_all_finite=False,
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
			Training set of samples to check / convert, where n_samples is the
			number of samples and n_features is the number of features.

		Returns
		-------
		X_converted : array-like, shape (n_samples, n_features)
			Returns the converted and validated inputs.

		Raises
		------
		ValueError
			If the number of samples of `X`, n_samples, is less than or equal
			to the number of features of `X`, n_features.
		"""

		X = self._check_inputs(X)

		n_samples, n_features = X.shape

		if n_samples <= n_features:
			raise ValueError("The number of samples of `X` must be strictly"
				" greater than the number of features of `X`.")

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
			Test set of samples to check / convert, where n_samples is the
			number of samples and n_features is the number of features.

		Returns
		-------
		X_converted : array-like, shape (n_samples, n_features)
			Returns the converted and validated inputs.

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_`.
		"""

		X = self._check_inputs(X)

		n_features = X.shape[1]
		if self.n_features_ != n_features:
			raise ValueError("The number of features of `X` must be equal to"
				" the number of features of the training set.")

		return X
			
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt

	np.random.seed(42)

	n = 1000
	m = 100
	p = 4

	true_mean = np.array([4, -1.3, 8.7, -5.4])
	true_cov = np.array([
		[1, 0.4, -0.4, 0.1],
		[0.4, 1, 0.6, -0.2],
		[-0.4, 0.6, 1, 0.02],
		[0.1, -0.2, 0.02, 1]
	])

	train = np.random.multivariate_normal(true_mean, true_cov, size=n)
	test = np.random.multivariate_normal(true_mean, true_cov, size=m)

	print("--- Inputs ---\n")

	print(f"True mean vector: {true_mean}")
	print(f"True covariance matrix:\n{true_cov}")

	print("\n--- Hotelling's T-squared fitting on the training set---\n")

	hotelling = HotellingT2()
	hotelling.fit(train)

	print(f"Computed mean vector: {hotelling.mean_}")
	print(f"Computed covariance matrix:\n{hotelling.cov_}")
	print(f"Hotelling's T-squared UCL: {hotelling.ucl(test)}")

	print("\n--- Hotelling's T-squared scores on the test set ---\n")

	t2_scores = hotelling.score_samples(test)
	scaled_t2_scores = hotelling.scaled_score_samples(test)

	print(f"Hotelling's T-squared score for each sample:\n{t2_scores}")
	print(f"Scaled Hotelling's T-squared score for each sample:"
		f"\n{scaled_t2_scores}")

	print("\n--- Outlier detection ---\n")

	pred = hotelling.predict(test)
	outliers = test[pred == -1]

	print(f"Detected outliers:\n{outliers}")

	print("\n--- Hotelling's T-squared score on the test set ---\n")

	t2_score = hotelling.score(test)
	ucl = n / (n + 1) * hotelling.ucl_indep_

	print(f"Hotelling's T-squared score for the test set: {t2_score}")
	print(f"Do the training set and the test set come from the same"
		f" distribution? {t2_score <= ucl}")

	fig, ax = plt.subplots(figsize=(14, 8))
	plt.scatter(range(scaled_t2_scores.size), scaled_t2_scores)
	ucl_line = plt.axhline(y=0.1, color='r', linestyle='-')
	ax.set_title('Scaled Hotelling\'s T2 scores')
	ax.set_xlabel('Index')
	ax.set_ylabel('Scaled Hotelling\'s T2 score')
	ucl_line.set_label('UCL')
	plt.legend()
	fig.tight_layout()
	plt.show()
