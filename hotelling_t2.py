from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import numpy as np
from scipy import stats
from scipy.stats import lognorm
import math
from scipy import stats

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
	ucl_not_indep_. Note that the first one is the UCL used by default, but you
	can change this behavior by calling the set_default_ucl method.

	When predicting, for each sample x from a test set, a T-squared score is
	computed and compared to the default upper control limit. If this score
	exceeds this limit, then x will be classified as an outlier. Otherwise, x
	will be classified as an inlier.

	Parameters
	----------
	alpha: float, between 0 and 1, optional (default=0.05)
		The significance level for computing the upper control limit.

	Attributes
	----------
	mean_ : ndarray, shape (n_features,)
		Per-feature empirical mean, estimated from the training set.

		Equal to `X.mean(axis=0)`.

	cov_ : ndarray, shape (n_features, n_features)
		Sample covariance matrix estimated from the training set.

		Equal to `np.cov(X.T, ddof=1)`.

	ucl_indep_: float
		Upper control limit (UCL) when assuming:

		- the parameters of the underlying multivariate normal distribution are
		  unknown and are estimated using a training set;
		- samples in test set are independent of the estimated parameters. In
		  other words, these samples are not used to estimate the parameters.

		For a single sample x, if the T-squared score is greater than the UCL,
		then x will be reported as an outlier. Otherwise, x will be reported as
		an inlier.

	ucl_not_indep_: float
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
	
	X_fit_: {array-like, sparse matrix}, shape (n_samples, n_features)
		A reference to the training set of samples. It is used to infer which
		UCL should be used.

	Other variables
	---------------
	default_ucl: {'auto', 'indep', 'not indep'} (default='indep')
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
	>>> from hotelling_t2 import HotellingT2
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
		# previously inverse_cov = np.linalg.inv(self.cov_) but failed on singular matrix
		# explanation    https://stackoverflow.com/questions/49357417/why-is-numpy-linalg-pinv-preferred-over-numpy-linalg-inv-for-creating-invers/49364727
		
		
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

		return np.where(t2_scores > self.ucl(X),  -1, 1)

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
		
	def cleanfit(self, X,res=1,iter=-1):
		"""
		
		Recursively remove outliers until conditions are encountered (including Henze-Zirkler test), and fit
		->Merge the several methods in one code
				- minimum number of outliers detected for stopping iterations
				- number of iterations decided by the user
				- max number of iterations based on the data size
				- smart cleaning based on normality coefficient
					- door open for other coefs
		
		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Training set of samples, where n_samples is the number of samples and
			n_features is the number of features.
		n : stop criteria  - minimum number of outliers (default=5)  

		Returns
		-------
		self : object
			Returns the instance itself.
		X_filtered : array-like, shape (n_samples_filtered, n_features)
			Returns inliers.
		n_iterations: number of iterations of cleaning

		Raises
		------
		ValueError
			If the number of features of `X` is not equal to the number of
			features of the training set, that is `self.n_features_`.
		"""

		#####INIT###self.fit(X)#######
		X = self._check_train_inputs(X)
		self.n_samples_, self.n_features_ = X.shape
		self.ucl_indep_ = self._ucl_indep(self.n_samples_, self.n_features_,
			alpha=self.alpha)
		self.ucl_not_indep_ = self._ucl_not_indep(self.n_samples_,
			self.n_features_, alpha=self.alpha)
		self.X_fit_ = X		
		
		
		#cleanfit specific initialisation
			
		_res=self.n_samples_/2  #variable - init to the maximum allowed points to be removed
		TOTP=self.n_samples_    #CONST - Initial number of points
		Xclean2=X          #Init second cleaned X for boostraping the iteration
		_iter=0
		hzprev=100		   #Empiricaly fixed based on observations on Pyod dataset - hypothesis of normality rejected if too large (generally >300)
		_continue=1
		
		hz,pval,flag=self.HenzeZirkler(Xclean2)
		if(hz<hzprev):
			_continue=1
			hzprev=hz
		else:
			_continue=0
		#print("hz0",hz)
		
		self.set_default_ucl('not indep')
		
		#recursivity
		while ((_res>res) and (_iter!=iter) and (Xclean2.shape[0] > TOTP/2) and _continue==1):
			Xclean=Xclean2
			
			self.fit(Xclean)
			Xclean2=self.transform(Xclean)
			hz,pval,flag=self.HenzeZirkler(Xclean2)
			if(hz<hzprev):
				_continue=1
				hzprev=hz
			else:
				if(iter > -1):   # If iter is given, it discards criteria on HZ coef
					_continue=1
				else:
					_continue=0
			_res=Xclean.shape[0]-Xclean2.shape[0]
			#print("hz",_iter,hz)
			_iter+=1
   
		self.set_default_ucl('indep')
		self.fit(Xclean2)

		t2_scores = self.score_samples(X)

		return self,Xclean2,_iter,hz
	
	
		
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

		alpha: float, between 0 and 1, optional (default=0.05)
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

		alpha: float, between 0 and 1, optional (default=0.05)
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
			force_all_finite=True,
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
		
		
	def HenzeZirkler(self,X, alpha=.05):
	
		#from https://github.com/CPernet/Robust-Correlations/blob/master/HZmvntest.m
		#from https://pingouin-stats.org/generated/pingouin.multivariate_normality.html
	
		"""Henze-Zirkler multivariate normality test.
		Parameters
		----------
		X : np.array
			Data matrix of shape (n_samples, n_features).
		alpha : float
			Significance level.
		Returns
		-------
		hz : float
			The Henze-Zirkler test statistic.
		pval : float
			P-value.
		normal : boolean
			True if X comes from a multivariate normal distribution.
		See Also
		--------
		normality : Test the univariate normality of one or more variables.
		homoscedasticity : Test equality of variance.
		sphericity : Mauchly's test for sphericity.
		Notes
		-----
		The Henze-Zirkler test [1]_ has a good overall power against alternatives
		to normality and works for any dimension and sample size.
		Adapted to Python from a Matlab code [2]_ by Antonio Trujillo-Ortiz and
		tested against the
		`MVN <https://cran.r-project.org/web/packages/MVN/MVN.pdf>`_ R package.
		Rows with missing values are automatically removed.
		References
		----------
		.. [1] Henze, N., & Zirkler, B. (1990). A class of invariant consistent
		   tests for multivariate normality. Communications in Statistics-Theory
		   and Methods, 19(10), 3595-3617.
		.. [2] Trujillo-Ortiz, A., R. Hernandez-Walls, K. Barba-Rojo and L.
		   Cupul-Magana. (2007). HZmvntest: Henze-Zirkler's Multivariate
		   Normality Test. A MATLAB file.
		Examples
		--------
		>>> import pingouin as pg
		>>> data = pg.read_dataset('multivariate')
		>>> X = data[['Fever', 'Pressure', 'Aches']]
		>>> pg.multivariate_normality(X, alpha=.05)
		HZResults(hz=0.5400861018514641, pval=0.7173686509624891, normal=True)
		"""
		

		# Check input and remove missing values
	#    X = np.asarray(X)
	#    assert X.ndim == 2, 'X must be of shape (n_samples, n_features).'
	#    X = X[~np.isnan(X).any(axis=1)]
		
		n, p = X.shape
		# undersampling if length too long
		#print(n,p)
		if n>9999:
			factor=math.ceil(n/10000)
			X=X[::factor,:]


		n, p = X.shape
		#print(n,p)
		assert n >= 3, 'X must have at least 3 rows.'
		assert p >= 2, 'X must have at least two columns.'

		
		# Covariance matrix
		S = np.cov(X, rowvar=False, bias=True) #X.T, ddof=1)
		
		S_inv = np.linalg.pinv(S).astype(X.dtype)  # Preserving original dtype
		difT = X - X.mean(0)
		
		
		# Squared-Mahalanobis distances
		
		#Dj = np.diag(np.linalg.multi_dot([difT, S_inv, difT.T]))
		#T2=np.einsum('ij,ij->i', difT @ S_inv, difT)
		#print("T2.shape",T2.shape)
		T2=np.linalg.multi_dot([difT, S_inv, difT.T])
		
		
		Dj = np.diag(T2)
		
		Y = np.linalg.multi_dot([X, S_inv, X.T])
		
		Djk = -2 * Y.T + np.repeat(np.diag(Y.T), n).reshape(n, -1) + \
			np.tile(np.diag(Y.T), (n, 1))
		
		# Smoothing parameter
		b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * \
			(n**(1 / (p + 4)))
		
		# Is matrix full-rank (columns are linearly independent)?
		if np.linalg.matrix_rank(S) == p:
			hz = n * (1 / (n**2) * np.sum(np.sum(np.exp(-(b**2) / 2 * Djk))) - 2
					  * ((1 + (b**2))**(-p / 2)) * (1 / n)
					  * (np.sum(np.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
					  + ((1 + (2 * (b**2)))**(-p / 2)))
		else:
			hz = n * 4

		wb = (1 + b**2) * (1 + 3 * b**2)
		a = 1 + 2 * b**2
		# Mean and variance
		mu = 1 - a**(-p / 2) * (1 + p * b**2 / a + (p * (p + 2)
													* (b**4)) / (2 * a**2))
		si2 = 2 * (1 + 4 * b**2)**(-p / 2) + 2 * a**(-p) * \
			(1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4)) \
			- 4 * wb**(-p / 2) * (1 + (3 * p * b**4) / (2 * wb)
								  + (p * (p + 2) * b**8) / (2 * wb**2))
		
		# Lognormal mean and variance
		pmu = np.log(np.sqrt(mu**4 / (si2 + mu**2)))
		psi = np.sqrt(np.log((si2 + mu**2) / mu**2))
		
		# P-value
		pval = lognorm.sf(hz, psi, scale=np.exp(pmu))
		normal = True if pval > alpha else False

		#HZResults = namedtuple('HZResults', ['hz', 'pval', 'normal'])
		#return HZResults(hz=hz, pval=pval, normal=normal)
		return hz,pval,normal
	
	
	
	
	
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

	#print("\n--- Plot of Hotelling's T-squared scores on the test set ---\n")

	fig, ax = plt.subplots(figsize=(14, 8))
	plt.scatter(range(scaled_t2_scores.size), scaled_t2_scores)
	ucl_line = plt.axhline(y=0.1, color='r', linestyle='-')
	ax.set_title('Scaled Hotelling\'s T-squared scores')
	ax.set_xlabel('Index')
	ax.set_ylabel('Scaled Hotelling\'s T-squared score')
	ucl_line.set_label('UCL')
	plt.legend()
	fig.tight_layout()
	plt.show()
