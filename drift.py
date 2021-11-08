from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

import numpy as np
from scipy import stats
from scipy.stats import lognorm
import math
from scipy import stats
from hotelling_t2 import HotellingT2
import pandas
import matplotlib.pyplot as plt

class Drift(BaseEstimator):
	"""Drift

	Drift is an unsupervised multivariate outlier detection.

	....

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
	>>> from TSquared.hotelling_t2 import HotellingT2
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

	def __init__(self,xgb_n_estimators=100,xgb_max_depth=3): #, alpha=0.05
		#self.alpha = alpha

		self.default_ucl = 'indep'
		self.inputs=[]
		self.dictmodel={}
		self.xgb_n_estimators=xgb_n_estimators
		self.xgb_max_depth=xgb_max_depth

	def fit(self, df,targets,inputs,feature_selection=True,feature_to_keep=[], y=None, model=None):
		"""
		Fit Drift. 

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Training set of samples, where n_samples is the number of samples
			and n_features is the number of features. It should be clean and
			free of outliers.
		feature_selection:bool (default=True)
			performs a spearman correlation on entries and keeps correlations higher than 0.1
		feature_to_keep: list
			if feature_selection active, avoids the filtration of customed features
		y : None
			Not used, present for scikit-learn's API consistency by convention.
		model: Drift (default=None)
			A Drift model to be used before training.

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
		self.inputs=inputs
		dfresidus=pandas.DataFrame()

		if feature_selection==True:
			print('Calculation of Correlation matrix for feature selection')
			dfcorr=df[inputs].corr()#(method='spearman')
			self.check_list_in_list(inputs,feature_to_keep)

		index=0
		for target in targets: #['PF309A[bar r]']: #
			index+=1
			self.dictmodel[target]={}  # init imbricated dictionary			
			
			
			if feature_selection==True:
				
				local_corrdic=dfcorr[target].to_frame().to_dict('dict')
				local_in=[k for (k,v) in local_corrdic[target].items() if abs(v) > 0.10]+feature_to_keep
				local_in = list(dict.fromkeys(local_in)) #remove potential duplicates following concat feature_to_keep
				self.dictmodel[target]['corrfilt']=local_in	

			else:
				self.dictmodel[target]['corrfilt']=False
				local_in=inputs[:]

			print('*******************************************')
			print("TARGET=",target)
			local_in.remove(target)
			print('inputs=',local_in)
			dftemp=df[inputs].dropna()
			X=dftemp[local_in].values
			y=dftemp[target].values
			#Xstd = StandardScaler().fit_transform(X)

			X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0,shuffle=False)

			#reg=make_pipeline(StandardScaler(), xgb.XGBRegressor(n_estimators=100,max_depth=3))
			reg=xgb.XGBRegressor(n_estimators=self.xgb_n_estimators,max_depth=self.xgb_max_depth,learning_rate=0.15,importance_type='weight')

			xgb_model = None
			if model and target in model.dictmodel:
				xgb_model = model.dictmodel[target]['model']
			reg.fit(X_train, y_train, xgb_model=xgb_model)
			
			self.dictmodel[target]['model']=reg
			
			predictions=reg.predict(X_train)
			predictscore=reg.score(X_train, y_train)
			print(predictscore)
			dfresidus[target+'_residu']=y_train-predictions
			###### Stocker les std des résidus ######
			self.dictmodel[target]['std_residu']=dfresidus[target+'_residu'].std()
			#####################################################
			
			###### Stocker Features_Importances par ordre ######   
			featimportance = {}
       
			lst=list(local_in)
			for i in range(0,len(lst)):
				featimportance[lst[i]]=reg.feature_importances_[i]
			
			self.dictmodel[target]['featimportance']={k: featimportance[k] for k in sorted(featimportance, key=featimportance.get,reverse=True)}

			print("features importance")
			print(self.dictmodel[target]['featimportance'])
			#####################################################
			
			plt.scatter(x=y_train,y=predictions)
			plt.show()

			predictions=reg.predict(X_test)
			predictscore=reg.score(X_test, y_test)
			print(predictscore)

			plt.scatter(x=y_test,y=predictions)
			plt.show()
			
		
		print(dfresidus.head())	
		monit= HotellingT2()
		self.dictmodel["tsquared"]={}  # init imbricated dictionary	
		monit.cleanfit(dfresidus.values,iter=1)
		self.dictmodel["tsquared"]['model']=monit
		#t2_scores = monit.score_samples(dfresidus.values)
		#define threshold  ->  np.percentile(dfout['T2'].rolling(3000).mean().dropna(), 97)
			
		return self.dictmodel
		#return self

	def score_samples(self, df):
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

		#check_is_fitted(self)
		#X = self._check_test_inputs(X)
		index=0

		df = df.copy()
		
		for i in self.dictmodel:
			index+=1
			#print(i)
			#print(self.dictmodel[i])
			if i == "tsquared":
				break
			if self.dictmodel[i]['corrfilt']==False:
				local_in=self.inputs[:]
				#print(local_in)
				local_in.remove(i)
				#print("***DEBUG2****",local_in)
			else:
				local_in=self.dictmodel[i]['corrfilt']
				#print("***DEBUG****",local_in)
			Model=self.dictmodel[i]['model']
			
			df.loc[:,i+'_pred'] = Model.predict(df[local_in].values)
			df.loc[:,i+'_residu'] =df[i] - df[i+'_pred'] 
		
		cols = [col for col in df.columns if 'residu' in col]		
		
		TSquared = self.dictmodel["tsquared"]['model']	
		#print(df[cols].head())
		df.loc[:,"T2"] = TSquared.score_samples(df[cols].values)
		
		return df

	def diagnostic(self,in_df,t2_threshold=100):  
		"""
		Apply score_samples to a well-defined time entity (test, batch...) represented by the pandas dataframe in_df,
		and interpret results on the whole time entity.
		Interpretation starts with the mean of the T2 on the time entity.
		If the mean of T2 is over t2_threshold, diagnostic is triggered and backward analysis is done, 
		first on the zscore of the residus, then on the features importance. 

		Parameters
		----------
		in_df: pandas dataframe, shape (n_samples, n_features)
		t2_threshold : int (default=100)
			triggers diagnostic

		Returns
		-------
		NULL (a report is printed)

		"""
		dfout = self.score_samples(in_df)
		T2mean=dfout['T2'].mean()
		
		dicresidus={}
			
		#Compare T2 MEAN to HISTORICAL HIGHEST??? T2 on Training data??
		
		if T2mean > t2_threshold:
			# Look at the residus
			print(T2mean,"> t2_threshold (=",t2_threshold,")")
			for i in self.dictmodel.keys():
			#for i in [col for col in dfout.columns if 'residu' in col]:
				#rechercher les std et calculer combien de fois le z-score pour chaque, on a ensuite un ordre d'importance entre les résidus
				if i=="tsquared":
					break
				print(i)

				dicresidus[i+"_residu"]=abs(dfout[i+"_residu"].mean()/self.dictmodel[i]['std_residu'])
				print('z-score residu',i,"_residu ",str(dicresidus[i+"_residu"]))


			print("***** FIRST METHOD *****")

			sorted_dicresidus={k: dicresidus[k] for k in sorted(dicresidus, key=dicresidus.get,reverse=True)}
			print(sorted_dicresidus)
			
			for j in sorted_dicresidus.keys():
				if sorted_dicresidus[j]>3:
					print("Please, check",j[:-7]," sensor")

					print("or direct relatives:")
					print({k:v for (k,v) in self.dictmodel[j[:-7]]['featimportance'].items() if v > 0.10})
					

			print("***** SECOND METHOD *****")

			#Idée pour amélioration du système d'information
			# dans le cas où plusieurs capteurs auraient un z-score résidu élevé (>3)
			#-> filtrer sur ces capteurs
			#-> standardiser par rapport à la valeur la plus haute -> facteur par capteur	

			
			filt_sorted_dicresidus = {k:v for k,v in sorted_dicresidus.items() if v > 3}
			if filt_sorted_dicresidus:
				max_value = max(filt_sorted_dicresidus.values())
				if max_value > 0:
					for k in filt_sorted_dicresidus:
						filt_sorted_dicresidus[k] = filt_sorted_dicresidus[k]/max_value

				# et appliquer (multiplier) ce facteur à l'importance des features du capteur correspondant

				# z={}
				# for j in [x,y]:
				# 	print(j)
				# 	for (k,v) in j.items():
				# 		if k in z:
				# 			if z[k] < j[k]:
				# 				z[k]=v
				# 		else:
				# 			z[k]=v

				finaldict={}
				for j,jval in filt_sorted_dicresidus.items():
					finaldict[j[:-7]]=filt_sorted_dicresidus[j]
					for (k,v) in self.dictmodel[j[:-7]]['featimportance'].items():
						if v > 0.10:
							if k in finaldict:
								if finaldict[k] < v*jval:
									finaldict[k]=v*jval
							else:
								finaldict[k]=v*jval

				#-> supprimer les doublons, retrier et proposer la vérification de ces paramètres dans cet ordre
				#Ex: FNM et WFE ont des zscore résidus très élevé - 10 et 8 à cause d'un capteur N1 à zéro
				# 10 -> 1 et 8 -> 8/10=0,8
				# attribuer ces valeurs à ces paramètres FNM=1, WFE=0,8
				#multiplier les features relatives par ces coefficients
				#N1 sera dans les deux pools de features avec une importance différentes ex: 0,3 et 0,2
				#doublon N1=1x0,3 et N1=0,8x0,2   -> on supprime la valeur la plus faible.
				#dans l'ordre de vérification, on aurait FNM, WFE, N1, ...

				sorted_finaldict={k: finaldict[k] for k in sorted(finaldict, key=finaldict.get,reverse=True)}
				print(sorted_finaldict)
				
				print("Please check these sensors in this order")
				for j in sorted_finaldict.keys():
						print(j)
			else:
				print("Can't find zscore with value >3")			
		else:
			print(str(T2mean)," is below threshold ",str(t2_threshold))



		return 0
	
	def check_list_in_list(self,list1,list2):
		'''    
		check if list1 contains all elements in list2
		'''
		result =  all(elem in list1  for elem in list2)
		if result == False:
  			raise Exception("Error,",list1,"does not contains all elements in ",list2)
		
		return result	




	################# Rest From hotellingT2 #############"	
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
	import pandas as pd
	from time import process_time
	import numpy as np
	import time
	import matplotlib.pyplot as plt
	import h5py
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.neural_network import MLPRegressor
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	from sklearn import linear_model
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.pipeline import make_pipeline
	from sklearn.linear_model import LinearRegression

	import xgboost as xgb
	from hotelling_t2 import HotellingT2

	filename="../data/LEAP-proc2.hdf5"
	tabname='LEAP'

	dfi = pd.DataFrame(np.array(h5py.File(filename)[tabname]))

	filter_col=['datetime','TestID','EngineID','EngineType','EngineRating']
	for k in filter_col:
			dfi[k] = dfi[k].apply(lambda x: x.decode()) 
			
	# DISTINCTION LEAP PRODUCTION / MAINTENANCE 

	dfi["prod_or_maint"]=dfi['EngineType'].str.contains("rodu", regex=False)
	
	dfi['PLC.1024.388_diff']=dfi['PLC.1024.388[Â°]'].diff() 
	memory=15
	difflist=[]
	for i in range(1,memory):
		dfi['PLC.1024.388_diff'+str(i)]=dfi['PLC.1024.388_diff'].shift(periods=i) 
		difflist.append('PLC.1024.388_diff'+str(i))
		
	dfclean=dfi.drop_duplicates(subset='datetime', keep="last").reset_index()


	def Echelon(x): 
		if x > 35 :    
			return 1

	def Echelon_inverse(x): 
		if x < -10 :    
			return 1


	dfclean['Echelon']=dfclean['PLC.1024.388_diff'].apply(Echelon)
	dfclean['Echelon_inverse']=dfclean['PLC.1024.388_diff'].apply(Echelon_inverse)
	
	
	inputs=['PSCE01[bar r]',
	'PSCE02[bar r]',
	'PSCE03[bar r]',
	'PSCE04[bar r]',
	'PSCE05[bar r]',
	'PSCE06[bar r]',
	'P0_PREMIUM[mbar]',
	'RH[%]',
	'FNM[daN]']
	targets=['PSCE01[bar r]',
	'PSCE02[bar r]',
	'PSCE03[bar r]']
	keep=inputs+['datetime','Echelon','Echelon_inverse','TSMAX','TimeSpan[s]']
	mask = (dfclean['datetime'] < '2020-01-01T00:00:00')
	mask2 = (dfclean['datetime'] > '2020-01-01T00:00:00')


	dftrain=dfclean[keep].loc[mask]
	dftest=dfclean[keep].loc[mask2]

	#filter

	dftrain2=dftrain.loc[(dfclean['TSMAX'] > 1500)] #& (dfclean["Echelon"].isna()) & (dfclean["Echelon_inverse"].isna())]

	dftrain2[inputs]



	###### MAIN CODE #####        
			

	od=Drift()
	start=process_time()
	out=od.fit(dftrain2,targets,inputs)
	stop=process_time()
	print(out)
	print("time=",str(stop-start))

	start=process_time()
	dfout=od.score_samples(dfclean)
	stop=process_time()

	print("time=",str(stop-start))

	#COMPUTE T2 /CYCLE

	dfout['T2med']=dfout['T2'].rolling(5).median()

	dfout['cycle']=0
	dfout.loc[dfout['TimeSpan[s]'] <=1, 'cycle'] = 1
	dfout['cycle']=dfout['cycle'].cumsum()

	group=dfout[['cycle','T2med']].groupby('cycle').mean()
	group.rename(columns={'T2med': 'T2meancycle'}, inplace=True)
	dfout=dfout.set_index('cycle').join(group,on='cycle').reset_index()

	del dfout['index']

	
	out=['cycle',
	'TimeSpan[s]',
	'P0_PREMIUM[mbar]',
	'P0_PREMIUM[mbar a]',
	'P0_PREMIUM[bar a]',
	'FNM[daN]',
	'RH[%]',
	'PSCE01[bar r]',
	'PSCE02[bar r]',
	'PSCE03[bar r]',
	'PSCE04[bar r]',
	'PSCE05[bar r]',
	'PSCE06[bar r]',
	'TestID',
	'EngineID',
	'EngineType',
	'EngineRating',
	'datetime',
	'TSDiff',
	'TSMAX',
	'prod_or_maint',
	'Echelon',
	'Echelon_inverse',
	'PSCE01[bar r]_pred',
	'PSCE01[bar r]_residu',
	'PSCE02[bar r]_pred',
	'PSCE02[bar r]_residu',
	'PSCE03[bar r]_pred',
	'PSCE03[bar r]_residu',
	'PSCE04[bar r]_pred',
	'PSCE04[bar r]_residu',
	'PSCE05[bar r]_pred',
	'PSCE05[bar r]_residu',
	'PSCE06[bar r]_pred',
	'PSCE06[bar r]_residu',
	'T2',
	'T2med',
	'T2meancycle']

	dfout[out].to_csv("../data/drift_main_test.csv")