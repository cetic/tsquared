# -*- coding: utf-8 -*-
"""Benchmark of all implemented algorithms
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
from time import time

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define data file and read X and y
mat_file_list = [#'arrhythmia.mat', too few points compared to number of variables
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                 'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'shuttle.mat',
                 'vertebral.mat',
                 'vowels.mat',
                 'wbc.mat']

# define the number of iterations
n_ite = 5
n_classifiers = 15

df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
              'ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest', 'KNN', 'LOF', 'MCD',
              'OCSVM', 'PCA', 'TSquared','TSquared_with_cleaning','TSquared_with_2cleaning','TSquared_with_3cleaning','TSquared_autocleaning']
# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

for j in range(len(mat_file_list)):

    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join('/content/drive/MyDrive/MultivariateMonitoring/PYOD/data', mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
            contamination=outliers_fraction),
            'Cluster-based Local Outlier Factor': CBLOF(
                n_clusters=10,
                contamination=outliers_fraction,
                check_estimator=False,
                random_state=random_state),
            'Feature Bagging': FeatureBagging(contamination=outliers_fraction,
                                              random_state=random_state),
            'Histogram-base Outlier Detection (HBOS)': HBOS(
                contamination=outliers_fraction),
            'Isolation Forest': IForest(contamination=outliers_fraction,
                                        random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
            'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction),
            'Minimum Covariance Determinant (MCD)': MCD(
                contamination=outliers_fraction, random_state=random_state),
            'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
            'Principal Component Analysis (PCA)': PCA(
                contamination=outliers_fraction, random_state=random_state),
        }
        classifiers_indices = {
            'Angle-based Outlier Detector (ABOD)': 0,
            'Cluster-based Local Outlier Factor': 1,
            'Feature Bagging': 2,
            'Histogram-base Outlier Detection (HBOS)': 3,
            'Isolation Forest': 4,
            'K Nearest Neighbors (KNN)': 5,
            'Local Outlier Factor (LOF)': 6,
            'Minimum Covariance Determinant (MCD)': 7,
            'One-class SVM (OCSVM)': 8,
            'Principal Component Analysis (PCA)': 9,
            'TSquared': 10,
            'TSquared_with_cleaning': 11,
            'TSquared_with_2cleaning': 12,
            'TSquared_with_3cleaning': 13,
            'TSquared_autocleaning': 14
        }

        for clf_name, clf in classifiers.items():
            t0 = time()
            clf.fit(X_train_norm)
            test_scores = clf.decision_function(X_test_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)

            roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
            prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

            print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name=clf_name, roc=roc, prn=prn, duration=duration))

            time_mat[i, classifiers_indices[clf_name]] = duration
            roc_mat[i, classifiers_indices[clf_name]] = roc
            prn_mat[i, classifiers_indices[clf_name]] = prn
        
       

        #TSQUARED 
        clf=HotellingT2()
        clf_name='TSquared'
        t0 = time()
        clf.fit(X_train_norm)
        test_scores = clf.score_samples(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)

        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name='TSquared', roc=roc, prn=prn, duration=duration))

        time_mat[i, classifiers_indices[clf_name]] = duration
        roc_mat[i, classifiers_indices[clf_name]] = roc
        prn_mat[i, classifiers_indices[clf_name]] = prn
        #TSQUARED CLEANING
        clf=HotellingT2()
        clf_name='TSquared_with_cleaning'
        t0 = time()
        clf.fit(X_train_norm)
        clf.set_default_ucl('not indep')
        Xtrans=clf.transform(X_train_norm)
        clf.set_default_ucl('indep')
        clf.fit(Xtrans)
        test_scores = clf.score_samples(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        

        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name='TSquared_with_cleaning', roc=roc, prn=prn, duration=duration))

        time_mat[i, classifiers_indices[clf_name]] = duration
        roc_mat[i, classifiers_indices[clf_name]] = roc
        prn_mat[i, classifiers_indices[clf_name]] = prn
      #TSQUARED 2CLEANING
        clf=HotellingT2()
        clf_name='TSquared_with_2cleaning'
        t0 = time()
        clf.fit(X_train_norm)
        clf.set_default_ucl('not indep')
        Xtrans=clf.transform(X_train_norm)
        clf.fit(Xtrans)
        Xtrans2=clf.transform(Xtrans)
        clf.set_default_ucl('indep')
        clf.fit(Xtrans2)
        test_scores = clf.score_samples(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        

        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name='TSquared_with_2cleaning', roc=roc, prn=prn, duration=duration))

        time_mat[i, classifiers_indices[clf_name]] = duration
        roc_mat[i, classifiers_indices[clf_name]] = roc
        prn_mat[i, classifiers_indices[clf_name]] = prn
      #TSQUARED 3CLEANING
        clf=HotellingT2()
        clf_name='TSquared_with_3cleaning'
        t0 = time()
        clf.fit(X_train_norm)
        clf.set_default_ucl('not indep')
        Xtrans=clf.transform(X_train_norm)
        clf.fit(Xtrans)
        Xtrans2=clf.transform(Xtrans)
        clf.fit(Xtrans2)
        Xtrans3=clf.transform(Xtrans2)
        clf.set_default_ucl('indep')
        clf.fit(Xtrans3)
        test_scores = clf.score_samples(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        

        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name='TSquared_with_3cleaning', roc=roc, prn=prn, duration=duration))

        time_mat[i, classifiers_indices[clf_name]] = duration
        roc_mat[i, classifiers_indices[clf_name]] = roc
        prn_mat[i, classifiers_indices[clf_name]] = prn


      # TSquared_autocleaning
        clf=HotellingT2()
        clf_name='TSquared_autocleaning'
        t0 = time()
        n=X_train_norm.shape[0] # var init
        totpoints=X_train_norm.shape[0] #const
        Xtrans2=X_train_norm
        clf.set_default_ucl('not indep')
        while ((n>5) and (Xtrans2.shape[0] > totpoints/2)):
          print(n)
          Xtrans=Xtrans2
          clf.fit(Xtrans)
          Xtrans2=clf.transform(Xtrans)
          #print(Xtrans2.shape[0],"***",Xtrans.shape[0])
          n=Xtrans.shape[0]-Xtrans2.shape[0]
          #print(n)
        clf.set_default_ucl('indep')
        clf.fit(Xtrans2)
        test_scores = clf.score_samples(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        

        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name='TSquared_with_autocleaning', roc=roc, prn=prn, duration=duration))

        time_mat[i, classifiers_indices[clf_name]] = duration
        roc_mat[i, classifiers_indices[clf_name]] = roc
        prn_mat[i, classifiers_indices[clf_name]] = prn




    print(time_mat)
    print(time_list)
    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    print(time_list)
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)


    # Save the results for each run
    time_df.to_csv('time.csv', index=False, float_format='%.3f')
    roc_df.to_csv('roc.csv', index=False, float_format='%.3f')
    prn_df.to_csv('prc.csv', index=False, float_format='%.3f')
