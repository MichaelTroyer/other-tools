#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Author:
Michael D. Troyer

Date:
01/12/17

Purpose:
Exploratory Data Analysis

Comments:
"""

#--- Imports -------------------------------------------------------------------


# The Core
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Additionals
from data_science_tools import correlation_matrix
from sklearn.model_selection import train_test_split

# Data Transformation
from sklearn.preprocessing   import Imputer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.preprocessing   import LabelEncoder
from sklearn.preprocessing   import PolynomialFeatures
from data_science_tools      import PolynomialFeatures_labeled

# Scaling
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import RobustScaler
from sklearn.preprocessing   import Normalizer

# Feature Extraction
from sklearn.decomposition   import PCA
from sklearn.decomposition   import IncrementalPCA
from sklearn.decomposition   import RandomizedPCA
from sklearn.decomposition   import FactorAnalysis
from sklearn.decomposition   import NMF

# Classification
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import LinearSVC
from sklearn.svm             import SVC
from sklearn.naive_bayes     import GaussianNB
from sklearn.naive_bayes     import BernoulliNB
from sklearn.naive_bayes     import MultinomialNB

# Regression
from sklearn.linear_model    import LinearRegression
from sklearn.linear_model    import Lasso
from sklearn.linear_model    import Ridge

# Clustering
from sklearn.cluster         import KMeans
from sklearn.cluster         import AgglomerativeClustering
from sklearn.cluster         import DBSCAN

# Model Tuning and Cross-validation
from sklearn.grid_search     import GridSearchCV
from sklearn.model_selection import cross_val_score

# Classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Clustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import completeness_score


#--- Create Data Frames --------------------------------------------------------


from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()

fea_df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
tar_df = pd.DataFrame(raw_data.target, columns=['target'])

fea_df['target'] = tar_df

# The prediction class
target_id = 'target'

# Drop some of the columns
drop_columns = []

if drop_columns:
    fea_df.drop(drop_columns, axis=1, inplace=True)
del drop_columns

# One-hot encode columns
one_hot_columns = []
if one_hot_columns:
    fea_df = pd.get_dummies(fea_df, columns=one_hot_columns)
del one_hot_columns

desc_ = False

if desc_:
    print
    print ' Raw Input Data '.center(60, '-')
    print
    print fea_df.info()
    print fea_df.describe()

    print
    print ' Correlation '.center(60, '-')
    print
    print fea_df.corr()

    # Scatter matrix
    if len(fea_df.columns) <= 10:
        fea_df.hist()
        pd.scatter_matrix(fea_df)


#--- Prepare Variables ---------------------------------------------------------


print
print ' Data Transformations '.center(60, '-')
print

# Remove the target column from df
target_lbls = fea_df[target_id]
fea_df.drop(target_id, axis=1, inplace=True)

# Add polynomials and interactions
poly_thrshld = 1
if poly_thrshld:
    fea_df = PolynomialFeatures_labeled(fea_df, poly_thrshld)

print
print "Polynomial Order: {}".format(poly_thrshld)
print

# Get an updated feature list
features = fea_df.columns

# Rescale the data
scalers = {
          'Standard Scaler' : StandardScaler,
          'Min/Max Scaler'  : MinMaxScaler,
          'Robust Scaler'   : RobustScaler,
          'Normalizer'      : Normalizer
          }

scaler = 'Standard Scaler'

print
print "Scaler: {}".format(scaler if scaler else 'None')
print

if scaler:
    sclr = scalers[scaler]()
    sclr.fit(fea_df)
    fea_df = pd.DataFrame(sclr.transform(fea_df), columns=features)

# Apply a decomposition
decomps = {
           'PCA' : PCA,
           'Iterative PCA' : IncrementalPCA,
           'Randomized PCA' : RandomizedPCA,
           'NMF' : NMF
           }

decomposition = ''

print
print "Decomposition: {}".format(decomposition if decomposition else 'None')
print

if decomposition:
    n_components = len(features)
    dcmp = decomps[decomposition]()
    dcmp_scores = []

    for n in range(1, n_components+1):
        dcmp.n_components = n
        dcmp_scores.append([n, np.mean(cross_val_score(dcmp, fea_df))])

    optimal_n = max(dcmp_scores, key=lambda x: x[1])[0]
    print '\tOptimal number of components {}'.format(optimal_n)
    dcmp.n_components = optimal_n

    dcmp.fit(fea_df)
    fea_df = pd.DataFrame(dcmp.transform(fea_df))

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(dcmp.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')


#--- Exploratory Analysis ------------------------------------------------------


# Try to ID clusters within the entire data set to determine if seperable

_cluster = False

if _cluster:
    kmeans      = KMeans(
                         n_clusters=2
                                           )

    agglom      = AgglomerativeClustering(
                                           )

    dbscan      = DBSCAN(
                                           )

    clusterers  = {
                  'k-Means Clustering'       : kmeans,
                  'Agglomerative Clustering' : agglom,
                  'DBSCAN'                   : dbscan
                   }

    print
    print ' Clustering Results '.center(60, '-')
    print

    for name, model in clusterers.items():
        predict_lbls = model.fit_predict(fea_df)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(target_lbls)) - (1 if -1 in target_lbls else 0)

        print '\n{}:\n'.format(name)
        print 'Estimated number of clusters: {}'.format(n_clusters_)
        print 'Homogeneity: {:.3f}'.format(homogeneity_score(target_lbls, predict_lbls))
        print 'Completeness: {:.3f}'.format(completeness_score(target_lbls, predict_lbls))
        print 'V-measure: {:.3f}'.format(v_measure_score(target_lbls, predict_lbls))
        print 'Adjusted Rand Index: {:.3f}'.format(adjusted_rand_score(target_lbls, predict_lbls))
        print
        # Print the parameters
        print '\t\t', 'Params'.center(45, '-')
        for k, v in sorted(model.get_params().items()):
            if len(str(v)) > 15:
                v = str(v)[:15]
            print '\t\t|', k.ljust(25, '.'), str(v).rjust(15, '.'), '|'
        print '\t\t', ''.center(45, '-')
        print

#TODO: Cluster visualizations


#--- Prepare Classifiers--------------------------------------------------------


KNrNgb = KNeighborsClassifier(
                                       n_neighbors=1,        # Default is 5
                                       n_jobs=-1
                                       )

RandFr = RandomForestClassifier(
                                       n_estimators=100,     # Default is 100
                                       max_features='auto',  # Default is 'auto'
                                       max_depth=None,       # Default is None
                                       min_samples_split=2,  # Default is 2
                                       min_samples_leaf=1,   # Default is 1
                                       max_leaf_nodes=None,  # Default is None
                                       n_jobs=-1,
                                       random_state=42
                                       )

GrdtBC = GradientBoostingClassifier(
                                       loss='deviance',      # D: 'deviance'
                                       learning_rate=0.1,    # Default is 0.1
                                       n_estimators=100,     # Default is 100
                                       max_features='auto',  # Default is 'auto'
                                       max_depth=None,       # Default is None
                                       min_samples_split=2,  # Default is 2
                                       min_samples_leaf=1,   # Default is 1
                                       max_leaf_nodes=None,  # Default is None
                                       random_state=42
                                       )

LogReg = LogisticRegression(
                                       penalty='l1',         # Default is 'l2'
                                       C=250,                # Default is 1.0
                                       n_jobs=-1,
                                       solver='liblinear',
                                       random_state=42
                                       )

nnwMLP = MLPClassifier(
                                       max_iter=1000         # Default is 100
                                       )

LinSVC = LinearSVC(                    # Best Params: {'penalty': 'l2', 'C': 0.01}
                                       penalty='l2',         # Default is 'l2'
                                       dual=False,           # Default is 'True'
                                       C=0.01,               # Default is 1.0
                                       random_state=42
                                       )

svmSVC = SVC(
                                       kernel='rbf',         # Default is 'rbf'
                                       C=1.0,                # Default is 1.0
                                       gamma='auto',         # Default is 'auto'
                                       random_state=42
                                       )

classifiers = {
#              'K Nearest Neighbors'                   : KNrNgb,
#              'Random Forest'                         : RandFr,
#              'Gradient Boosted Decision Trees'       : GrdtBC,
              'Logistic Regression'                   : LogReg,
#              'Neural Network Multi-layer Perceptron' : nnwMLP,
#              'Linear Support Vector Classifier'      : LinSVC,
#              'Support Vector Machine'                : svmSVC
              }

# TODO: Raw classifier visualizations


#--- Evaluate Models -----------------------------------------------------------


print
print ' Model Results '.center(60, '-')
print

cv = 10
performance = defaultdict(float)

for name, model in classifiers.items():

    scores = cross_val_score(model, fea_df, target_lbls, cv=cv)
    print "\n{}:\n\n\tAccuracy: {:.3f} (+/- {:.3f})"\
          "".format(name, scores.mean(), scores.std() * 2)

    performance[name] = scores.mean()

    print
    print '\t\t', 'Params'.center(45, '-')
    for k, v in sorted(model.get_params().items()):
        if len(str(v)) > 15:
            v = str(v)[:15]
        print '\t\t|', k.ljust(25, '.'), str(v).rjust(15, '.'), '|'
    print '\t\t', ''.center(45, '-')
    print

    top_model   = max(performance.items(), key=lambda x: x[1])
    worst_model = min(performance.items(), key=lambda x: x[1])

# TODO: Evaluation visualizations


#--- Tune a Final Model --------------------------------------------------------


# Pass along the top or bottom performing cold model for further tuning
final_classifier_name = worst_model[0]  # Change
final_classifier = classifiers[final_classifier_name]

# Get a baseline for performance independent of the previous evaluation
cv = 10
scores = cross_val_score(final_classifier, fea_df, target_lbls, cv=cv)

print
print ' Model Cross-Validation '.center(60, '-')
print
print 'Top Model: {}'.format(final_classifier_name).center(60)
print
print
print " {} Baseline ".format(final_classifier_name).center(60, '#')
print
scores = cross_val_score(final_classifier, fea_df, target_lbls, cv=cv)

print "Accuracy: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std() * 2)
print
for item in scores:
    print item

# The model params to grid search
parameters = {'Linear Support Vector Classifier':
                 {'penalty': ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 250, 1000],
                  'dual': [True, False]},
              'Logistic Regression':
                  {'penalty': ['l1', 'l2'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 250, 1000],
                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']},
              'Gradient Boosted Decision Trees':
                  {'loss': ['deviance', 'exponential'],
                   'learning_rate': [0.01, 0.1, 1, 10, 100],
                   'n_estimators': [10, 100, 200, 400],
                   'max_features': ['auto', None, 1, 2, 4, 8],
                   'max_depth': [None, 3, 6, 9, 12],
                   'min_samples_split': [None, 1, 2, 4, 8, 16],
                   'min_samples_leaf': [None, 1, 2, 4, 8, 16],
                   'max_leaf_nodes': [None, 1, 2, 4, 8, 16]}
}

gscv = GridSearchCV(final_classifier,
                    parameters[final_classifier_name],
                    n_jobs=-1,
                    cv=cv,
                    error_score=-1)

gscv.fit(fea_df, target_lbls)

print
print " {} Grid Search ".format(final_classifier_name).center(60, '#')
print
print "Best Estimator: {}".format(gscv.best_estimator_)
print
print "Best Score: {}".format(gscv.best_score_)
print
print "Best Params: {}".format(gscv.best_params_)
print
for grid_score in gscv.grid_scores_:
    print grid_score


#--- Test Final Model --------------------------------------------------------


# Get the best estimator and rerun the analysis with a new train/test
X_train, X_test, y_train, y_test = train_test_split(fea_df,
                                                    target_lbls,
                                                    train_size = 0.5)

americas_next_top_model = eval(str(gscv.best_estimator_))
americas_next_top_model.fit(X_train, y_train)
predicted_y = americas_next_top_model.predict(X_test)

print
print " {} Final Trial ".format(final_classifier_name).center(60, '#')
print
print "Accuracy Score: {}".format(accuracy_score(y_test, predicted_y))
print
print "Confusion Matrix:"
print "|TN|FP|"
print "|FN|TP|"
print confusion_matrix(y_test, predicted_y)
print
print "Class Report:"
print classification_report(y_test, predicted_y)
print
