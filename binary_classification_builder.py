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

#--- Imports --------------------------------------------------------------------------------------


# The Core
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Additionals
from sklearn.model_selection import train_test_split
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

# Model Tuning and Cross-validation
from sklearn.grid_search     import GridSearchCV
from sklearn.model_selection import cross_val_score

# Classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#--- Create Data Frames ---------------------------------------------------------------------------


from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()

fea_df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
tar_df = pd.DataFrame(raw_data.target, columns=['target'])

fea_df['target'] = tar_df

# The prediction class
target_id = 'target'


#--- Prepare Variables ----------------------------------------------------------------------------


print
print ' Data Transformations '.center(60, '-')
print

# Remove the target column from df
target_lbls = fea_df[target_id]
fea_df.drop(target_id, axis=1, inplace=True)

# Add polynomials and interactions
poly_thrshld = 0
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
           'PCA'             : PCA,
           'Iterative PCA'   : IncrementalPCA,
           'Randomized PCA'  : RandomizedPCA,
           'Factor Analysis' : FactorAnalysis,
           'NMF'             : NMF
           }

decomposition = 'PCA'

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


# Split the final test set out
fea_df, test_df, target_lbls, test_target_lbls = train_test_split(fea_df,
                                                                  target_lbls,
                                                                  train_size=0.8,
                                                                  random_state=42)


#--- Prepare Classifiers---------------------------------------------------------------------------


# Trained on []
KNrNgb_params = {'algorithm'                : 'auto',               # Default is 'auto'
                 'leaf_size'                : 1,                    # Default is 30
                  'metric'                  : 'manhattan',          # Default is 'minkowski'
                  'metric_params'           : None,                 # Default is None
                  'n_jobs'                  : 1,                    # Default is 1
                  'n_neighbors'             : 3,                    # Default is 5
                  'p'                       : 2,                    # Default is 2
                  'weights'                 : 'uniform'}            # Default is 'uniform'
KNrNgb = KNeighborsClassifier(**KNrNgb_params)
#KNrNgb = KNeighborsClassifier()


# Trained on []
RandFr_params = {'bootstrap'                : False,                # Default is True
                 'class_weight'             : None,                 # Default is None
                 'criterion'                : 'entropy',            # Default is 'gini'
                 'max_depth'                : None,                 # Default is None
                 'max_features'             : 2,                    # Default is 'auto'
                 'max_leaf_nodes'           : None,                 # Default is None
                 'min_impurity_split'       : 1e-07,                # Default is 1e-07
                 'min_samples_leaf'         : 1,                    # Default is 1
                 'min_samples_split'        : 2,                    # Default is 2
                 'min_weight_fraction_leaf' : 0.0,                  # Default is 0.0
                 'n_estimators'             : 42,                   # Default is 10
                 'n_jobs'                   : 1,                    # Default is 1
                 'oob_score'                : False,                # Default is False
                 'random_state'             : 42,                   # Default is None
                 'verbose'                  : False,                # Default is False
                 'warm_start'               : False}                # Default is False
RandFr = RandomForestClassifier(**RandFr_params)
#RandFr = RandomForestClassifier()


# Trained on []
GrdtBC_params = {'criterion'                : 'friedman_mse',       # Default is 'friedman_mse'
                 'init'                     : None,                 # Default is None
                 'learning_rate'            : 1,                    # Default is 0.1
                 'loss'                     : 'deviance',           # Default is 'deviance' 
                 'max_depth'                : 1,                    # Default is 3
                 'max_features'             : 10,                   # Default is None
                 'max_leaf_nodes'           : None,                 # Default is None
                 'min_impurity_split'       : 1e-07,                # Default is 1e-07
                 'min_samples_leaf'         : 1,                    # Default is 1
                 'min_samples_split'        : 2,                    # Default is 2
                 'min_weight_fraction_leaf' : 0.0,                  # Default is 0.0
                 'n_estimators'             : 200,                  # Default is 100
                 'presort'                  : 'auto',               # Default is 'auto'
                 'random_state'             : 42,                   # Default is None
                 'subsample'                : 1.0,                  # Default is 1.0
                 'verbose'                  : False,                # Default is False
                 'warm_start'               : False}                # Default is False
GrdtBC = GradientBoostingClassifier(**GrdtBC_params)
#GrdtBC = GradientBoostingClassifier()


# Trained on [0, SS, PCA] Best Score: 0.973684210526 [41, 1, 2, 69]
LogReg_params = {'C'                        : 0.08164183673469387,  # Default is 1.0
                 'class_weight'             : None,                 # Default is None
                 'dual'                     : True,                 # Default is False
                 'fit_intercept'            : True,                 # Default is True
                 'intercept_scaling'        : 0.0001,               # Default is 1
                 'max_iter'                 : 400,                  # Default is 100
                 'multi_class'              : 'ovr',                # Default is 'ovr'
                 'n_jobs'                   : -1,                   # Default is 1
                 'penalty'                  : 'l2',                 # Default is 'l2'
                 'random_state'             : 42,                   # Default is None
                 'solver'                   : 'liblinear',          # Default is 'liblinear'
                 'tol'                      : 1e-4,                 # Default is 1e-4
                 'verbose'                  : False,                # Default is False
                 'warm_start'               : False}                # Default is False
LogReg = LogisticRegression(**LogReg_params)
#LogReg = LogisticRegression()


# Trained on [] 
nnwMLP_params = {'activation'               : 'identity',           # Default is 'relu'
                 'alpha'                    : 0.0001,               # Default is 0.0001
                 'batch_size'               : 'auto',               # Default is 'auto'
                 'beta_1'                   : 0.9,                  # Default is 0.9
                 'beta_2'                   : 0.999,                # Default is 0.999
                 'early_stopping'           : False,                # Default is False
                 'epsilon'                  : 1e-08,                # Default is 1e-08
                 'hidden_layer_sizes'       : (100,),               # Default is (100,)
                 'learning_rate'            : 'constant',           # Default is 'constant'
                 'learning_rate_init'       : 0.0002,               # Default is 0.001
                 'max_iter'                 : 200,                  # Default is 200
                 'momentum'                 : 0.9,                  # Default is 0.9
                 'nesterovs_momentum'       : True,                 # Default is True
                 'power_t'                  : 0.5,                  # Default is 0.5
                 'random_state'             : 42,                   # Default is None
                 'shuffle'                  : True,                 # Default is True
                 'solver'                   : 'adam',               # Default is 'adam'
                 'tol'                      : 1e-4,                 # Default is 1e-4
                 'validation_fraction'      : 0.1,                  # Default is 0.1
                 'verbose'                  : False,                # Default is False
                 'warm_start'               : False}                # Default is False
nnwMLP = MLPClassifier(**nnwMLP_params)
#nnwMLP = MLPClassifier()


# Trained on [0, SS, PCA] Best Score: 0.991228070175 [41, 1, 0, 71]
LinSVC_params = {'C'                        : 0.020417959183673468, # Default is 1.0
                 'class_weight'             : None,                 # Default is None
                 'dual'                     : True,                 # Default is True
                 'fit_intercept'            : True,                 # Default is True
                 'intercept_scaling'        : 0.445,                # Default is 1
                 'loss'                     : 'hinge',              # Default is 'squared_hinge'
                 'max_iter'                 : 100,                  # Default is 100
                 'multi_class'              : 'ovr',                # Default is 'ovr'
                 'penalty'                  : 'l2',                 # Default is 'l2'
                 'random_state'             : 42,                   # Default is None
                 'tol'                      : 1e-4,                 # Default is 1e-4
                 'verbose'                  : False}                # Default is False
LinSVC = LinearSVC(**LinSVC_params)
#LinSVC = LinearSVC()



svmSVC_params = {'C'                        : 0.105,                # Default is 1.0
                 'cache_size'               : 200,                  # Default is 200
                 'class_weight'             : 'balanced',           # Default is None
                 'coef0'                    : 0.0,                  # Default is 0.0
                 'decision_function_shape'  : None,                 # Default is None
                 'degree'                   : 3,                    # Default is 3
                 'gamma'                    : 'auto',               # Default is 'auto'
                 'kernel'                   : 'linear',             # Default is 'rbf'
                 'max_iter'                 : -1,                   # Default is -1
                 'probability'              : False,                # Default is False
                 'random_state'             : 42,                   # Default is None
                 'shrinking'                : True,                 # Default is True
                 'tol'                      : 0.001,                # Default is 0.001
                 'verbose'                  : False}                # Default is False
svmSVC = SVC(**svmSVC_params)
#svmSVC = SVC()


classifiers = {
              'K Nearest Neighbors'                   : KNrNgb,
              'Random Forest'                         : RandFr,
              'Gradient Boosted Decision Trees'       : GrdtBC,
#              'Logistic Regression'                   : LogReg,    
              'Neural Network Multi-layer Perceptron' : nnwMLP,
#              'Linear Support Vector Classifier'      : LinSVC,
#              'Support Vector Machine'                : svmSVC
              }


#--- Evaluate Models ------------------------------------------------------------------------------


print
print ' Model Results '.center(60, '-')
print

cv = 5
performance = defaultdict(float)

for name, model in classifiers.items():

    scores = cross_val_score(model, fea_df, target_lbls, cv=cv)
    print "\n{}:\n\n\tAccuracy: {:.12f} (+/- {:.3f})".format(name, scores.mean(), scores.std() * 2)

    performance[name] = scores.mean()

    print
    print '\t\t', 'Params'.center(45, '-')
    for k, v in sorted(model.get_params().items()):
        if len(str(v)) > 15:
            v = str(v)[:15]
        print '\t\t|', k.ljust(25, '.'), str(v).rjust(15, '.'), '|'
    print '\t\t', ''.center(45, '-')
    print

    top_models = sorted(performance.items(), key=lambda x: x[1], reverse=True)


#--- Tune a Final Model ---------------------------------------------------------------------------


tune_final = True

if tune_final:
    # Pass along the top performing model for further intensive tuning
    final_classifier_name = top_models[0][0] 
    final_classifier = classifiers[final_classifier_name]
    
    
    # Get a baseline for performance independent of the previous evaluation
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
    
    print "Accuracy: {:.12f} (+/- {:.3f})".format(scores.mean(), scores.std() * 2)
    print
#    for num, score in zip(range(len(scores+1)), scores):
#        print "{}:\t{}".format(num+1, score)
    for score in scores:
        print score    
        
    # The model params to grid search
    
    low = np.linspace(0.00001, 1, 50)
    high = np.linspace(1.00001, 10000, 50)
    C_ = low.tolist()
    C_.append(high.tolist())
    
    parameters = {
       
    'K Nearest Neighbors':                  
        {
         'algorithm'                : ['ball_tree', 'kd_tree', 'brute'],
         'leaf_size'                : [1],
         'metric'                   : ['manhattan', 'euclidean', 'chebyshev', 
                                       'seuclidean', 'mahalanobis'],
         'n_neighbors'              : [1, 3, 5],
         'p'                        : [2],
         'weights'                  : ['distance', 'uniform']
         },  
    
    
    'Random Forest':
        {
         'bootstrap'                : [True, False], 
         'criterion'                : ['entropy', 'gini'],  
         'max_depth'                : [None],
         'max_features'             : [None, 2],   
         'max_leaf_nodes'           : [None],
         'min_impurity_split'       : [1e-07],
         'min_samples_leaf'         : [1],
         'min_samples_split'        : [2], 
         'min_weight_fraction_leaf' : [0.0],
         'n_estimators'             : [41, 42, 43], 
         'oob_score'                : [True, False]
         }, 
                                                                     
                     
    'Gradient Boosted Decision Trees':
        {
         'criterion'                : ['friedman_mse', 'mse', 'mae'], 
         'loss'                     : ['deviance', 'exponential'],    
         'learning_rate'            : [0.1, 1, 10],
         'max_depth'                : [None, 1],
         'max_features'             : [None, 1, 10],
         'max_leaf_nodes'           : [None, 1, 10],
         'min_impurity_split'       : [1e-07],
         'min_samples_leaf'         : [1],
         'min_samples_split'        : [2], 
         'min_weight_fraction_leaf' : [0.0],
         'n_estimators'             : [200],
         'presort'                  : ['auto'],
         'subsample'                : [1]
         },
    
                     
    'Logistic Regression':
        {
         'C'                        : C_,
         'class_weight'             : [None, 'balanced'],
         'dual'                     : [True, False],
         'fit_intercept'            : [True, False],
         'intercept_scaling'        : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
         'max_iter'                 : [400],
         'penalty'                  : ['l1', 'l2'],
         'solver'                   : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
#         'tol'                      : [0.0001]
         },
    
    
    'Neural Network Multi-layer Perceptron':
        {
         'activation'               : ['identity', 'relu', 'logistic', 'tanh'],
         'alpha'                    : C_, 
         'batch_size'               : ['auto'],
         'beta_1'                   : [0.9],   # adam
         'beta_2'                   : [0.999], # adam
         'early_stopping'           : [False], 
         'epsilon'                  : [1e-08],
         'hidden_layer_sizes'       : [(100,)], 
         'learning_rate'            : ['constant', 'invscaling', 'adaptive'],  # sgd
         'learning_rate_init'       : C_, 
         'max_iter'                 : [200], 
         'momentum'                 : [0.9],         # sgd
         'nesterovs_momentum'       : [True],        # sgd
         'power_t'                  : [0.5],         # sgd
         'shuffle'                  : [True],
         'solver'                   : ['adam', 'lbfgs', 'sgd'],
         'tol'                      : [0.0001],
         'validation_fraction'      : [0.1]          # early-stopping
         },
        
    
    'Linear Support Vector Classifier':
       {
        'C'                         : C_,
        'class_weight'              : [None, 'balanced'],  
        'dual'                      : [True, False],  
        'fit_intercept'             : [True, False],  
        'intercept_scaling'         : np.linspace(0.001, 1, 10),
        'loss'                      : ['hinge', 'squared_hinge'],
        'max_iter'                  : [100, 200, 400],
        'penalty'                   : ['l1', 'l2'],  
        'tol'                       : [0.0001]
        },
        
    
    'Support Vector Machine':
        {
         'C'                        : C_,
         'cache_size'               : [200],
         'coef0'                    : np.linspace(0.001, 1, 10),
         'class_weight'             : ['balanced', None],
         'degree'                   : range(1, 11), 
         'gamma'                    : ['auto'],
         'kernel'                   : ['linear', 'poly', 'rbf', 'sigmoid'],
         'max_iter'                 : [400],
         'probability'              : [True, False], 
         'shrinking'                : [True, False], 
         'tol'                      : [0.0001]
         }
    
    }
    
    gscv = GridSearchCV(final_classifier, parameters[final_classifier_name], 
                        n_jobs=-1, cv=cv, error_score=0)
    
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
    for grid_score in sorted(gscv.grid_scores_, key=lambda x: x[1], reverse=True)[:10]:
        print grid_score

# TODO: Gradient descent for individual param fitting


#--- Validate Final Model -------------------------------------------------------------------------


    # Get the best estimator and rerun the analysis with test_df
    americas_next_top_model = eval(str(gscv.best_estimator_))
    # Train on the whole data set less final validation set
    americas_next_top_model.fit(fea_df, target_lbls)
    predicted_y = americas_next_top_model.predict(test_df)
    
    print
    print " {} Final Trial ".format(final_classifier_name).center(60, '#')
    print
    print "Accuracy Score: {}".format(accuracy_score(test_target_lbls, predicted_y))
    print
    print "Confusion Matrix:"
    print "|TN|FP|"
    print "|FN|TP|"
    print confusion_matrix(test_target_lbls, predicted_y)
    print
    print "Class Report:"
    print classification_report(test_target_lbls, predicted_y)
#    print
