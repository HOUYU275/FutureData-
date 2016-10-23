# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:00:26 2016

@author: Jun Liu
"""
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import random

random.seed(2016)
np.random.seed(2016)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
def intersect(a, b):
    return list(set(a) & set(b))
    
def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance
    
def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('category')
    return sorted(output)


def read_test_train(name):
    print("Load train.csv")
    train = pd.read_csv(name + "_train1.csv")
    print("Load test.csv")
    test = pd.read_csv(name + "_test1.csv")
    print("Process tables...")
    features = get_features(train, test)
    return train, test, features
    
def create_submission(dtest_predictions, dtest, name):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(name) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    ### merge the predictions with the test dataset
    indices = dtest.index
    dtest['avgprice'] = pd.Series(dtest_predictions, index=indices)
    ### save the results to csv files
    dtest.to_csv(sub_file, index = False)
    
def run_single(train, test, features, target, random_state=1):
    eta = 0.02
    max_depth = 5
    subsample = 0.9
    colsample_bytree = 0.9
    
    print('XGboost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eval_metric": "rmse",
        "eta": eta,
        "tree_method": 'auto',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 1000
    early_stopping_rounds = 50
    
    kf = KFold(len(train.index), n_folds=5, shuffle=True, random_state=random_state)
    train_index, test_index = list(kf)[0]
    print('Length of train: {}'.format(len(train_index)))
    print('Length of valid: {}'.format(len(test_index)))
    
    X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
    y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
                    
    print('Validating...')
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = mean_squared_error(y_valid, check)
    print('Check error value: {:.3f}'.format(score))
    
    imp = get_importance(gbm, features)
    print('Importance array: ', imp)
    
    print('Predict test set...')
    test_prediction = gbm.predict(xgb.DMatrix(test[features].as_matrix()), ntree_limit=gbm.best_iteration+1)
    
    return test_prediction, score

if __name__ == '__main__':    
    file_head_name = ['oil', 'flower', 'seafood', 'fruit', 'meat', 'hostveg', 'guestveg', 'vegetable', 'egg']
    for name in file_head_name:
        train, test, features = read_test_train(name)
        test_predictions, score = run_single(train, test, features, 'avgprice')
        create_submission(test_predictions, test, name)
        print('Mean squared error: ', score)
    
    
    