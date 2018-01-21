# -*- coding: utf-8 -*-
# Created on Sun Jan 21 2018 18:53:27
# Author: WuLC
# EMail: liangchaowu5@gmail.com

#####################################################
# This script offers a framework for stacking, a ensemble method in machine learning
#####################################################

import numpy as np
from sklearn.model_selection import KFold


# load train data and test data as ndarray
# a m×n ndarray means that there are m samples， while each sample has n dimension feature
x_train_file = './data/selectedFeatures/X_train_select.npy'
y_train_file = './data/selectedFeatures/label.npy'
x_test_file = './data/selectedFeatures/X_test_select.npy'
x_train = np.load(x_train_file).astype(np.float)
y_train = np.load(y_train_file).astype(np.float)
x_test = np.load(x_test_file).astype(np.float)
print(x_train.shape, y_train.shape, x_test.shape)


class BasicModel(object):
    """Parent class of basic models"""
    def train(self, x_train, y_train, x_val, y_val):
        """return a trained model and eval metric o validation data"""
        pass
    
    def predict(self, model, x_test):
        """return the predicted result"""
        pass
    
    def get_oof(self, x_train, y_train, x_test, n_folds = 5):
        """K-fold stacking"""
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        oof_train = np.zeros((num_train,)) 
        oof_test = np.zeros((num_test,))
        oof_test_all_fold = np.zeros((num_test, n_folds))
        aucs = []
        KF = KFold(n_splits = n_folds, random_state=2017)
        for i, (train_index, val_index) in enumerate(KF.split(x_train)):
            print('{0} fold, train {1}, val {2}'.format(i, len(train_index), len(val_index)))
            x_tra, y_tra = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]
            model, auc = self.train(x_tra, y_tra, x_val, y_val)
            aucs.append(auc)
            oof_train[val_index] = self.predict(model, x_val)
            oof_test_all_fold[:, i] = self.predict(model, x_test)
        oof_test = np.mean(oof_test_all_fold, axis=1)
        print('all aucs {0}, average {1}'.format(aucs, np.mean(aucs)))
        return oof_train, oof_test


# create two models for first-layer stacking: xgb and lgb
import xgboost as xgb
class XGBClassifier(BasicModel):
    def __init__(self):
        """set parameters"""
        self.num_rounds=1000
        self.early_stopping_rounds = 15
        self.params = {
            'objective': 'binary:logistic',
            'eta': 0.1,
            'max_depth': 8,
            'eval_metric': 'auc',
            'seed': 0,
            'silent' : 0
         }
        
    def train(self, x_train, y_train, x_val, y_val):
        print('train with xgb model')
        xgbtrain = xgb.DMatrix(x_train, y_train)
        xgbval = xgb.DMatrix(x_val, y_val)
        watchlist = [(xgbtrain,'train'), (xgbval, 'val')]
        model = xgb.train(self.params, 
                          xgbtrain, 
                          self.num_rounds)
                          watchlist,
                          early_stopping_rounds = self.early_stopping_rounds)
        return model, float(model.eval(xgbval).split()[1].split(':')[1])

    def predict(self, model, x_test):
        print('test with xgb model')
        xgbtest = xgb.DMatrix(x_test)
        return model.predict(xgbtest)

import lightgbm as lgb
class LGBClassifier(BasicModel):
    def __init__(self):
        self.num_boost_round = 2000
        self.early_stopping_rounds = 15
        self.params = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'num_leaves': 80,
            'learning_rate': 0.05,
            # 'scale_pos_weight': 1.5,
            'feature_fraction': 0.5,
            'bagging_fraction': 1,
            'bagging_freq': 5,
            'max_bin': 300,
            'is_unbalance': True,
            'lambda_l2': 5.0,
            'verbose' : -1
            }
        
    def train(self, x_train, y_train, x_val, y_val):
        print('train with lgb model')
        lgbtrain = lgb.Dataset(x_train, y_train)
        lgbval = lgb.Dataset(x_val, y_val)
        model = lgb.train(self.params, 
                          lgbtrain,
                          valid_sets = lgbval,
                          verbose_eval = self.num_boost_round,
                          num_boost_round = self.num_boost_round)
                          early_stopping_rounds = self.early_stopping_rounds)
        return model, model.best_score['valid_0']['auc']
    
    def predict(self, model, x_test):
        print('test with lgb model')
        return model.predict(x_test, num_iteration=model.best_iteration)


# get output of first layer models and construct as input for the second layer          
lgb_classifier = LGBClassifier()
lgb_oof_train, lgb_oof_test = lgb_classifier.get_oof(x_train, y_train, x_test)
print(lgb_oof_train.shape, lgb_oof_test.shape)        
    
xgb_classifier = XGBClassifier()
xgb_oof_train, xgb_oof_test = xgb_classifier.get_oof(x_train, y_train, x_test)
print(xgb_oof_train.shape, xgb_oof_test.shape)

input_train = [xgb_oof_train, lgb_oof_train] 
input_test = [xgb_oof_test, lgb_oof_test]

stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
print(stacked_train.shape, stacked_test.shape)


# use LR as the model of the second layer
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# split for validation
n = int(stacked_train.shape[0] * 0.8)
x_tra, y_tra = stacked_train[:n], y_train[:n]
x_val, y_val = stacked_train[n:], y_train[n:]
model = LinearRegression()
model.fit(x_tra,y_tra)
y_pred = model.predict(x_val)
print(metrics.roc_auc_score(y_val, y_pred))

# predict on test data
final_model = LinearRegression()
final_model.fit(stacked_train, y_train)
test_prediction = final_model.predict(stacked_test)