#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy.stats import poisson
from scipy.special import entr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



def missing_value_generator(X, missing_rate, seed):
    row_num = X.shape[0]
    column_num = X.shape[1]
    missing_value_average_each_row = column_num * (missing_rate/100)

    np.random.seed(seed)
    poisson_dist = poisson.rvs(mu = missing_value_average_each_row, size = row_num, random_state = seed)
    poisson_dist = np.clip(poisson_dist, 0, X.shape[1]-1)
    
    column_idx = np.arange(column_num)
    X_missing = X.copy().astype(float)
    for i in range(row_num):
        missing_idx = np.random.choice(column_idx, poisson_dist[i], replace=False)
        for j in missing_idx:
            X_missing[i,j] = np.nan
    
    
    return X_missing


def multiple_imputation(X_missing, m_imputations, seed):
    mice = IterativeImputer(sample_posterior=True, random_state = seed)
    mice.fit(X_missing)
    imputed_list = [mice.transform(X_missing) for i in range(m_imputations)]
        
    X_imputed = np.mean(imputed_list, axis=0)
    
    assert (X_imputed != imputed_list[0]).any()
    
    return X_imputed, imputed_list, mice


def imputation_uncertainty(imputed_list):
    delta = np.var(imputed_list, axis=0, ddof=1)
    delta = np.sum(delta, axis=1)
    
    return delta

def low_k_idx(array, k):
    array_idx = np.arange(len(array))
    sorted_array = sorted(array_idx, key=lambda x: array[x])
    low_k_index = sorted_array[:k]
    
    return low_k_index


def top_k_idx(array, k):
    array_idx = np.arange(len(array))
    sorted_array = sorted(array_idx, key=lambda x: array[x], reverse=True)
    top_k_index = sorted_array[:k]
    
    return top_k_index


def random_k_idx(array, k, seed):
    array_idx = np.arange(len(array))
    np.random.seed(seed)
    np.random.shuffle(array_idx)
    random_k_index = array_idx[:k]
    
    return random_k_index

def entropy(X_unlabeled, clf):
    predict_proba = clf.predict_proba(X_unlabeled)
    entropy_array = entr(predict_proba).sum(axis=1)
    
    return entropy_array


def least_confidence(X_unlabeled, clf):
    predict_proba = clf.predict_proba(X_unlabeled)
    lc_array = -np.max(predict_proba, axis=1)
    
    return lc_array

def least_margin(X_unlabeled, clf):
    predict_proba = clf.predict_proba(X_unlabeled)
    top_2_proba = np.sort(predict_proba, axis=1)[:,-2:]
    lm_array = - (top_2_proba[:,1] - top_2_proba[:,0])
    
    return lm_array    



def proposed_acquisition_function(phi, delta_unlabeled):
    sigma_phi = np.std(phi)
    sigma_delta = np.std(delta_unlabeled)
    alpha = (sigma_phi / sigma_delta).reshape(-1,1)
    
    proposed_phi = phi.reshape(-1,1) - alpha*delta_unlabeled.reshape(-1,1)
    
    return proposed_phi


def gini(X_unlabeled, clf):
    predict_proba = clf.predict_proba(X_unlabeled)
    squared_proba = predict_proba **2
    gini_value = 1- np.sum(squared_proba, axis=1)
    
    return gini_value
  
        
def CUMIR(X_unlabeled, clf, idx_unlabeled, imputed_list):
    
    
    within_variance_list = [CU(imputed_data[idx_unlabeled], clf) for imputed_data in imputed_list]
    within_variance = np.mean(within_variance_list, axis=0)
    
    
    proba_list = [clf.predict_proba(imputed_data[idx_unlabeled]) for imputed_data in imputed_list]
    Q_bar = np.mean(proba_list, axis=0)
    Q_bar_idx = np.argmax(Q_bar, axis=1)
    proba_idx_list = np.argmax(proba_list, axis=2)
    
    eq_list = [np.equal(Q_bar_idx, proba_idx_list[k]) for k in range(len(imputed_list))]
    eq_sum = np.sum(eq_list,axis=0)
    between_variance = (1/(len(imputed_list)-1))*(len(imputed_list)-eq_sum)
    
    total_variance = within_variance + (1+1/len(imputed_list))*between_variance
    
    return total_variance
    
    
