#!/usr/bin/env python
# coding: utf-8

from utils import missing_value_generator, multiple_imputation, imputation_uncertainty, random_k_idx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def preprocessing(X, y , missing_rate, m_imputations, s, seed, test_size):
    
    
    
    
    X = missing_value_generator(X, missing_rate, seed)
    
    X_unlabeled, X_tst, y_unlabeled, y_tst = train_test_split(X,y, stratify=y, random_state=seed, test_size=test_size)
    
    
    scaler = StandardScaler()
    X_unlabeled = scaler.fit_transform(X_unlabeled)
    X_tst = scaler.transform(X_tst)
    
    X_unlabeled, imputed_list, mice = multiple_imputation(X_unlabeled, m_imputations, seed)
    
    
    
    
    tst_imputed_list = [mice.transform(X_tst) for i in range(m_imputations)]
    X_tst = np.mean(tst_imputed_list, axis=0)

    delta_unlabeled = imputation_uncertainty(imputed_list)
    idx_unlabeled = np.arange(len(X_unlabeled)).astype(np.int)
    

    random_s_selection = random_k_idx(X_unlabeled, s, seed)

    
    X_labeled = X_unlabeled[random_s_selection]
    y_labeled = y_unlabeled[random_s_selection]
    
    
    X_unlabeled = np.delete(X_unlabeled,random_s_selection,axis=0)
    y_unlabeled = np.delete(y_unlabeled,random_s_selection,axis=0)
    delta_unlabeled = np.delete(delta_unlabeled, random_s_selection, axis=0)
    idx_unlabeled = np.delete(idx_unlabeled, random_s_selection, axis=0)
    idx_unlabeled = idx_unlabeled.astype(np.int)
    
    
    return X_unlabeled, X_labeled, y_unlabeled, y_labeled, X_tst, y_tst, delta_unlabeled,idx_unlabeled, imputed_list