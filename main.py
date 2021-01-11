#!/usr/bin/env python
# coding: utf-8


from preprocessing import preprocessing
from algorithm import algorithm
import csv
import numpy as np
import pandas as pd
import sys

seed_list = range(20)
dataset_list = ['avila', 'biodeg', 'landcover',  'letter', 'magic',  'optdigits', 'page-blocks',  'penbased', 'ring', 'satimage', 'segment', 'spambase', 'steel', 'texture', 'twonorm', 'vehicle', 'vowel', 'waveform', 'winequality-red', 'winequality-white']
missing_rate_list = [10, 20, 30, 40, 50]
classifier_list = ['rf','nn','lr']
method_list = ['random', 'lc', 'lm', 'entropy', 'gini', 'cumir', 'proposed_lm']


seed = int(sys.argv[1])
dataset_id = int(sys.argv[2])
missing_rate_id = int(sys.argv[3])
classifier_id = int(sys.argv[4])
method_id = int(sys.argv[5])


dataset = dataset_list[dataset_id]
missing_rate = missing_rate_list[missing_rate_id]

classifier = classifier_list[classifier_id]
method = method_list[method_id]


m_imputations = 10
s = 50
t_max = 250
test_size=0.5


data_df = pd.read_csv('./datasets/{}.csv'.format(dataset), delimiter=',', header=None)
data = data_df.values

if len(data)>10000:
    np.random.seed(seed)
    random_sampled_idx = np.random.choice(len(data), 10000, replace=False)
    data = data[random_sampled_idx]



X = data[:,:-1]
y = data[:,-1]


X_unlabeled, X_labeled, y_unlabeled, y_labeled, X_tst, y_tst, delta_unlabeled,idx_unlabeled, imputed_list = preprocessing(X, y, missing_rate, m_imputations, s, seed, test_size)



accList = algorithm(X_unlabeled, X_labeled, y_unlabeled, y_labeled, X_tst, y_tst, delta_unlabeled,idx_unlabeled, imputed_list, method, t_max, classifier, seed, m_imputations)

a = open('./result_new/{0}_{1}_{2}_{3}_{4}.csv'.format(method, dataset, missing_rate, classifier, seed), 'w', newline='')
wra = csv.writer(a)
                        
print(method, ' dataset:',dataset, ' missing_rate:',missing_rate, ' classifier:', classifier, ' seed:', seed, ' acc:', accList,'\n')
                            
for i in accList:
  wra.writerow([i])
a.close()