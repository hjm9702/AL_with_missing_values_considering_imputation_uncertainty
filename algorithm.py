#!/usr/bin/env python
# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import *

def algorithm(X_unlabeled, X_labeled, y_unlabeled, y_labeled, X_tst, y_tst, delta_unlabeled,idx_unlabeled, imputed_list, method, t_max, classifier, seed, m_imputations):
    if classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state = seed)
        clf.fit(X_labeled, y_labeled)
      
    elif classifier == 'nn':
        clf = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', solver='lbfgs', random_state = seed)
        clf.fit(X_labeled, y_labeled)
          
    elif classifier == 'lr':
        clf = LogisticRegression(random_state = seed)
        clf.fit(X_labeled, y_labeled)
      
      
    accList = query_selection(X_unlabeled, X_labeled, y_unlabeled, y_labeled, X_tst, y_tst, delta_unlabeled, idx_unlabeled, imputed_list, method, t_max, seed, m_imputations, clf)

    return accList



def query_selection(X_unlabeled, X_labeled, y_unlabeled, y_labeled, X_tst, y_tst, delta_unlabeled, idx_unlabeled, imputed_list, method, t_max, seed, m_imputations, clf):
    
    accList = []
    
    t = 1
    
    y_tst_hat = clf.predict(X_tst)
    accuracy = accuracy_score(y_tst, y_tst_hat)
    accList.append(accuracy)

    if method == 'proposed':
        
        while t <= t_max:
            phi = CU(X_unlabeled, clf)
            proposed_phi = proposed_acquisition_function(phi, delta_unlabeled)
            x_star_idx = np.argmax(proposed_phi)
        
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            delta_unlabeled = np.delete(delta_unlabeled, x_star_idx, axis=0)
        
            clf.fit(X_labeled, y_labeled)
        
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
        
            t+=1
          
        return accList
    
    elif method == 'proposed_lc':
        
        while t <= t_max:
            phi = least_confidence(X_unlabeled, clf)
            proposed_phi = proposed_acquisition_function(phi, delta_unlabeled)
            x_star_idx = np.argmax(proposed_phi)
        
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            delta_unlabeled = np.delete(delta_unlabeled, x_star_idx, axis=0)
        
            clf.fit(X_labeled, y_labeled)
        
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
        
            t+=1
          
        return accList
        
    elif method == 'proposed_lm':
        
        while t <= t_max:
            phi = least_margin(X_unlabeled, clf)
            proposed_phi = proposed_acquisition_function(phi, delta_unlabeled)
            x_star_idx = np.argmax(proposed_phi)
        
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            delta_unlabeled = np.delete(delta_unlabeled, x_star_idx, axis=0)
        
            clf.fit(X_labeled, y_labeled)
        
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
        
            t+=1
          
        return accList
        
        
    elif method == 'lc':
        while t <= t_max:
            phi = least_confidence(X_unlabeled, clf)
            x_star_idx = np.argmax(phi)
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList
    
    elif method == 'lm':
        while t <= t_max:
            phi = least_margin(X_unlabeled, clf)
            x_star_idx = np.argmax(phi)
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList
    
    
        
    elif method == 'entropy':
    
        while t <= t_max:
            phi = entropy(X_unlabeled, clf)
            x_star_idx = np.argmax(phi)
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList
            
    
    elif method == 'cu':
    
        while t <= t_max:
            phi = CU(X_unlabeled, clf)
            x_star_idx = np.argmax(phi)
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList
        
    elif method == 'gini':
    
        while t <= t_max:
            phi = gini(X_unlabeled, clf)
            x_star_idx = np.argmax(phi)
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList
    
    
        
    elif method == 'cumir':
        
        while t <= t_max:
            phi = CUMIR(X_unlabeled, clf, idx_unlabeled, imputed_list)
            x_star_idx = np.argmax(phi)
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            idx_unlabeled = np.delete(idx_unlabeled, x_star_idx, axis=0)
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList
        
        
        
    elif method == 'random':
        
        while t <= t_max:
            
            x_star_idx = np.random.choice(len(X_unlabeled), 1)[0]
            
            X_labeled = np.concatenate((X_labeled, X_unlabeled[x_star_idx].reshape(-1,X_unlabeled.shape[1])), axis=0)
            y_labeled = np.concatenate((y_labeled, y_unlabeled[x_star_idx].reshape(-1)))
        
            X_unlabeled = np.delete(X_unlabeled, x_star_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, x_star_idx, axis=0)
            
            
            clf.fit(X_labeled, y_labeled)
            
            y_tst_hat = clf.predict(X_tst)
            acc = accuracy_score(y_tst, y_tst_hat)
            accList.append(acc)
            
            t+=1
        
        return accList

