import numpy as np
import pandas as pd

from fit_and_score import score, fit_and_score, train_test_fit_and_score
from data_accessor import balance_dataset

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def get_dataset_name(df):
    try:
        return namestr(df, globals())[0]
    except:
        return "Unknown"

def train_on_one_test_on_another(train_on, train_on_labels, test_on, test_on_labels, train_on_name, test_on_name, method=None, balance=True, silent=True):
    cols = ([e for e in list(train_on) if e in list(test_on)])
    #print(cols)
    scores = {
        'train_on': train_on_name,
        'test_on': test_on_name,
    }
    for i in range(1,6):
        if balance:
            train_on_balanced, train_on_labels_balanced = balance_dataset(train_on, train_on_labels)
            clf, *_ = fit_and_score(train_on_balanced[cols], train_on_labels_balanced, method=method, silent=silent)
            test_on_balanced, test_on_labels_balanced = balance_dataset(test_on, test_on_labels)
            a,p,r,f = (score(clf, test_on_balanced[cols], test_on_labels_balanced, silent=True))
        else:
            clf, *_ = fit_and_score(train_on[cols], train_on_labels, method=rf, silent=silent)
            a,p,r,f = (score(clf, test_on[cols], test_on_labels, silent=True))
        
        scores = {
            **scores, 
            f'a{i}': a,
            f'p{i}': p,
            f'r{i}': r,
            f'f{i}': f
        }
    if not silent:
        print(f"trained on: {train_on_name}, tested on: {test_on_name}, acc: {a:.2}, prec: {p:.2}, recall: {r:.2}, f1: {f:.2}, test bot freq: {np.round_(prop_bots,2)}")

    return scores  

def train_test_botometer_combined(bots, humans, max_depth, silent=True):
    dataset_size = min(len(bots), len(humans))
    
    cols = set.intersection(
        set(bots.columns),
        set(humans.columns)
    )
    if 'created_at' in cols:
        cols.remove('created_at')
    cols = list(cols)
    
    X = pd.concat([bots.sample(n=dataset_size)[cols], humans.sample(n=dataset_size)[cols]])
    X = X.fillna(0)
    y = pd.Series([1] * dataset_size + [0] * dataset_size)
        
        
    scores = {
        'dataset': get_dataset_name(bots),
    }
    
    for i in range(1, max_depth+1):
        a,p,r,f = train_test_fit_and_score(X, y, depth=i, silent=silent, balance=True)
        #prop_bots = sum(test_on_labels)/len(test_on_labels)
        scores.update({
            f'a{i}': a,
            f'p{i}': p,
            f'r{i}': r,
            f'f{i}': f,
        })
        
    return scores
        

def train_on_one_test_on_another_botometer_combined(bots1, bots2, humans, max_depth, silent=True):
    dataset_size = min(len(bots1), len(bots2), len(humans))
    bot1_inds = np.random.permutation(dataset_size)
    human1_inds = np.random.permutation(dataset_size)
    
    bot2_inds = np.random.permutation(dataset_size)
    human2_inds = np.random.permutation(dataset_size)
    
    cols = set.intersection(
        set(bots1.columns),
        set(bots2.columns),
        set(humans.columns)
    )
    if 'created_at' in cols:
        cols.remove('created_at')
    
    train_on = pd.concat([bots1.iloc[bot1_inds][cols], humans.iloc[human1_inds][cols]])
    train_on_labels = [1] * dataset_size + [0] * dataset_size
    train_on = train_on[cols].fillna(0)
    
    test_on = pd.concat([bots2.iloc[bot2_inds][cols], humans.iloc[human2_inds][cols]])
    test_on_labels = [1] * dataset_size + [0] * dataset_size
    test_on = test_on[cols].fillna(0)
    
    
    scores = {
        'train_on': get_dataset_name(bots1),
        'test_on': get_dataset_name(bots2),
    }
    
    for i in range(1, max_depth+1):
        clf, *_ = fit_and_score(train_on, train_on_labels, depth=i, silent=silent)
        a,p,r,f = score(clf, test_on, test_on_labels, silent=silent)
        #prop_bots =:: sum(test_on_labels)/len(test_on_labels)
        scores = {
            **scores, 
            f'a{i}': a,
            f'p{i}': p,
            f'r{i}': r,
            f'f{i}': f
        }
        
    return scores
