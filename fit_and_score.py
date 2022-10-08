""" Functions for fitting shallow decision trees and random forests and scoring them. """
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
import pandas as pd
from sklearn import tree, ensemble
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.tree import export_text
import time

from preprocess import drop_and_one_hot 
from data_accessor import load_bot_repo_dataset

def timeit(func):
    def wrapper(*args, **kwargs):
        if 'silent' not in kwargs or not kwargs['silent']:
            start_time = time.time()
            print(f"Starting {func.__name__} at {time.strftime('%D %H:%M:%S')}")
        result = func(*args, **kwargs)
        if 'silent' not in kwargs or not kwargs['silent']:
            end_time = time.time()
            print(f"Finished {func.__name__} at {end_time}. Execution time: {end_time - start_time} s")
        return result
    return wrapper




def fit(X, y, method=None, depth=3):
    """
    Fit a decision tree (or method, if provided) using features X, labels y and maximum depth equal to depth. 
    """
    if method is None:
        # Use decision tree
        clf = tree.DecisionTreeClassifier(max_depth=depth)
    else:
        clf = clone(method)
    clf = clf.fit(X, y)
    return clf
 

def score(clf, X, y, method=None, silent=False, prec_rec=True):
    """
    Score trained model, and print out results if not silent.
    """
    if method is None and not silent:
        r = export_text(clf, feature_names=list(X.columns), show_weights=True)
        print(r)
    accuracy = clf.score(X, y)
    preds = clf.predict(X)
    if prec_rec:
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)
    else:
        precision = -1
        recall = -1
        f1 = -1
    if (not silent):
        print(f"Accuracy:", accuracy)
        print(f"Precision", precision)
        print(f"Recall:", recall)
        print(f"F1:", f1)
    return accuracy, precision, recall, f1


def fit_and_score(X, y, method=None, depth=3, silent=False, prec_rec=True):
    """ 
    Fit model, print out the ascii tree and scores and return the model/scores.
    """
    clf = fit(X, y, method, depth)
    accuracy, precision, recall, f1 = score(clf, X, y, method, silent, prec_rec)
    return clf, accuracy, precision, recall, f1


@timeit
def kfold_cv(X, y, method=None, depth=3, k=5, calibrate=False, silent=True, balance=False):
    """
    Run fit_and_score k times and compute test score statistics.
    """
    scores = []
    if balance:
        humans = X.loc[(y == 0).values]
        bots = X.loc[(y == 1).values]
        n_accts = min(len(humans), len(bots))
        balanced_labels = pd.Series([0]*n_accts + [1]*n_accts)
        balanced_X = pd.concat([humans.sample(n_accts), bots.sample(n_accts)])
        inds = np.random.permutation(len(balanced_X))
        shuffled_X = balanced_X.iloc[inds]
        shuffled_y = balanced_labels.iloc[inds]
    else:
        inds = np.random.permutation(len(X))
        shuffled_X = X.iloc[inds]
        shuffled_y = y.iloc[inds]
    fold_size = (len(shuffled_X) // k)+1
    for i in range(k):
        if not silent:
            print(f"Fold {i} in progress")
        # Train test split
        train_X = pd.concat([shuffled_X.iloc[:i*fold_size], shuffled_X.iloc[(i+1)*fold_size:]])
        test_X = shuffled_X.iloc[i*fold_size:(i+1)*fold_size]
        train_y = [label for j,label in enumerate(shuffled_y) if ((j < i*fold_size) or (j >= (i+1)*fold_size))]
        test_y = [label for j,label in enumerate(shuffled_y) if ((j >= i*fold_size) and (j < (i+1)*fold_size))]
        # Fit
        clf = fit(train_X, train_y, method=method, depth=depth)
        if not calibrate:
            scr = score(clf, test_X, test_y, method, silent)
        else:
            probs = clf.predict_proba(test_X)
            preds = [0 if p[0] > 0.5 else 1 for p in probs]
            scr = (accuracy_score(y, preds), precision_score(y, preds), recall_score(y, preds), f1_score(y, preds))
        scores.append(scr)
    avg_scores = [sum([row[i] for row in scores])/k for i in range(4)]
    return avg_scores
    
@timeit
def train_test_fit_and_score_clf(X, y, method=None, depth=3, silent=False, prec_rec=True, balance=False):
    """ Train test split. """
    if balance:
        balanced_X, balanced_y = balance_dataset(X, y)
        train, test, train_labels, test_labels = train_test_split(balanced_X, balanced_y, test_size=0.2)
    else:
        train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
    clf, *_ = fit_and_score(train, train_labels, method=method, depth=depth, silent=True, prec_rec=prec_rec)
    scr = score(clf, test, test_labels, method=method, silent=silent, prec_rec=prec_rec)
    return clf, scr


def train_test_fit_and_score(X, y, method=None, depth=3, silent=False, prec_rec=True, balance=False):
    """ Train test split. """
    _, scr = train_test_fit_and_score_clf(X, y, method, depth, silent, prec_rec, balance)
    return scr


def nonnumeric(df):
    """
    Print out columns that contain NA or have dtype=object.
    """
    print("Columns with NA values:", df.isnull().any()[lambda x: x])
    print("Columns with dtype=object", list(df.select_dtypes(include='object')))
    print("Columns with dtype=bool", list(df.select_dtypes(include='bool')))


@timeit
def mdi_feature_importance(clf, labels, ax):
    """
    Plot most important features.
    
    @param clf: fitted classifier
    """
    importances = clf.feature_importances_
    top10 = np.argsort(importances)[-10:] # Get top 10 most important features
    top10 = top10[::-1]
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    mdi_importances = pd.Series(importances, index=labels)
    mdi_importances[top10].plot.bar(yerr=std[top10], ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    return mdi_importances

@timeit
def permutation_feature_importance(ax, X, y, drop_cols):
    """
    Random forest feature importance on fitted model clf, features. Plot results on ax.
    Warning: computationally expensive, since refits model multiple times.
    """
    processed = drop_and_one_hot(X, drop_cols, [])
    
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    clf, *_ = fit_and_score(X, y, method=rf, silent=True)

    pi = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=1)
    top10 = np.argsort(pi.importances_mean)[-10:] # Get top 10 most important features
    top10 = top10[::-1]
    pi_importances = pd.Series(pi.importances_mean, index=list(X.columns))
    pi_importances[top10].plot.bar(ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")

def plot_metrics(one_hot, labels, soa_accuracy=None, soa_precision=None, soa_recall=None):
    """ Plot accuracy, precision and recall for different numbers of features. """
    rng = range(1,6)
    models = [train_test_fit_and_score(one_hot, labels, depth=i, silent=True) for i in rng]
    clfs, accuracies, precisions, recalls, f1s = zip(*models)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(rng, accuracies, label="Accuracy")
    ax.scatter(rng, precisions, label="Precision")
    ax.scatter(rng, recalls, label="Recall")
    ax.set_title("Accuracy, precision and recall as a function of model complexity")
    ax.set_xlabel("Model complexity (tree depth)")
    ax.set_ylabel("Score")
    ax.set_ylim([0,1.05])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if soa_accuracy:
        plt.axhline(y= soa_accuracy, color='C0', linestyle='-', label="State-of-art accuracy")
    if soa_precision:
        plt.axhline(y= soa_precision, color='C1', linestyle='-', label="State-of-art precision")
    if soa_recall:
        plt.axhline(y= soa_recall, color='C2', linestyle='-', label="State-of-art recall")
    ax.legend()
    plt.tight_layout()
    

def calculate_accuracy(precision, recall, num_bots, num_humans):
    """
    Calculate accuracy from precision, recall and support.
    """
    total = num_bots + num_humans
    true_positive = num_bots * recall
    true_negative = total - true_positive * (1/precision + 1/recall - 2)
    false_negative = num_bots - true_positive
    false_positive = num_humans - true_negative
    return (true_positive + true_negative) / total


def analyze_bot_repo_dataset(one_hot, labels, k=5, silent=False, kfold=True):
    """
    Compute k-fold cross validation for decision trees of depths 1-5, return scores for each.
    """
    if kfold:
        unbalanced_scores = [kfold_cv(one_hot, pd.Series(labels), depth=i, k=k, balance=False) for i in range(1,6)]
        balanced_scores = [kfold_cv(one_hot, pd.Series(labels), depth=i, k=k, balance=True) for i in range(1,6)]
        scores = [[u[0], u[1], u[2], u[3], b[0]] for u, b in zip(unbalanced_scores, balanced_scores)]
        return scores 
    return [train_test_fit_and_score(one_hot, labels, depth=i, silent=silent) for i in range(1,6)]


def analyze_bot_repo_dataset_full(data_path, labels_path, depth=5, folds=5, soa_accuracy=None, soa_precision=None, soa_recall=None):
    df, one_hot, labels = load_bot_repo_dataset(data_path, labels_path)
    # Fit and score on decision tree
    print("-------------- DECISION TREE --------------")
    plot_metrics(one_hot, labels, soa_accuracy=soa_accuracy, soa_precision=soa_precision, soa_recall=soa_recall)
    humans = len([lab for lab in labels if lab == 0])
    bots = len([lab for lab in labels if lab == 1])
    print("# humans: ", humans, "# bots", bots)
    # Fit and score on random forest
    print("-------------- RANDOM FOREST --------------")
    train, test, train_labels, test_labels = train_test_split(one_hot, labels, test_size=0.3)
    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=3)
    clf_sig = CalibratedClassifierCV(rf, method='sigmoid')
    rf_scores = kfold_cv(one_hot, labels, method=clf_sig, k=5)
    print("Random forest k-fold cv scores:", rf_scores)
    fig, axes = plt.subplots(1,2, figsize=(15,7))
    mdi_feature_importance(rf_clf, list(one_hot.columns), axes[0])
    permutation_feature_importance(axes[1], df, labels, DUMMY_COLUMNS + ['is_translator', 'contributors_enabled'])
    fig.tight_layout()
    print("-------------- DECISION TREE: k-fold cv --------------")
    return analyze_bot_repo_dataset(df, one_hot, labels, labels_path, k, soa_accuracy, soa_precision, soa_recall)


