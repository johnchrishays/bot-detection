import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

from fit_and_score import fit_and_score

def split_human_df(human_df, n_datasets, dataset_size):
    human_df_sample = human_df.sample(n=n_datasets*dataset_size)
    return [human_df_sample[i*dataset_size:(i+1)*dataset_size] for i in range(n_datasets)]

    
def get_bot_classifier(bots, humans, bots_name, dataset_size, cols, test_aggregate, test_labels_aggregate, method=None, depth=3):
    dataset_size = min(len(bots), len(humans))
    n=min(dataset_size, dataset_size)

    X = pd.concat([bots.sample(n=n)[cols], humans.sample(n=n)[cols]])
    y = [1] * n + [0] * n

    X = X.fillna(0)
    train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.05)
    test_aggregate.append(test)
    test_labels_aggregate.append(test_labels)

    scores = {
        'dataset': bots_name
    }

    clf, *_ = fit_and_score(train, train_labels, method=method, depth=depth, silent=True)
    return clf

def get_confidence(value, i):
    pair = value[i][0]
    return max(pair)/sum(pair)

def get_class(value, i):
    pair = value[i][0]
    return np.argmax(pair)

def get_max_confidence(confidences, i):
    conf_i = [confidences[j][i] for j in range(n_classifiers)]
    return np.argmax(conf_i), max(conf_i)

def get_max_confidence_pred(predictions, confidences, i):
    argmaximum, maximum = get_max_confidence(confidences, i)
    return predictions[argmaximum][i]
    
def get_majority_vote(botometer_datasets, predictions, i):
    preds_i = [predictions[j][i] for j in range(n_classifiers)]
    majority = sum(preds_i)/len(botometer_datasets)
    return 1 if majority >= 1/2 else 0

def get_weighted_majority_vote(predictions, confidences, i):
    conf_i = [confidences[j][i] for j in range(n_classifiers)]
    preds_i = [2*predictions[j][i]-1 for j in range(n_classifiers)]
    weights = [conf_i[j] * preds_i[j] for j in range(n_classifiers)]
    return 1 if sum(weights) > 0 else 0

def get_max_captured(value, i):
    pair = value[i][0]
    return max(pair)

def get_max_captured_multiclassifier(predictions, captured, i):
    return get_max_confidence_pred(predictions, captured, i)

def get_ensemble_of_classifiers(botometer_datasets, human_dfs, bot_names, method=None, depth=3):
    test_aggregate = []
    test_labels_aggregate = []
    dataset_size = 1200
    human_df = pd.concat(human_dfs)

    cols = set.intersection(
        *[set(b.columns) for b in botometer_datasets],
        *[set(h.columns) for h in human_dfs]
    )
    if 'created_at' in cols:
        cols.remove('created_at')
    cols = list(cols)

    clfs = [get_bot_classifier(bots, humans, bot_name, dataset_size, cols, test_aggregate, test_labels_aggregate, method=method, depth=depth) for bots, humans, bot_name in zip(botometer_datasets, human_dfs, bot_names)]
    test_aggregate_df = pd.concat(test_aggregate)
    test_labels_aggregate_concat = np.concatenate(test_labels_aggregate)

    # make human classifier
    test_all = []
    test_all_labels = []
    all_bots = pd.concat([d[cols] for d in botometer_datasets])
    all_clf = get_bot_classifier(all_bots, human_df, "all-bots", min(len(all_bots), len(human_df)), cols, test_all, test_all_labels, method=method, depth=depth)
    test_all = test_all[0]
    test_all_labels = test_all_labels[0]

    clfs = clfs + [all_clf]
    test_aggregate_df = pd.concat([test_aggregate_df, test_all])
    test_labels_aggregate_concat = np.concatenate([test_labels_aggregate_concat, test_all_labels])
    
    return clfs, test_aggregate_df, test_labels_aggregate_concat

def score_ensemble_speicalized_trees(clfs, test_aggregate_df, test_labels_aggregate_concat, botometer_datasets):
    n_classifiers = len(clfs)
    captured = [list(map(lambda j: get_max_captured(clfs[i].tree_.value, j), clfs[i].apply(test_aggregate_df))) for i in range(n_classifiers)]
    confidences = [list(map(lambda j: get_confidence(clfs[i].tree_.value, j), clfs[i].apply(test_aggregate_df))) for i in range(n_classifiers)]
    predictions = [clfs[i].predict(test_aggregate_df) for i in range(n_classifiers)]

    max_confidences = [get_max_confidence_pred(predictions, confidences, i) for i in range(len(test_aggregate_df))]
    majority_votes = [get_majority_vote(botometer_datasets, predictions, i) for i in range(len(test_aggregate_df))]
    weighted_majority_votes = [get_weighted_majority_vote(predictions, confidences, i) for i in range(len(test_aggregate_df))]

    print(f"Accuracy using max confidences: {accuracy_score(max_confidences, test_labels_aggregate_concat)}")
    print(f"Accuracy using majority vote: {accuracy_score(majority_votes, test_labels_aggregate_concat)}")
    print(f"Accuracy using weighted majority vote: {accuracy_score(weighted_majority_votes, test_labels_aggregate_concat)}")
    print(f"F1 using max confidences: {f1_score(max_confidences, test_labels_aggregate_concat)}")
    print(f"F1 using majority votes: {f1_score(majority_votes, test_labels_aggregate_concat)}")
    print(f"F1 using weighted majority votes: {f1_score(weighted_majority_votes, test_labels_aggregate_concat)}")

    most_captured = [get_max_captured_multiclassifier(predictions, captured, i) for i in range(len(test_aggregate_df))]
    weighted_majority_captured = [get_weighted_majority_vote(predictions, captured, i) for i in range(len(test_aggregate_df))]

    print(f"Accuracy using most captured: {accuracy_score(most_captured, test_labels_aggregate_concat)}")
    print(f"Accuracy using weighted most captured: {accuracy_score(weighted_majority_captured, test_labels_aggregate_concat)})")

    print(f"F1 using most captured: {f1_score(most_captured, test_labels_aggregate_concat)}")
    print(f"F1 using weighted majority captured: {f1_score(weighted_majority_captured, test_labels_aggregate_concat)}")

def get_max_confidence_rf(probabilities, i):
    conf_i = [probabilities[j][i] for j in range(n_classifiers)]
    conf_i[-1] = 1 - conf_i[-1]
    return np.argmax(conf_i), max(conf_i)

# Ensemble of specialized decision trees

def split_human_df(human_df, n_datasets, dataset_size):
    human_df_sample = human_df.sample(n=n_datasets*dataset_size)
    return [human_df_sample[i*dataset_size:(i+1)*dataset_size] for i in range(n_datasets)]

botometer_datasets = [simple_df, spammers_df, fake_followers_df, self_declared_df, political_bots_df, other_bots]
human_dfs = split_human_df(human_df, len(botometer_datasets), dataset_size)

bot_names = [get_dataset_name(d) for d in botometer_datasets]


clfs, test_aggregate_df, test_labels_aggregate_concat = get_ensemble_of_classifiers(botometer_datasets, human_dfs, bot_names, depth=4)


score_ensemble_speicalized_trees(clfs, test_aggregate_df, test_labels_aggregate_concat)

rf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clfs, test_aggregate_df, test_labels_aggregate_concat = get_ensemble_of_classifiers(botometer_datasets, human_dfs, bot_names, method=rf)

# Ensemble of specialized random forests

predictions = [rf_clfs[i].predict(test_aggregate_df) for i in range(n_classifiers)]
probs = [rf_clfs[i].predict_proba(test_aggregate_df) for i in range(n_classifiers)]
probs = [p[:,0] for p in probs]
most_confident_clf = [get_max_confidence_rf(probs, i)[0] for i in range(len(test_aggregate_df))]
max_prediction = [predictions[j][i] for i,j in enumerate(most_confident_clf)]
print(f"Accuracy: {accuracy_score(max_prediction, test_labels_aggregate_concat)}")
print(f"F1: {f1_score(max_prediction, test_labels_aggregate_concat)}")


# Just simple decision rules

all_clf = clfs[-1]
all_predictions = all_clf.predict(test_aggregate_df)
print(f"Accuracy: {accuracy_score(all_predictions, test_labels_aggregate_concat)}")
print(f"F1: {f1_score(all_predictions, test_labels_aggregate_concat)}")

# Just random forest 

all_clf_rf = rf_clfs[-1]
all_predictions_rf = all_clf_rf.predict(test_aggregate_df)
print(f"Accuracy: {accuracy_score(all_predictions_rf, test_labels_aggregate_concat)}")
print(f"F1: {f1_score(all_predictions_rf, test_labels_aggregate_concat)}")

# Testing on held-out datasets

hold_out_test_df = pd.concat([cresci_stock_2018_one_hot[cols], cresci_rtbust_2019_one_hot[cols], gilani_2017_one_hot[cols]])
hold_out_test_labels = pd.concat([cresci_stock_labels, rtbust_labels, gilani_labels])

score_ensemble_speicalized_trees(clfs, hold_out_test_df, hold_out_test_labels)

score_ensemble_speicalized_trees(clfs, cresci_stock_2018_one_hot[cols], cresci_stock_labels)

score_ensemble_speicalized_trees(clfs, cresci_rtbust_2019_one_hot[cols], rtbust_labels)

score_ensemble_speicalized_trees(clfs, gilani_2017_one_hot[cols], gilani_labels)

all_predictions = all_clf.predict(hold_out_test_df)
print(f"Accuracy: {accuracy_score(all_predictions, hold_out_test_labels)}")
print(f"F1: {f1_score(all_predictions, hold_out_test_labels)}")

all_predictions = all_clf.predict(cresci_stock_2018_one_hot[cols])
print(f"Accuracy: {accuracy_score(all_predictions, cresci_stock_labels)}")
print(f"F1: {f1_score(all_predictions, cresci_stock_labels)}")

all_predictions = all_clf.predict(cresci_rtbust_2019_one_hot[cols])
print(f"Accuracy: {accuracy_score(all_predictions, rtbust_labels)}")
print(f"F1: {f1_score(all_predictions, rtbust_labels)}")

all_predictions = all_clf.predict(gilani_2017_one_hot[cols])
print(f"Accuracy: {accuracy_score(all_predictions, gilani_labels)}")
print(f"F1: {f1_score(all_predictions, gilani_labels)}")

all_clf_rf = rf_clfs[-1]
all_predictions_rf = all_clf_rf.predict(hold_out_test_df)
print(f"Accuracy: {accuracy_score(all_predictions_rf, hold_out_test_labels)}")
print(f"F1: {f1_score(all_predictions_rf, hold_out_test_labels)}")

all_predictions_rf = all_clf_rf.predict(cresci_stock_2018_one_hot[cols])
print(f"Accuracy: {accuracy_score(all_predictions_rf, cresci_stock_labels)}")
print(f"F1: {f1_score(all_predictions_rf, cresci_stock_labels)}")

all_predictions_rf = all_clf_rf.predict(cresci_rtbust_2019_one_hot[cols])
print(f"Accuracy: {accuracy_score(all_predictions_rf, rtbust_labels)}")
print(f"F1: {f1_score(all_predictions_rf, rtbust_labels)}")

all_predictions_rf = all_clf_rf.predict(gilani_2017_one_hot[cols])
print(f"Accuracy: {accuracy_score(all_predictions_rf, gilani_labels)}")
print(f"F1: {f1_score(all_predictions_rf, gilani_labels)}")
