import pandas as pd

from data_accessor import get_shared_cols, balance_dataset
from fit_and_score import fit_and_score, score


def leave_dataset_out_helper(i, use_datasets, use_labels, datasets, dataset_names, labels, method, max_depth):

    cols = get_shared_cols(datasets)

    scores = {
        'left_out': dataset_names[i]
    }
    
    train_on = pd.concat([d[cols] for d in use_datasets])
    train_on_labels = pd.concat([pd.Series(d) for d in use_labels])
        
    test_on = datasets[i][cols]
    test_on_labels = labels[i]

    train_on_balanced, train_on_labels_balanced = balance_dataset(train_on, train_on_labels)
    test_on_balanced, test_on_labels_balanced = balance_dataset(test_on, test_on_labels)

    if method is not None:
        clf, *_ = fit_and_score(train_on[cols], train_on_labels, method=method, silent=True)
        a,p,r,f = (score(clf, test_on[cols], test_on_labels, silent=True))

        clf, *_ = fit_and_score(train_on_balanced[cols], train_on_labels_balanced, method=method, silent=True)
        ba, *_  = (score(clf, test_on_balanced[cols], test_on_labels_balanced, silent=True))

        scores.update({
            f'a_rf': a,
            f'p_rf': p,
            f'r_rf': r,
            f'f_rf': f,
            f'ba_rf': ba
            })
    else:
        for j in range(1,max_depth+1):
            clf, *_ = fit_and_score(train_on[cols], train_on_labels, depth=j, silent=True)
            a,p,r,f = (score(clf, test_on[cols], test_on_labels, silent=True))

            clf, *_ = fit_and_score(train_on_balanced[cols], train_on_labels_balanced, depth=j,  silent=True)
            ba, *_  = (score(clf, test_on_balanced[cols], test_on_labels_balanced, silent=True))
            scores.update({
                f'a{j}': a,
                f'p{j}': p,
                f'r{j}': r,
                f'f{j}': f,
                f'ba_rf': ba
                })
    return scores


def leave_dataset_out(i, datasets, dataset_names, labels, method=None, max_depth=4):
    
    use_datasets = datasets[:i] + datasets[i+1:]
    use_labels = labels[:i] + labels[i+1:]
    return leave_dataset_out_helper(i, use_datasets, use_labels, datasets, dataset_names, labels, method, max_depth)


def leave_dataset_out_botometer(i, datasets, dataset_names, human_df, method=None, max_depth=4):
    labels = [[1]*len(df) for df in datasets]
    use_datasets = datasets[:i] + datasets[i+1:]
    use_labels = [[1]*len(df) for df in use_datasets]
    use_datasets = use_datasets + [human_df]
    use_labels = use_labels + [[0]*len(human_df)]
    return leave_dataset_out_helper(i, use_datasets, use_labels, datasets, dataset_names, labels, method, max_depth)

