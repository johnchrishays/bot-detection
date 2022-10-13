import pandas as pd

from data_accessor import get_shared_cols
from fit_and_score import fit_and_score, score, train_test_fit_and_score_clf


def leave_dataset_out_helper(i, use_datasets, use_labels, datasets, dataset_names, labels, method, max_depth):

    cols = get_shared_cols(datasets)

    scores = {
        'left_out': dataset_names[i]
    }
    in_sample_scores = {
        'left_out': dataset_names[i]
    }
    
    train_on = pd.concat([d[cols] for d in use_datasets])
    train_on_labels = pd.concat([pd.Series(d) for d in use_labels])
        
    test_on = datasets[i][cols]
    test_on_labels = labels[i]

    if method is not None:
        clf, (in_a,in_p,in_r,in_f,in_ba) = train_test_fit_and_score_clf(train_on[cols], train_on_labels, method=method, silent=True)
        a,p,r,f,ba = (score(clf, test_on[cols], test_on_labels, silent=True))

        scores.update({
            f'a_rf': a,
            f'p_rf': p,
            f'r_rf': r,
            f'f_rf': f,
            f'ba_rf': ba
            })
        in_sample_scores.update({
            f'a_rf': in_a,
            f'p_rf': in_p,
            f'r_rf': in_r,
            f'f_rf': in_f,
            f'ba_rf': in_ba
            })
    else:
        for j in range(1,max_depth+1):
            clf, (in_a,in_p,in_r,in_f,in_ba) = fit_and_score(train_on[cols], train_on_labels, depth=j, silent=True)
            a,p,r,f,ba = (score(clf, test_on[cols], test_on_labels, silent=True))

            scores.update({
                f'a{j}': a,
                f'p{j}': p,
                f'r{j}': r,
                f'f{j}': f,
                f'ba{j}': ba
                })
            in_sample_scores.update({
                f'a{j}': in_a,
                f'p{j}': in_p,
                f'r{j}': in_r,
                f'f{j}': in_f,
                f'ba{j}': in_ba
                })
    return scores, in_sample_scores


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

