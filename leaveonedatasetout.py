import pandas as pd

from data_accessor import get_shared_cols
from fit_and_score import fit_and_score, score

def leave_dataset_out(i, datasets, dataset_names, labels, method, max_depth=4):
    
    use_datasets = datasets[:i] + datasets[i+1:]
    use_labels = labels[:i] + labels[i+1:]

    cols = get_shared_cols(datasets)

    scores = {
        'left_out': dataset_names[i]
    }
    
    train_on = pd.concat([d[cols] for d in use_datasets])
    train_on_labels = pd.concat([pd.Series(d) for d in use_labels])
        
    test_on = datasets[i][cols]
    test_on_labels = labels[i]
    if method is not None:
        clf, *_ = fit_and_score(train_on[cols], train_on_labels, method=method, silent=True)
        a,p,r,f = (score(clf, test_on[cols], test_on_labels, silent=True))
        scores.update({
            f'a_rf': a,
            f'p_rf': p,
            f'r_rf': r,
            f'f_rf': f
            })
    else:
        for j in range(1,max_depth+1):
            clf, *_ = fit_and_score(train_on[cols], train_on_labels, method=rf, depth=j, silent=True)
            a,p,r,f = (score(clf, test_on[cols], test_on_labels, silent=True))
            scores.update({
                f'a{j}': a,
                f'p{j}': p,
                f'r{j}': r,
                f'f{j}': f
                })
    return scores




