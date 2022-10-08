

def leave_dataset_out(i):
    
    use_datasets = datasets[:i] + datasets[i+1:]
    use_labels = labels[:i] + labels[i+1:]

    scores = {
        'left_out': get_dataset_name(datasets[i])
    }
    
    train_on = pd.concat([d[cols] for d in use_datasets])
    train_on_labels = pd.concat([pd.Series(d) for d in use_labels])
        
    test_on = datasets[i][cols]
    test_on_labels = labels[i]
    
    for j in range(1,max_depth+1):
        rf = ensemble.RandomForestClassifier(n_estimators=100)
        clf, *_ = fit_and_score(train_on[cols], train_on_labels, method=rf, depth=j, silent=True)
        a,p,r,f = (score(clf, test_on[cols], test_on_labels, silent=True))
        scores = {
            **scores, 
            f'a{j}': a,
            f'p{j}': p,
            f'r{j}': r,
            f'f{j}': f
        }
    return scores


def print_leave_one_out_table(df):
    max_depth = 5
    tolerance = 0.025

    for row in df.to_dict(orient='records'):
        k = row['left_out']

        accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
        f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
        accuracy_sdt = max(accuracies)
        f1_sdt = max(f1s)

        max_ind, accuracy_sdt, f1_sdt = get_shallower_tree(0.025, accuracies, f1s)
                
        ret = k
        if k.endswith('_one_hot'):
            ret = k[:-8]
        if k.endswith("_df"):
            ret = k[:-3]
        ret = ret.replace('_', '-')

        print(f"{ret} & {accuracy_sdt:0.2f}/{f1_sdt:0.2f} & {max_ind+1} \\\\")

 

