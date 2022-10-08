import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plotting import process_tick_label

def print_leave_one_out_table(df):
    max_depth = 5
    tolerance = 0.025

    for row in df.to_dict(orient='records'):
        k = row['left_out']

        accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
        a_max_ind = np.argmax(accuracies)
        f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
        f_max_ind = np.argmax(f1s)
        accuracy_sdt = accuracies[a_max_ind]
        f1_sdt = f1s[f_max_ind]

        for i, acc in enumerate(accuracies):
            if accuracy_sdt - acc <= tolerance:
                a_max_ind = i
                accuracy_sdt = acc
                break
        f_max_ind = i
        f1_sdt = f1s[i]
                
        ret = k
        if k.endswith('_one_hot'):
            ret = k[:-8]
        if k.endswith("_df"):
            ret = k[:-3]
        ret = ret.replace('_', '-')

        print(f"{ret} & {accuracy_sdt:0.2f}/{f1_sdt:0.2f} & {a_max_ind+1} \\\\")

 
def botometer_leave_dataset_out(datasets, i):
    use_datasets = datasets[:i] + datasets[i+1:]
    num_bots = sum([len(d) for d in datasets[:i]]) + sum([len(d) for d in datasets[i+1:]])
    
    dataset_size = min(num_bots, len(datasets[i]), len(human_df))
    
    bot_train_inds = np.random.permutation(n)
    human_train_inds = np.random.permutation(n)
    
    bot_test_inds = np.random.permutation(dataset_size)
    human_test_inds = np.random.permutation(dataset_size)
    

    scores = {
        'left_out': get_dataset_name(datasets[i])
    }
    
    train_on_bots = pd.concat([d[cols] for d in use_datasets]).iloc[bot_train_inds]
    train_on = pd.concat([train_on_bots, human_df.iloc[human_train_inds][cols]])
    train_on_labels = [1]*n + [0]*n
    train_on = train_on.fillna(0)

        
    test_on_bots = datasets[i][cols].iloc[bot_test_inds]
    test_on = pd.concat([test_on_bots, human_df.iloc[human_test_inds][cols]])
    test_on_labels = [1]*dataset_size + [0]*dataset_size
    test_on = test_on.fillna(0)

    
    for j in range(1,max_depth+1):
        clf, *_ = fit_and_score(train_on, train_on_labels, depth=j, silent=True)
        a,p,r,f = (score(clf, test_on, test_on_labels, silent=True))
        scores = {
            **scores, 
            f'a{j}': a,
            f'p{j}': p,
            f'r{j}': r,
            f'f{j}': f
        }
    return scores

def get_score(score, dataset_name):
    if score == "":
        return -1
    match = re.search(f"all: ([0-9]*(\.\d+)?)", score)
    if match:
        return -1
    match = re.search(f"{dataset_name}: ([0-9]*(\.\d+)?)", score)
    if match: 
        return float(match.group(1))
    return float(score)

def get_max_score(df, dataset_name, metric):
    scores = df[df['dataset(s) used'].str.contains(dataset_name)][metric].map(lambda x: get_score(x, dataset_name))
    max_score_ind = scores.idxmax()
    return scores.loc[max_score_ind], df.at[max_score_ind, 'bibtex_id']

def print_single_dataset_score_table(score_dict):
    max_depth = 5
    tolerance = 0.025

    for k,v in score_dict.items():
        accuracy_sota = float(v['accuracy'][0])
        f1_sota = float(v['f1'][0])
        row = sdt_df[sdt_df['name'] == k].to_dict(orient="records")[0]
        accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
        a_max_ind = np.argmax(accuracies)
        f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
        f_max_ind = np.argmax(f1s)
        accuracy_sdt = accuracies[a_max_ind]
        f1_sdt = f1s[f_max_ind]

        for i, acc in enumerate(accuracies):
            if accuracy_sdt - acc <= tolerance:
                a_max_ind = i
                accuracy_sdt = acc
                break
        for i, f in enumerate(f1s):
            if f1_sdt - f <= tolerance:
                f_max_ind = i
                f1_sdt = f
                break



        accuracy_diff = accuracy_sdt - accuracy_sota
        f1_diff = f1_sdt - f1_sota
        if v['accuracy'][1] == v['f1'][1]:
            cite = f"{v['accuracy'][1]}"
        else:
            cite = f"{v['accuracy'][1]}, {v['f1'][1]}"
        sepa = '\\phantom{-}' if accuracy_diff > 0 else ''
        sepf = '\\phantom{-}' if f1_diff > 0 else ''
        print('\\data{' + f"{k}" + "} & " + f"{accuracy_sdt:0.2f}" + f"/{f1_sdt:0.2f} " \
              + f"& {f_max_ind+1}" \
              + " & " \
              + " \\cite{" \
              + cite + "} & " \
              + f"{sepa}{accuracy_diff:0.2f}/{sepf}{f1_diff:0.2f} \\\\")

 
def print_totoa_matrix(train_on_one_test_on_another_performance, col_name):
    cmap = plt.cm.get_cmap('RdYlBu')
    if col_name == 'f3': cmap = plt.cm.get_cmap('YlGn')
    # accuracy
    print("\\begin{tikzpicture}[]")
    print("  \\matrix[matrix of nodes,row sep=-\\pgflinewidth, column sep=-.1em,")
    print("nodes={{rectangle}},")
    print("column 1/.style={{anchor=east}},]{")
    _df = pd.pivot_table(train_on_one_test_on_another_performance, values=col_name, index='train_on', columns='test_on').round(2)
    for i,r in _df.round(2).iterrows():
        s = "\\data{\\small{" + process_tick_label(i).replace('_','\\_')  + '}} & '
        for e in list(r):
            s += "|[fill={{rgb,255:red,{};green,{};blue,{}}}, value={}]|&".format(int(255*cmap(e)[0]), int(255*cmap(e)[1]), int(255*cmap(e)[2]), (e))
        print(s[:-1],'\\\\')
    print("};")
    for i,e in enumerate(list(_df.index)):
        print("\\node[label={{[label distance=0.5cm,text depth=-1ex,rotate=45]left:\\data{{\\small{{{}}}}} }}] at ({},-2.4) {{}};".format(process_tick_label(e).replace("_","\\_"), -.7 + (.7*i), end=' & '))
    print("\\end{tikzpicture}\n\n")

