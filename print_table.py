import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plotting import process_tick_label
import re

from plotting import process_tick_label

def data(w):
    return "\\data{" + w + "}"

def cite(w):
    return "\\cite{" + w + "}"

def print_row(w_list):
    print(" & ".join(w_list), "\\\\")


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


def get_shallowest_good_results(tolerance, accuracies, f1s, balanced_accuracies):
    accuracy_sdt = max(accuracies)
    f1_sdt = max(f1s)
    for i, (acc, f) in enumerate(zip(accuracies, f1s)):
        if (accuracy_sdt - acc <= tolerance) and (f1_sdt - f <= tolerance):
            max_ind = i
            accuracy_sdt = acc
            f1_sdt = f
            return i, accuracy_sdt, f1_sdt, balanced_accuracies[i]


def print_dataset_table(dataset_df, benchmark_only):
    for row in dataset_df.to_dict(orient="records"):
        if not row['analyzed?']:
            continue
        if benchmark_only and row['benchmark?'] != '1':
            continue
        if not benchmark_only and row['benchmark?'] == '1':
            continue
        name = data(row['dataset name'])
            
        num_humans = int(row['# humans we have']) if row['# humans we have']!="" else '-'
        num_bots = int(row['# bots we have']) if row['# bots we have']!="" else '-'
        
        if benchmark_only:
            desc_list = row['description'].split("; ")
            if len(desc_list) == 1:
                desc = row['description']
            else:
                desc_list = [data(d) for d in desc_list]
                desc = ", ".join(desc_list)
        else:
            desc = row['description']
        print_row([
            name,
            cite(row['bibtex_id']),
            str(num_humans),
            str(num_bots)
            ])


def print_single_dataset_score_table(sota_dict, sdt_df):
    max_depth = 5
    tolerance = 0.025

    for k,v in sota_dict.items():
        # sota
        accuracy_sota = float(v['accuracy'][0])
        f1_sota = float(v['f1'][0])
        # sdt
        row = sdt_df[sdt_df['name'] == k].to_dict(orient="records")[0]
        accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
        f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
        balanced_accuracies = [row[f'ba{i}'] for i in range(1, max_depth+1)]
        accuracy_sdt = max(accuracies)
        f1_sdt = max(f1s)
        # prefer shallower decision trees if accuracy isn't too different
        max_ind, accuracy_sdt, f1_sdt, balanced_accuracy_sdt = get_shallowest_good_results(0.025, accuracies, f1s, balanced_accuracies)

        accuracy_diff = accuracy_sdt - accuracy_sota
        f1_diff = f1_sdt - f1_sota
        if v['accuracy'][1] == v['f1'][1]:
            citation = f"{v['accuracy'][1]}"
        else:
            citation = f"{v['accuracy'][1]}, {v['f1'][1]}"
        sepa = '\\phantom{-}' if accuracy_diff > 0 else ''
        sepf = '\\phantom{-}' if f1_diff > 0 else ''
        print_row([data(k), 
                   f"{accuracy_sdt:0.2f}/{f1_sdt:0.2f}/{balanced_accuracy_sdt:0.2f} ", 
                   f"{max_ind+1}",
                   cite(citation),
                   f"{sepa}{accuracy_diff:0.2f}/{sepf}{f1_diff:0.2f}"
                   ])


def print_leave_one_out_table(df, random_forest=True):
    max_depth = 5
    tolerance = 0.025

    for row in df.to_dict(orient='records'):
        k = row['left_out']
        name = process_tick_label(k)

        if random_forest:
            accuracy = row[f'a_rf']
            f1 = row[f'f_rf']
            balanced_accuracy = [row[f'ba_rf'] for i in range(1, max_depth+1)]
            print_row([name, f"{accuracy:0.2f}/{f1:0.2f}/f{balanced_accuracy:0.2f}"])
        else:
            accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
            f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
            balanced_accuracy = [row[f'ba{i}'] for i in range(1, max_depth+1)]
            max_ind, accuracy, f1, balanced_accuracy = get_shallowest_good_results(0.025, accuracies, f1s, balanced_accuracies)
            print_row([name, f"{accuracy:0.2f}/{f1:0.2f}/f{balanced_accuracy:0.2f}", f"{max_ind+1}"])



 
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

