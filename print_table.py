import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


from plotting import process_tick_label
import re

def command(w, com):
    return "\\" + com + "{" + w + "}"

def data(w):
    return command(w, "data")

def cite(w):
    return command(w, "cite")

def print_row(w_list):
    print(" & ".join(w_list), "\\\\")


def get_score(score, dataset_name):
    if score == "":
        return -1
    match = re.search(f"{dataset_name}: ([0-9]*(\.\d+)?)", score)
    if match: 
        return float(match.group(1))
    match = re.search(f"[a-zA-Z]+-[0-9]+: ([0-9]*(\.\d+)?)", score)
    if match:
        return -1
    match = re.search(f"all: ([0-9]*(\.\d+)?)", score)
    if match:
        return -1
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

        if accuracy_diff > 0 and f1_diff > 0:
            c = '\\;\\;---\\;\\;'
        elif accuracy_diff > 0:
            citation = f"{v['accuracy'][1]}"
            c = cite(citation)
        elif f1_diff > 0:
            citation = f"{v['f1'][1]}"
            c = cite(citation)
        elif v['accuracy'][1] == v['f1'][1]:
            citation = f"{v['accuracy'][1]}"
            c = cite(citation)
        else:
            citation = f"{v['accuracy'][1]}, {v['f1'][1]}"
            c = cite(citation)
            
        #sepa = '\\phantom{-}' if accuracy_diff > 0 else ''
        #sepf = '\\phantom{-}' if f1_diff > 0 else ''
        sepa = ''
        sepf = ''
        if accuracy_diff > 0:
            accuracy_diff = '\\;\\;---\\;\\;\\,'
        else:
            accuracy_diff = f'{accuracy_diff:0.2f}'
        if f1_diff > 0:
            f1_diff = '\\;\\;---'
        else:
            f1_diff = f'{f1_diff:0.2f}'

        print_row([data(k), 
                   f"{accuracy_sdt:0.2f}/{f1_sdt:0.2f}/{balanced_accuracy_sdt:0.2f} ", 
                   f"{max_ind+1}",
                   c,
                   f"{sepa}{accuracy_diff}/{sepf}{f1_diff}"
                   ])

def print_single_dataset_score_table_without_sota(sdt_df, max_depth, print_balanced_acc=True):
    tolerance = 0.025

    for row in sdt_df.to_dict(orient="records"):
        # sdt
        accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
        f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
        balanced_accuracies = [row[f'ba{i}'] for i in range(1, max_depth+1)]
        accuracy_sdt = max(accuracies)
        f1_sdt = max(f1s)
        # prefer shallower decision trees if accuracy isn't too different
        max_ind, accuracy_sdt, f1_sdt, balanced_accuracy_sdt = get_shallowest_good_results(0.025, accuracies, f1s, balanced_accuracies)

        
        if print_balanced_acc:
            perf = f"{accuracy_sdt:0.2f}/{f1_sdt:0.2f}/{balanced_accuracy_sdt:0.2f} "
        else:
            perf = f"{accuracy_sdt:0.2f}/{f1_sdt:0.2f} "
        print_row([data(process_tick_label(row['dataset'])), 
                    perf,
                    f"{max_ind+1}"
                    ])


def print_leave_one_out_table(oos_df, is_df, random_forest=True):
    max_depth = 5
    tolerance = 0.025

    for i in range(len(oos_df)):
        row = oos_df.iloc[i]
        is_row = is_df.iloc[i]
        k = row['left_out']
        name = process_tick_label(k)

        if random_forest:
            accuracy = row[f'a_rf']
            f1 = row[f'f_rf']
            balanced_accuracy = row[f'ba_rf']

            is_accuracy = is_row[f'a_rf']
            is_f1 = is_row[f'f_rf']
            is_balanced_accuracy = is_row[f'ba_rf']
            print_row([data(name), 
                f"{is_accuracy:0.2f}/{is_f1:0.2f}/{is_balanced_accuracy:0.2f}",
                f"{accuracy:0.2f}/{f1:0.2f}/{balanced_accuracy:0.2f}"])
        else:
            accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
            f1s = [row[f'f{i}'] for i in range(1, max_depth+1)]
            balanced_accuracy = [row[f'ba{i}'] for i in range(1, max_depth+1)]
            max_ind, accuracy, f1, balanced_accuracy = get_shallowest_good_results(0.025, accuracies, f1s, balanced_accuracies)
            print_row([data(name), f"{accuracy:0.2f}/{f1:0.2f}/{balanced_accuracy:0.2f}", f"{max_ind+1}"])



 
def print_totoa_matrix(train_on_one_test_on_another_performance, col_name, start_x = -0.7, start_y = -2.4, label_offset=0.6):
    colors = ["darkred", "lightyellow", "mediumblue"]
    cmap = LinearSegmentedColormap.from_list("ryb", colors)
    if col_name == 'f': 
        colors = ["lightyellow", "mediumblue"]
        cmap = LinearSegmentedColormap.from_list("yb", colors)
    # accuracy
    print("\\begin{tikzpicture}[]")
    print("  \\matrix[matrix of nodes,row sep=-\\pgflinewidth, column sep=-.39em,")
    print("nodes={{rectangle}},")
    print("column 1/.style={{anchor=east}},]{")
    train_on_one_test_on_another_performance['train_on2'] = train_on_one_test_on_another_performance['train_on'].map(lambda x: x[:-8] if 'one_hot' in x else x)
    train_on_one_test_on_another_performance['test_on2'] = train_on_one_test_on_another_performance['test_on'].map(lambda x: x[:-8] if 'one_hot' in x else x)
    train_on_one_test_on_another_performance['train_on_year'] = train_on_one_test_on_another_performance['train_on2'].str[-4:]
    train_on_one_test_on_another_performance['test_on_year'] = train_on_one_test_on_another_performance['test_on2'].str[-4:]

    train_on_one_test_on_another_performance = train_on_one_test_on_another_performance.sort_values(by=['train_on_year', 'train_on2', 'test_on_year', 'test_on2'], ascending=[False, True, False, True])

    #_df = pd.pivot_table(train_on_one_test_on_another_performance, values=col_name, index='train_on', columns='test_on').round(2)
    labels = []
    cur_train_on = ''
    first_row = True
    for row in train_on_one_test_on_another_performance.to_dict(orient='records'):
        i = row['train_on']
        if cur_train_on != i:
            labels.append(i)
            cur_train_on = i
            if not first_row:
                print(s[:-1],'\\\\')
            first_row = False
            s = "\\data{\\small{" + process_tick_label(i).replace('_','\\_')  + '}} & '
        e = round(row[col_name], 2)
        s += "|[fill={{rgb,255:red,{};green,{};blue,{}}}, value={}]|&".format(int(255*cmap(e)[0]), int(255*cmap(e)[1]), int(255*cmap(e)[2]), (e))
    print(s[:-1],'\\\\')
    print("};")
    for i,e in enumerate(labels):
        t_lab = process_tick_label(e)
        print("\\node[label={[label distance=0.5cm,text depth=-1ex,rotate=45]left:", data(command(t_lab, 'small')), "}] at", f"({start_x + (label_offset*i)},{start_y})", "{};")
    print("\\end{tikzpicture}\n\n")


def print_intratype_test(intraclass_df, max_depth=4):
    for name, row in intraclass_df.to_dict(orient='index').items():
        accuracies = [row[f'a{i}'] for i in range(1, max_depth+1)]
        balanced_accuracies = [row[f'ba{i}'] for i in range(1, max_depth+1)]
        accuracy_sdt = max(accuracies)
        # prefer shallower decision trees if accuracy isn't too different
        max_ind, accuracy_sdt, _, balanced_accuracy_sdt = get_shallowest_good_results(0.025, accuracies, [-1]*len(accuracies), balanced_accuracies)
        print_row([data(name), f'{accuracy_sdt:.2f}/{balanced_accuracy_sdt:.2f}', f'{max_ind + 1}', f'{row["n_datasets"]}'])


