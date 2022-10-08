import numpy as np
import matplotlib.pyplot as plt

def process_tick_label(tick_label):
    ret = tick_label
    if tick_label.endswith('_one_hot'):
        ret = tick_label[:-8]
    if tick_label.endswith("_df"):
        ret = tick_label[:-3]
    ret = ret.replace('_', '-')
    print(ret)
    if ret == 'botometer-feedback-2019': return 'feedback-2019'
    if ret == 'cresci-rtbust-2019': return 'rtbust-2019'
    if ret == 'cresci-stock-2018': return 'stock-2018'
    return ret

def heatmap_train_on_one_test_on_another(df, metric_name, depth):    
    tick_labels = df['train_on'].unique()

    d = len(tick_labels)
    data = [[] for i in range(d)]
    for i in range(d):
        for j in range(d):
            metric = df[(df['train_on'] == tick_labels[i]) & (df['test_on'] == tick_labels[j])]
            if metric_name == "accuracy":
                metric = metric[f'a{depth}'].values[0]
            if metric_name == "f1":
                metric = metric[f'f{depth}'].values[0]
            data[i].append(metric)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(data, cmap="RdYlBu", vmin=0, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_xticklabels([process_tick_label(t) for t in tick_labels])
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_yticklabels([process_tick_label(t) for t in tick_labels])
    #ax.spines.set_visible(False)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    d = len(tick_labels)
    for i in range(d):
        for j in range(d):
            metric = data[i][j]
            text = ax.text(j, i, f"{metric:.2f}",
                           ha="center", va="center")
            

    
    #ax.set_title(f"Train on one test on another: {metric_name}, depth={depth}")
    fig.tight_layout()
    plt.savefig(f"trainononetestonanother_{metric_name}.svg")
    plt.show()
    return pd.DataFrame(data, index=tick_labels, columns=tick_labels)

def hist_train_on_one_test_on_another(performance_df):
    fig, axes = plt.subplots(4, 1, figsize=(8,20))
    max_depth = 5

    accs = []
    fs = []
    precs = []
    recs = []

    for ind, row in performance_df.iterrows():
        metric_a = [row[f'a{i}'] for i in range(1,max_depth+1)]
        accs.append(max(metric_a))

        metric_f = [row[f'f{i}'] for i in range(1,max_depth+1)]
        fs.append(max(metric_f))


        metric_p = [row[f'p{i}'] for i in range(1,max_depth+1)]
        precs.append(max(metric_p))


        metric_r = [row[f'r{i}'] for i in range(1,max_depth+1)]
        recs.append(max(metric_r))


    axes[0].hist(accs)
    axes[1].hist(fs)
    axes[2].hist(precs)
    axes[3].hist(recs)


    axes[0].set_xlabel("Accuracy")
    axes[1].set_xlabel("F1")
    axes[2].set_xlabel("Precision")
    axes[3].set_xlabel("Recall")

    axes[0].set_ylabel("Number of models")
    axes[1].set_ylabel("Number of models")
    axes[2].set_ylabel("Number of models")
    axes[3].set_ylabel("Number of models")

    #axes[0].legend(loc='lower left')
    fig.suptitle("Performance for training on one dataset and testing on another")
    fig.tight_layout()
    plt.show()
    
def plot_botometer_combined_dataset_performance():
    fig, axes = plt.subplots(4, 1, figsize=(8,20))
    max_depth = 5

    for ind, row in botometer_leave_one_out_scores.iterrows():
        metric_a = [row[f'a{i}'] for i in range(1,max_depth+1)]
        axes[0].plot(range(1,6), metric_a, label=f"Bot dataset: {row['dataset']}", marker="o")

        metric_f = [row[f'f{i}'] for i in range(1,max_depth+1)]
        axes[1].plot(range(1,6), metric_f, label=f"{row['dataset']}", marker="o")

        metric_p = [row[f'p{i}'] for i in range(1,max_depth+1)]
        axes[2].plot(range(1,6), metric_p, label=f"{row['dataset']}", marker="o")

        metric_r = [row[f'r{i}'] for i in range(1,max_depth+1)]
        axes[3].plot(range(1,6), metric_r, label=f"{row['dataset']}", marker="o")


        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("F1")
        axes[2].set_ylabel("Precision")
        axes[3].set_ylabel("Recall")

    axes[0].legend(loc='lower left')
    fig.text(0.5, 0.0, 'Model complexity (tree depth)', ha='center')
    fig.suptitle("Botometer combined dataset performance")
    fig.tight_layout()
    plt.show()

def plot_leave_one_out_scores():
    fig, axes = plt.subplots(4, 1, figsize=(8,20))
    max_depth = 5

    for ind, row in leave_one_out_scores.iterrows():
        metric_a = [row[f'a{i}'] for i in range(1,max_depth+1)]
        axes[0].plot(range(1,6), metric_a, label=f"Left out: {row['left_out']}", marker="o")

        metric_f = [row[f'f{i}'] for i in range(1,max_depth+1)]
        axes[1].plot(range(1,6), metric_f, label=f"Left out: {row['left_out']}", marker="o")

        metric_p = [row[f'p{i}'] for i in range(1,max_depth+1)]
        axes[2].plot(range(1,6), metric_p, label=f"Left out: {row['left_out']}", marker="o")

        metric_r = [row[f'r{i}'] for i in range(1,max_depth+1)]
        axes[3].plot(range(1,6), metric_r, label=f"Left out: {row['left_out']}", marker="o")


        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("F1")
        axes[2].set_ylabel("Precision")
        axes[3].set_ylabel("Recall")

    axes[0].legend(loc='lower left')
    fig.text(0.5, 0.0, 'Model complexity (tree depth)', ha='center')
    fig.suptitle("Leave-one-out-dataset performance")
    fig.tight_layout()
    plt.show()

 
def plot_loo_performance():
    fig, axes = plt.subplots(4, 1, figsize=(8,20))
    max_depth = 5

    for ind, row in botometer_leave_one_out_scores.iterrows():
        metric_a = [row[f'a{i}'] for i in range(1,max_depth+1)]
        axes[0].plot(range(1,6), metric_a, label=f"Left out: {row['left_out']}", marker="o")

        metric_f = [row[f'f{i}'] for i in range(1,max_depth+1)]
        axes[1].plot(range(1,6), metric_f, label=f"Left out: {row['left_out']}", marker="o")

        metric_p = [row[f'p{i}'] for i in range(1,max_depth+1)]
        axes[2].plot(range(1,6), metric_p, label=f"Left out: {row['left_out']}", marker="o")

        metric_r = [row[f'r{i}'] for i in range(1,max_depth+1)]
        axes[3].plot(range(1,6), metric_r, label=f"Left out: {row['left_out']}", marker="o")


        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("F1")
        axes[2].set_ylabel("Precision")
        axes[3].set_ylabel("Recall")

    axes[0].legend(loc='lower left')
    fig.text(0.5, 0.0, 'Model complexity (tree depth)', ha='center')
    fig.suptitle("Botometer: Leave-one-out-dataset performance")
    fig.tight_layout()
    plt.show()
