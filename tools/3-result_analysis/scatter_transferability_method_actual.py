import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.preprocessing import minmax_scale
from scipy.stats import pearsonr

from transfergraph.config import get_root_path_string
from transfergraph.dataset.task import TaskType
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod
import seaborn as sns

sns.set_theme()
sns.set_context("talk", font_scale=1)
sns.set_style("white")

def ucfirst(s):
    if not s:
        return s
    return s[0].upper() + s[1:]

def shorten_dataset_name(dataset_name):
    return ucfirst(dataset_name.replace('rotten_tomatoes', 'Rotten').replace('tweet_eval/irony', 'Tw/Irony')\
                       .replace('tweet_eval/sentiment', 'Tw/Senti').replace('tweet_eval/offensive', 'Tw/Offen')\
                       .replace('glue/cola', 'Glue/C').replace('glue/sst2', 'Glue/S')\
                       .replace('tweet_eval/hate', 'Tw/Hate').replace('tweet_eval/emotion', 'Tw/Emoti')\
                       .replace('smallnorb_label_elevation', 'SmallN/El') \
                       .replace('smallnorb_label_azimuth', 'SmallN/Az') \
                       .replace('diabetic_retinopathy_detection', 'Diabetic') \
                       .replace('stanfordcars', 'Stanford') \
                       .replace('average', ''))

def scatter_plot_transferability_method_and_actual_performances(method, task_type, all_target_dataset, all_baseline):
    directory_experiments = f"{get_root_path_string()}/resources/experiments/{task_type.value}"
    base_path = f"{directory_experiments}/rank_final"
    actual_performances = pd.read_csv(f"{directory_experiments}/records.csv")
    filename_baseline = f"{directory_experiments}/transferability_score_records.csv"
    baseline_scores = pd.read_csv(filename_baseline, index_col=0)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, target_dataset in enumerate(os.listdir(base_path)):
        if len(all_target_dataset) != 0 and target_dataset not in all_target_dataset:
            continue

        target_path = os.path.join(base_path, target_dataset)
        target_dataset = determine_target_dataset(target_dataset)

        # Filter actual performances based on the target_dataset
        actual_performances_target = actual_performances[actual_performances['finetuned_dataset'] == target_dataset]

        # Select the best record for each model based on eval_accuracy
        idxs_max_accuracy = actual_performances_target.groupby('model')['eval_accuracy'].idxmax()
        actual_performances_target = actual_performances_target.loc[idxs_max_accuracy]

        # Filter by fine-tuning method.
        actual_performances_target = filter_actual_performance_by_peft_method(actual_performances_target, None)

        filename = f"results_1.0_128_0.csv"
        method_path = os.path.join(target_path, method, filename)
        if os.path.isfile(method_path):
            method_path_final = method_path
        elif len(os.listdir(os.path.join(target_path, method))) == 1 and os.path.isfile(
                os.path.join(target_path, method, 'results.csv')
        ):
            method_path_final = os.path.join(target_path, method, 'results.csv')
        else:
            raise Exception("Method rank_final result file not found.")

        transferability_scores = pd.read_csv(method_path_final)

        merged_df = pd.merge(actual_performances_target, transferability_scores, on='model', how='inner')
        actual_list = merged_df['eval_accuracy'].tolist()
        transferability_list = replace_all_infinite_value(merged_df['score']).tolist()

        transferability_list = minmax_scale(transferability_list)

        plot_data(
            axes[i],
            method,
            target_dataset,
            actual_list,
            transferability_list,
            baseline_scores,
            actual_performances_target,
            all_baseline
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = f'{directory_experiments}/result_analysis/plots/scatter.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_data(ax, method, target_dataset, actual_list, transferability_list, baseline_scores, actual_performances_target, all_baseline):
    method_short = shorten_method_name(method)

    # Retrieve colors from the seaborn palette
    palette = sns.color_palette("Set2")
    scatter_color = palette[0]
    baseline_colors = palette[1:1 + len(all_baseline)]

    ax.scatter(transferability_list, actual_list, label=None, color=scatter_color)
    corr, _ = pearsonr(transferability_list, actual_list)

    ax.plot(
        np.unique(transferability_list), np.poly1d(np.polyfit(transferability_list, actual_list, 1))(np.unique(transferability_list)),
        linestyle='--', label=f'{method_short} (corr={corr:.2f})', color=scatter_color
    )

    for i, baseline in enumerate(all_baseline):
        baseline_scores_baseline = baseline_scores[baseline_scores['metric'] == baseline.__str__()]
        baseline_scores_baseline_target = baseline_scores_baseline[baseline_scores_baseline['target_dataset'] == target_dataset]

        if len(baseline_scores_baseline_target) == 0:
            continue

        merged_baseline_df = pd.merge(actual_performances_target, baseline_scores_baseline_target, on='model', how='inner')
        baseline_actual_list = merged_baseline_df['eval_accuracy'].tolist()
        baseline_list = replace_all_infinite_value(merged_baseline_df['score']).tolist()
        baseline_list = minmax_scale(baseline_list)

        ax.scatter(baseline_list, baseline_actual_list, label=None, color=baseline_colors[i % len(baseline_colors)])
        corr, _ = pearsonr(baseline_list, baseline_actual_list)
        ax.plot(
            np.unique(baseline_list), np.poly1d(np.polyfit(baseline_list, baseline_actual_list, 1))(np.unique(baseline_list)),
            linestyle='--', label=f'{baseline.value} (corr={corr:.2f})', color=baseline_colors[i % len(baseline_colors)]
        )

    ax.tick_params(axis='y')  # Set the y-axis labels
    ax.tick_params(axis='x')  # Set the x-axis labels
    ax.set_title(shorten_dataset_name(f'{target_dataset}'))
    ax.set_xlabel('Transferability Scores')
    ax.set_ylabel('Actual Performances')
    ax.legend(loc='lower right')

def determine_target_dataset(target_dataset):
    if "glue" in target_dataset:
        target_dataset = "glue/" + target_dataset.split("glue_")[1]
    elif "tweet_eval" in target_dataset:
        target_dataset = "tweet_eval/" + target_dataset.split("tweet_eval_")[1]
    return target_dataset


def filter_actual_performance_by_peft_method(actual_performances_target, peft_method):
    # Filter by fine-tuning method
    if 'peft_method' in actual_performances_target.columns:
        if peft_method is not None:
            actual_performances_target = actual_performances_target[actual_performances_target['peft_method'] == peft_method]
        else:
            actual_performances_target = actual_performances_target[pd.isna(actual_performances_target['peft_method'])]
    return actual_performances_target


def replace_all_infinite_value(series: Series):
    # Find the finite max and min values
    max_val = series[series != np.inf].max()
    min_val = series[series != -np.inf].min()

    # Replace inf and -inf with twice the max and min finite values respectively
    series.replace(to_replace=np.inf, value=2 * max_val, inplace=True)
    series.replace(to_replace=-np.inf, value=2 * min_val, inplace=True)
    series.replace(to_replace=np.nan, value=2 * min_val, inplace=True)

    return series


def shorten_method_name(method_name):
    if '_' in method_name:
        # Add TG_ prefix
        prefix = "TG:"
        parts = method_name.split('_')
        # Map specific terms
        parts = [part.replace("node2vec", "N2V").replace("node2vec+", "N2V+")
                 .replace("SAGEConv", "GraphSAGE").replace("GATConv", "GAT")
                 .replace("homoGCNConv", "GCN").replace("normalize", "").replace("homo", "") for part in parts]
        parts = [part.replace("all", "").replace("without", "").replace("transfer", "") for part in parts]
        # Join parts with underscores, remove redundant underscores, uppercase first part
        result = prefix + parts[0].upper() + '_' + '_'.join(parts[1:])
        result = result.strip('_')

        return result.replace('_', ',').replace(',,', ',').replace(',', ':')
    return method_name


def main():
    parser = argparse.ArgumentParser(description='Plot scatter plot for a specific method.')
    parser.add_argument('--task_type', type=TaskType, required=True)
    parser.add_argument('--method', type=str, required=True, help='Method to analyze')
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)
    parser.add_argument('--all_baseline', type=str, required=False, default=None)
    args = parser.parse_args()

    args.all_target_dataset = [s.strip() for s in args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []
    args.all_baseline = [TransferabilityMethod(s.strip()) for s in
                         args.all_baseline.split(",")] if args.all_baseline is not None else TransferabilityMethod
    scatter_plot_transferability_method_and_actual_performances(args.method, args.task_type, args.all_target_dataset, args.all_baseline)


if __name__ == "__main__":
    main()
