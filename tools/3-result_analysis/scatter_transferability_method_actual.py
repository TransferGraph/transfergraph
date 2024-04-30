import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.preprocessing import minmax_scale

from transfergraph.config import get_root_path_string
from transfergraph.dataset.task import TaskType


def scatter_plot_transferability_method_and_actual_performances(method, task_type, all_target_dataset):
    directory_experiments = f"{get_root_path_string()}/resources/experiments"
    base_path = f"{directory_experiments}/{task_type.value}/rank_final"
    actual_performances = pd.read_csv(f"{directory_experiments}/{task_type.value}/records.csv")

    actual_per_dataset = {target_dataset: pd.DataFrame() for target_dataset in all_target_dataset}
    transferability_per_dataset = {target_dataset: pd.DataFrame() for target_dataset in all_target_dataset}

    for target_dataset in os.listdir(base_path):
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

        if "normalize" not in method:
            # Normalize the scores using minmax_scale
            transferability_list = minmax_scale(transferability_list)

        actual_per_dataset[target_dataset] = actual_list
        transferability_per_dataset[target_dataset] = transferability_list

    plot_data(method, actual_per_dataset, transferability_per_dataset)

    pass


def plot_data(method, actual_performances, transferability_scores):
    plt.figure(figsize=(10, 6))
    for dataset, performances in actual_performances.items():
        plt.scatter(transferability_scores[dataset], performances, label=dataset)
    plt.title(f'Scatter Plot of Actual Performances vs Transferability Scores for {method}')
    plt.xlabel('Transferability Scores')
    plt.ylabel('Actual Performances')
    plt.legend()
    plt.grid(True)
    plt.show()


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


def main():
    parser = argparse.ArgumentParser(description='Plot scatter plot for a specific method.')
    parser.add_argument('--task_type', type=TaskType, required=True)
    parser.add_argument('--method', type=str, required=True, help='Method to analyze')
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)
    args = parser.parse_args()

    args.all_target_dataset = [s.strip() for s in args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []

    scatter_plot_transferability_method_and_actual_performances(args.method, args.task_type, args.all_target_dataset)


if __name__ == "__main__":
    main()
