import argparse
import os

import numpy as np
import pandas as pd
from pandas import Series

from transfergraph.config import get_directory_experiments
from transfergraph.dataset.task import TaskType
from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric
from transfergraph.transferability_estimation.result_analysis.result_analysis_utils import compute_correlation


def compute_and_save_correlation_by_rank_files(task_type, all_metrics, all_method, all_target_datasets, peft_method, finetuning_ratio):
    directory_experiments = get_directory_experiments(task_type)
    base_path = f"{directory_experiments}/rank_final"
    actual_performances = pd.read_csv(f"{directory_experiments}/{task_type.value}/records.csv")

    results = {metric: pd.DataFrame() for metric in all_metrics}

    for target_dataset in os.listdir(base_path):
        if len(all_target_datasets) != 0 and target_dataset not in all_target_datasets:
            continue

        target_path = os.path.join(base_path, target_dataset)
        target_dataset = determine_target_dataset(target_dataset)

        # Filter actual performances based on the target_dataset
        actual_performances_target = actual_performances[actual_performances['finetuned_dataset'] == target_dataset]

        # Select the best record for each model based on eval_accuracy
        idxs_max_accuracy = actual_performances_target.groupby('model')['eval_accuracy'].idxmax()
        actual_performances_target = actual_performances_target.loc[idxs_max_accuracy]

        # Filter by fine-tuning method.
        actual_performances_target = filter_actual_performance_by_peft_method(actual_performances_target, peft_method)

        # We also want a fake method, which randomly scores.
        add_random_transferability_method_results(actual_performances_target, all_metrics, results, target_dataset)

        for method in os.listdir(target_path):
            if len(all_method) != 0 and method not in all_method:
                continue

            file_suffix = f"{peft_method}_" if peft_method else ""
            filename = f"results_{file_suffix}{finetuning_ratio}_128_0.csv"
            method_path = os.path.join(target_path, method, filename)
            if os.path.isfile(method_path):
                method_path_final = method_path
            elif len(os.listdir(os.path.join(target_path, method))) == 1 and os.path.isfile(
                    os.path.join(target_path, method, 'results.csv')
                    ):
                method_path_final = os.path.join(target_path, method, 'results.csv')
            else:
                continue

            transferability_scores = pd.read_csv(method_path_final)
            merged_df = pd.merge(actual_performances_target, transferability_scores, on='model', how='inner')
            actual_list = merged_df['eval_accuracy'].tolist()
            transferability_list = replace_all_infinite_value(merged_df['score']).tolist()

            for metric in all_metrics:
                correlation = compute_correlation(actual_list, transferability_list, metric)
                results[metric].loc[method, target_dataset] = correlation

    # Output results
    for metric, df in results.items():
        if len(df) == 0:
            continue

        # Remove rows with any NaN values across columns, I want the method to have all datasets.
        df = df.dropna()

        # Calculate the mean correlation score for each method
        df['average'] = df.mean(axis=1)

        # Sort the DataFrame based on 'Average_Score' in descending order
        sorted_df = df.sort_values(by='average', ascending=False)

        if not os.path.exists(f"{directory_experiments}/{task_type.value}/result_analysis"):
            os.makedirs(f"{directory_experiments}/{task_type.value}/result_analysis")

        sorted_df.to_csv(f'{directory_experiments}/{task_type.value}/result_analysis/{metric.value}_correlation.csv')


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


def add_random_transferability_method_results(actual_performances_target, all_metrics, results, target_dataset):
    # Randomize the eval_accuracy for the 'random' method
    random_transferability_metric = np.random.permutation(actual_performances_target['eval_accuracy'].values)
    for metric in all_metrics:
        if metric == TransferabilityCorrelationMetric.RELATIVE_TOP_1:
            # Take the expected value as random pick, so we don't get lucky.
            random_correlation = actual_performances_target['eval_accuracy'].mean() / actual_performances_target['eval_accuracy'].max()
        elif metric == TransferabilityCorrelationMetric.RANDOM_RELATIVE_TOP_1_ERROR \
                or metric == TransferabilityCorrelationMetric.RANDOM_RELATIVE_TOP_3_ERROR \
                or metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_1 \
                or metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_3 \
                or metric == TransferabilityCorrelationMetric.RANDOM_ABSOLUTE_TOP_1_ERROR \
                or metric == TransferabilityCorrelationMetric.RANDOM_ABSOLUTE_TOP_3_ERROR:
            # This is already random corrected.
            continue
        else:
            # Compute the correlation for the 'random' method
            random_correlation = compute_correlation(
                actual_performances_target['eval_accuracy'].tolist(),
                random_transferability_metric.tolist(),
                metric
            )
        results[metric].loc['random', target_dataset] = random_correlation


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
    parser = argparse.ArgumentParser(description='Process transferability experiments.')
    parser.add_argument('--task_type', type=TaskType, required=True)
    parser.add_argument(
        '--all_metric',
        type=TransferabilityCorrelationMetric,
        required=False,
        nargs='+',
        default=[metric for metric in TransferabilityCorrelationMetric]
    )
    parser.add_argument('--all_method', type=str, required=False, default=None)
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)
    parser.add_argument('--peft_method', type=str, choices=[None, 'lora'], default=None, required=False)
    parser.add_argument('--finetuning_ratio', required=False, type=float, default=1.0)

    args = parser.parse_args()

    args.all_method = [s.strip() for s in args.all_method.split(",")] if args.all_method is not None else []
    args.all_target_dataset = [s.strip() for s in args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []

    compute_and_save_correlation_by_rank_files(
        args.task_type,
        args.all_metric,
        args.all_method,
        args.all_target_dataset,
        args.peft_method,
        args.finetuning_ratio
    )


if __name__ == "__main__":
    main()
