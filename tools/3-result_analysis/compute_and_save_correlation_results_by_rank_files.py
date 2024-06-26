import argparse
import math
import os
import re

import numpy as np
import pandas as pd
from pandas import Series
from scipy.stats import percentileofscore

from transfergraph.config import get_directory_experiments
from transfergraph.dataset.task import TaskType
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod
from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric
from transfergraph.transferability_estimation.result_analysis.result_analysis_utils import compute_correlation


def compute_and_save_correlation_by_rank_files(
        task_type,
        all_metrics,
        all_method,
        all_baseline,
        all_target_datasets,
        peft_method,
        finetuning_ratio
):
    directory_experiments = get_directory_experiments(task_type)
    base_path = f"{directory_experiments}/rank_final"
    actual_performances = pd.read_csv(f"{directory_experiments}/records.csv")
    filename_baseline = f"{directory_experiments}/transferability_score_records.csv"

    baseline_scores = pd.read_csv(filename_baseline, index_col=0)

    results = {metric: pd.DataFrame() for metric in all_metrics}
    results_by_ratio = {}

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

        all_model_by_ratio = {}

        for method in os.listdir(target_path):
            if len(all_method) != 0 and method not in all_method:
                continue

            method_base_name = re.sub(r'_model_ratio_\d+\.\d+', '', method)

            ratio = determine_model_ratio(method)

            file_suffix = f"{peft_method}_" if peft_method else ""
            filename = f"results_{file_suffix}{finetuning_ratio}_128_0.csv"
            method_path = os.path.join(target_path, method, filename)

            transferability_scores = pd.read_csv(method_path)
            merged_transferability_df = pd.merge(actual_performances_target, transferability_scores, on='model', how='inner')
            actual_list = merged_transferability_df['eval_accuracy'].tolist()
            transferability_list = replace_all_infinite_value(merged_transferability_df['score']).tolist()
            all_model_by_ratio[ratio] = merged_transferability_df[["model"]]

            for metric in all_metrics:
                correlation = compute_correlation(actual_list, transferability_list, metric)

                if ratio not in results_by_ratio:
                    results_ratio_new = {metric: pd.DataFrame() for metric in all_metrics}
                    results_by_ratio[ratio] = results_ratio_new

                add_random_transferability_method_results(merged_transferability_df, all_metrics, results_by_ratio[ratio], target_dataset)
                results_by_ratio[ratio][metric].loc[method_base_name, target_dataset] = correlation

                if "model_ratio" not in method:
                    results[metric].loc[method_base_name, target_dataset] = correlation

        for ratio in all_model_by_ratio:
            for baseline in all_baseline:
                baseline_scores_baseline = baseline_scores[baseline_scores['metric'] == baseline.__str__()]
                baseline_scores_baseline_target = baseline_scores_baseline[baseline_scores_baseline['target_dataset'] == target_dataset]

                if len(baseline_scores_baseline_target) == 0:
                    continue

                merged_baseline_df = pd.merge(actual_performances_target, baseline_scores_baseline_target, on='model', how='inner')

                # Merge it with one of our own method's results, to ensure we have the same models.
                merged_transferability_df_by_ratio = all_model_by_ratio[ratio]
                merged_baseline_df = pd.merge(merged_transferability_df_by_ratio, merged_baseline_df, on='model', how='inner')

                actual_list = merged_baseline_df['eval_accuracy'].tolist()
                baseline_list = replace_all_infinite_value(merged_baseline_df['score']).tolist()

                for metric in all_metrics:
                    correlation = compute_correlation(actual_list, baseline_list, metric)
                    results_by_ratio[ratio][metric].loc[baseline.value, target_dataset] = correlation

                    if ratio == 1.0:
                        results[metric].loc[baseline.value, target_dataset] = correlation

    # Output results
    for metric, df in results.items():
        if len(df) == 0:
            continue

        output_correlation_to_csv(df, directory_experiments, metric)

    # Output results
    for ratio, df_by_ratio in results_by_ratio.items():
        for metric, df in df_by_ratio.items():
            if len(df) == 0:
                continue

            output_correlation_to_csv(df, directory_experiments, metric, ratio)


def output_correlation_to_csv(df, directory_experiments, metric, model_ratio=None):
    # Remove rows with any NaN values across columns, I want the method to have all datasets.
    df = df.dropna()

    # Separate the weights (rand_abs_max row) and the scores
    weights = df.loc['rand_abs_max']
    df = df.drop('rand_abs_max')

    # Compute the weighted average for each method
    weighted_averages = df.apply(lambda x: (x * weights).sum() / weights.sum(), axis=1)
    df['average'] = weighted_averages

    # Sort the DataFrame based on 'Average_Score' in descending order
    sorted_df = df.sort_values(by='average', ascending=False)

    if model_ratio is None:
        directory_correlation = f"{directory_experiments}/result_analysis"
    else:
        directory_correlation = f"{directory_experiments}/result_analysis/model_ratio_{model_ratio}"

    if not os.path.exists(directory_correlation):
        os.makedirs(directory_correlation)

    sorted_df.to_csv(f'{directory_correlation}/{metric.value}_correlation.csv')


def determine_model_ratio(method):
    if 'model_ratio' in method:
        match = re.search(r'model_ratio_([\d.]+)', method)
        if match:
            ratio = float(match.group(1))
        else:
            ratio = 1
    else:
        ratio = 1
    return ratio


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
    for metric in all_metrics:
        if metric == TransferabilityCorrelationMetric.RELATIVE_TOP_1 or \
                metric == TransferabilityCorrelationMetric.RELATIVE_TOP_3 or \
                metric == TransferabilityCorrelationMetric.RELATIVE_TOP_5:
            # Take the expected value as random pick, so we don't get lucky.
            random_correlation = actual_performances_target['eval_accuracy'].mean() / actual_performances_target['eval_accuracy'].max()
        elif metric == TransferabilityCorrelationMetric.TOP_1 or \
                metric == TransferabilityCorrelationMetric.TOP_3 or \
                metric == TransferabilityCorrelationMetric.TOP_5:
            random_correlation = actual_performances_target['eval_accuracy'].mean()
        elif metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_1 or \
                metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_3 or \
                metric == TransferabilityCorrelationMetric.PERCENTILE_TOP_5:
            random_accuracy = actual_performances_target['eval_accuracy'].mean()
            random_correlation = percentileofscore(actual_performances_target['eval_accuracy'].tolist(), random_accuracy)
        else:
            random_correlation = None

        if random_correlation is not None:
            results[metric].loc['random', target_dataset] = random_correlation

        rabs = actual_performances_target['eval_accuracy'].max() - actual_performances_target['eval_accuracy'].mean()

        if math.isnan(rabs):
            raise ValueError("bla")
        results[metric].loc['rand_abs_max', target_dataset] = actual_performances_target['eval_accuracy'].max() - actual_performances_target['eval_accuracy'].mean()

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
    parser.add_argument('--all_metric', type=str, required=False, default=None)
    parser.add_argument('--all_method', type=str, required=False, default=None)
    parser.add_argument('--all_baseline', type=str, required=False, default=None)
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)
    parser.add_argument('--peft_method', type=str, choices=[None, 'lora'], default=None, required=False)
    parser.add_argument('--finetuning_ratio', required=False, type=float, default=1.0)

    args = parser.parse_args()

    args.all_metric = [TransferabilityCorrelationMetric(s.strip) for s in args.all_metric.split(",")] if args.all_metric is not None else [
        baseline for baseline in TransferabilityCorrelationMetric]
    args.all_baseline = [TransferabilityMethod(s.strip()) for s in
                         args.all_baseline.split(",")] if args.all_baseline is not None else TransferabilityMethod
    args.all_method = [s.strip() for s in args.all_method.split(",")] if args.all_method is not None else []
    args.all_target_dataset = [s.strip() for s in args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []

    compute_and_save_correlation_by_rank_files(
        args.task_type,
        args.all_metric,
        args.all_method,
        args.all_baseline,
        args.all_target_dataset,
        args.peft_method,
        args.finetuning_ratio
    )


if __name__ == "__main__":
    main()
