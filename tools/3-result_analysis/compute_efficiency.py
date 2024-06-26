import argparse
import os
import os.path
import re
import pandas as pd

from transfergraph.config import get_root_path_string
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod

def shorten_method_name(method_name):
    if "TransferabilityMethod" in method_name:
        return TransferabilityMethod[method_name.replace("TransferabilityMethod.", "")].value
    elif '_' in method_name:
        prefix = "TG:"
        parts = method_name.split('_')
        parts = [part.replace("node2vec", "N2V").replace("node2vec+", "N2V+")
                 .replace("SAGEConv", "GraphSAGE").replace("GATConv", "GAT")
                 .replace("homoGCNConv", "GCN").replace("normalize", "").replace("homo", "") for part in parts]
        parts = [part.replace("all", "").replace("without", "").replace("transfer", "") for part in parts]
        result = prefix + parts[0].upper() + '_' + '_'.join(parts[1:])
        result = result.strip('_')

        return result.replace('_', ',').replace(',,', ',').replace(',', ':')
    return method_name

def shorten_dataset_name(dataset_name):
    return dataset_name.replace('rotten_tomatoes', 'Rotten').replace('tweet_eval/irony', 'Tw/Irony')\
                       .replace('tweet_eval/sentiment', 'Tw/Senti').replace('tweet_eval/offensive', 'Tw/Offen')\
                       .replace('glue/cola', 'Glue/C').replace('glue/sst2', 'Glue/S')\
                       .replace('tweet_eval/hate', 'Tw/Hate').replace('tweet_eval/emotion', 'Tw/Emoti')\
                       .replace('smallnorb_label_elevation', 'SmallN/El') \
                       .replace('smallnorb_label_azimuth', 'SmallN/Az') \
                       .replace('diabetic_retinopathy_detection', 'diabetic') \
                       .replace('average', '')

def save_runtime_to_csv(all_method, all_target_dataset, all_baseline):
    def process_task_type(task_type, all_target_dataset, all_baseline):
        directory_results = f"{get_root_path_string()}/resources/experiments/{task_type}"
        performance_file = f"{directory_results}/log/performance_score.csv"
        runtime_file = f"{directory_results}/embedded_dataset/domain_similarity/runtime/runtime_results.csv"
        baseline_file = f"{directory_results}/transferability_score_records.csv"
        output_file = f"{directory_results}/result_analysis/efficiency/runtime.csv"

        model_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        performance_df = pd.read_csv(performance_file)
        runtime_df = pd.read_csv(runtime_file)
        baseline_df = pd.read_csv(baseline_file)

        # Remove model_ratio suffix from gnn_method in performance_df
        performance_df['gnn_method'] = performance_df['gnn_method'].str.replace(r'_model_ratio_\d+\.\d+', '', regex=True)

        # Merge performance and runtime dataframes on gnn_method, model_ratio, and test_dataset
        merged_df = pd.merge(performance_df, runtime_df, left_on='test_dataset', right_on='dataset', how='left', suffixes=('_perf', '_runtime'))
        merged_df['time_total'] = merged_df['time_total_perf'] + merged_df['time_total_runtime']

        # Filter for the specified methods if provided and shorten method names
        if len(all_method) != 0:
            pattern = '|'.join(all_method)
            merged_df = merged_df[merged_df['gnn_method'].str.contains(pattern)]

        merged_df['gnn_method'] = merged_df['gnn_method'].apply(shorten_method_name)

        if len(all_target_dataset) > 0:
            all_target_dataset = [determine_target_dataset(ds) for ds in all_target_dataset]
            baseline_df = baseline_df[baseline_df['target_dataset'].isin(all_target_dataset)]

        # Filter baselines
        if len(all_baseline) > 0:
            #baseline_df = baseline_df[baseline_df['metric'].isin(all_baseline)]

            # Calculate average runtime for baselines
            baseline_df = baseline_df.groupby(['metric', 'target_dataset'])['runtime'].sum().reset_index()

            # Calculate total runtime for baselines
            baseline_results = []
            for ratio in model_ratios:
                baseline_df_copy = baseline_df.copy()
                baseline_df_copy['model_ratio'] = ratio
                baseline_df_copy['time_total'] = baseline_df_copy['runtime'] * ratio
                baseline_results.append(baseline_df_copy)

            baseline_df = pd.concat(baseline_results)
            baseline_df['metric'] = baseline_df['metric'].apply(shorten_method_name)
            baseline_df = baseline_df.groupby(['metric', 'model_ratio', 'target_dataset'])['time_total'].mean().reset_index()
            baseline_df.rename(columns={'target_dataset': 'test_dataset', "metric": 'gnn_method'}, inplace=True)
        else:
            baseline_df = pd.DataFrame()

        combined_df = pd.concat([merged_df, baseline_df], ignore_index=True)
        # Pivot the merged_df to get a wide format suitable for CSV
        combined_df['test_dataset'] = combined_df['test_dataset'].apply(shorten_dataset_name)
        combined_df = combined_df.pivot(index=['gnn_method', 'model_ratio'], columns='test_dataset', values='time_total').reset_index()

        # Save to CSV
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        combined_df.to_csv(output_file, index=False)

    process_task_type("image_classification", all_target_dataset, all_baseline)
    process_task_type("sequence_classification", all_target_dataset, all_baseline)

def determine_target_dataset(target_dataset):
    if "glue" in target_dataset:
        target_dataset = "glue/" + target_dataset.split("glue_")[1]
    elif "tweet_eval" in target_dataset:
        target_dataset = "tweet_eval/" + target_dataset.split("tweet_eval_")[1]
    return target_dataset

def main():
    parser = argparse.ArgumentParser(description='Process transferability experiments.')
    parser.add_argument('--all_method', type=str, required=False, default=None)
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)
    parser.add_argument('--all_baseline', type=str, required=False, default=None)

    args = parser.parse_args()

    args.all_method = [s.strip() for s in args.all_method.split(",")] if args.all_method is not None else []
    args.all_target_dataset = [determine_target_dataset(s.strip()) for s in args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []
    args.all_baseline = [TransferabilityMethod(s.strip()) for s in args.all_baseline.split(",")] if args.all_baseline is not None else []

    save_runtime_to_csv(args.all_method, args.all_target_dataset, args.all_baseline)

if __name__ == "__main__":
    main()
