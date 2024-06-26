import argparse
import os
import os.path
import re
import itertools
from enum import Enum

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from transfergraph.config import get_root_path_string
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod

sns.set_theme()
sns.set_context("talk", font_scale=1.3)
sns.set_style("white")

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


def create_color_mapping(method_names):
    unique_methods = sorted(set(method_names))
    palette = sns.color_palette(palette='Set2', n_colors=len(unique_methods))
    color_map = {}
    color_iter = iter(palette)

    tg_colors = {}
    non_tg_colors = {}

    for method in unique_methods:
        if method.startswith('TG:'):
            last_part = method.split(':')[-1]
            if last_part not in tg_colors:
                tg_colors[last_part] = next(color_iter)
            color_map[method] = tg_colors[last_part]
        else:
            non_tg_colors[method] = next(color_iter)
            color_map[method] = non_tg_colors[method]

    return color_map


def create_line_style_mapping(method_names):
    markers = itertools.cycle(['o', 's', 'p', '>', 'v', '*', '8'])
    tg_markers = {}
    tg_linestyle = '-.'
    non_tg_linestyle = '-'
    line_styles = {}
    markers_map = {}

    for method in sorted(set(method_names)):
        if method.startswith('TG:'):
            second_part = method.split(':')[1]
            if second_part not in tg_markers:
                tg_markers[second_part] = next(markers)
            line_styles[method] = tg_linestyle
            markers_map[method] = tg_markers[second_part]
        else:
            line_styles[method] = non_tg_linestyle
            markers_map[method] = next(markers)

    return line_styles, markers_map


def plot_time_efficiency(all_method, all_target_dataset, all_baseline):
    def plot_for_task_type(task_type, axis, all_target_dataset, all_baseline):
        directory_results = f"{get_root_path_string()}/resources/experiments/{task_type}"
        performance_file = f"{directory_results}/log/performance_score.csv"
        runtime_file = f"{directory_results}/embedded_dataset/domain_similarity/runtime/runtime_results.csv"
        baseline_file = f"{directory_results}/transferability_score_records.csv"

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
        merged_df = merged_df.groupby(['gnn_method', 'model_ratio'])['time_total'].mean().reset_index()

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
            baseline_df = baseline_df.groupby(['metric', 'model_ratio'])['time_total'].mean().reset_index()

            baseline_df['metric'] = baseline_df['metric'].apply(shorten_method_name)

            # Combine method and baseline dataframes
            combined_df = pd.concat([merged_df, baseline_df.rename(columns={'metric': 'gnn_method'})])
        else:
            combined_df = merged_df

        # Set palette
        color_map = create_color_mapping(combined_df['gnn_method'].unique())
        line_styles, markers = create_line_style_mapping(combined_df['gnn_method'].unique())

        # Plot each method separately to handle different line styles and markers
        for method in sorted(combined_df['gnn_method'].unique()):
            method_data = combined_df[combined_df['gnn_method'] == method]
            sns.lineplot(
                ax=axis,
                data=method_data,
                x='model_ratio',
                y='time_total',
                label=method,
                color=color_map[method],
                linestyle=line_styles[method],
                marker=markers[method],
                markersize=16,
            )

        # Add title below the plot
        if task_type == "image_classification":
            axis.set_title("Image Datasets", pad=30)
        elif task_type == "sequence_classification":
            axis.set_title("Text Datasets", pad=30)

        axis.set_xlabel("Model Ratio")
        axis.set_xticks(model_ratios)
        axis.set_ylabel("Time (seconds)")
        axis.tick_params(axis='y')
        axis.tick_params(axis='x')
        #axis.set_yscale('log')  # Set the y-axis to logarithmic scale
        axis.get_legend().remove()

    # Prepare plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    plot_for_task_type("image_classification", axes[0], all_target_dataset, all_baseline)
    plot_for_task_type("sequence_classification", axes[1], all_target_dataset, all_baseline)

    # Place legend above the plot, outside the plot area
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=4)

    plots_directory = f"{get_root_path_string()}/resources/experiments/combined/plots"
    if not os.path.isdir(plots_directory):
        os.makedirs(plots_directory)
    fig_path = f'{plots_directory}/lineplot_time_efficiency_ours.pdf'
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


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
    args.all_target_dataset = [determine_target_dataset(s.strip()) for s in
                               args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []
    args.all_baseline = [TransferabilityMethod(s.strip()) for s in args.all_baseline.split(",")] if args.all_baseline is not None else []

    plot_time_efficiency(args.all_method, args.all_target_dataset, args.all_baseline)


if __name__ == "__main__":
    main()
