import argparse
import os
import os.path
import re
import itertools

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from transfergraph.config import get_root_path_string
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod
from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric

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
    palette = sns.color_palette(palette='Set2', n_colors=len(unique_methods))  # You can choose any palette
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
    tg_linestyle = '-'
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


def plot_correlation_metric(xaxis_type, all_method, all_target_dataset, metric):
    def plot_for_xaxis(xaxis_value, axis, task_type):
        directory_results = f"{get_root_path_string()}/resources/experiments/{task_type}/result_analysis"

        if xaxis_value == "k":
            ks = range(1, 6)  # k values from 1 to 5
            dfs = []
            for k in ks:
                filename_prefix = re.sub(r'@\d+', '@' + str(k), metric.value)
                correlation_file = f"{directory_results}/{filename_prefix}_correlation.csv"
                df = pd.read_csv(correlation_file, index_col=0)
                df['k'] = k
                dfs.append(df)
            df = pd.concat(dfs)
        elif xaxis_value == "model_ratio":
            model_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # model ratios to consider
            dfs = []
            for ratio in model_ratios:
                correlation_file = f"{directory_results}/model_ratio_{ratio}/{metric.value}_correlation.csv"
                df = pd.read_csv(correlation_file, index_col=0)
                df['model_ratio'] = ratio
                dfs.append(df)
            df = pd.concat(dfs)
        else:
            raise ValueError("Invalid xaxis value. It should be either 'k' or 'model_ratio'.")

        # Filter for the specified methods if provided and shorten method names
        if len(all_method) != 0:
            pattern = '|'.join(all_method)
            df = df[df.index.str.contains(pattern)]

        all_method_name_shortened = [shorten_method_name(name) for name in df.index]
        df.index = all_method_name_shortened
        df.index.name = "Method"

        # Melt the DataFrame to have one row per method and value for easy plotting with seaborn
        if len(all_target_dataset) > 0:
            all_dataset = set(all_target_dataset) & set(df.columns)
            df_melted = df.melt(ignore_index=False, id_vars=[xaxis_value], value_vars=all_dataset)
            df_melted.reset_index(inplace=True)
            df_melted.rename(columns={"value": "average"}, inplace=True)
        else:
            df_melted = df.melt(ignore_index=False, id_vars=[xaxis_value], value_vars=["average"])
            df_melted.reset_index(inplace=True)
            df_melted.rename(columns={"value": "average"}, inplace=True)

        # Set palette
        color_map = create_color_mapping(all_method_name_shortened)
        line_styles, markers = create_line_style_mapping(all_method_name_shortened)

        # Plot each method separately to handle different line styles and markers
        for method in sorted(set(all_method_name_shortened)):
            method_data = df_melted[df_melted['Method'] == method]
            sns.lineplot(
                ax=axis,
                data=method_data,
                x=xaxis_value,
                y='average',
                label=method,
                color=color_map[method],
                linestyle=line_styles[method],
                marker=markers[method],
                markersize=16,
            )

        if xaxis_value == "k":
            axis.set_xlabel("k")
            axis.set_xticks(ks)  # Only include the 5 ks axis labels
            axis.set_ylabel("rel@k")
        elif xaxis_value == "model_ratio":
            axis.set_xlabel("Model Ratio")
            axis.set_xticks(model_ratios)
            axis.set_ylabel("Correlation")

        axis.tick_params(axis='y')  # Set the y-axis labels
        axis.tick_params(axis='x')  # Set the x-axis labels
        axis.get_legend().remove()

        # Add title below the plot
        if task_type == "image_classification":
            axis.set_title("Image Datasets", pad=30)
        elif task_type == "sequence_classification":
            axis.set_title("Text Datasets", pad=30)
    # Prepare plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))  # Increase width for better separation

    plot_for_xaxis(xaxis_type, axes[0], "image_classification")
    plot_for_xaxis(xaxis_type, axes[1], "sequence_classification")

    # Place legend above the plot, outside the plot area
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4)

    plots_directory = f"{get_root_path_string()}/resources/experiments/combined/plots"
    if not os.path.isdir(plots_directory):
        os.makedirs(plots_directory)
    fig_path = f'{plots_directory}/lineplot_pred_model_method_average_{xaxis_type}_gat.pdf'
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
    parser.add_argument("--xaxis", help="xaxis value", choices=["k", "model_ratio"], required=True)
    parser.add_argument('--all_method', type=str, required=False, default=None)
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)
    parser.add_argument(
        '--metric',
        type=TransferabilityCorrelationMetric,
        required=True,
    )
    args = parser.parse_args()

    # Convert the comma-separated string to a list
    args.all_method = [s.strip() for s in args.all_method.split(",")] if args.all_method is not None else []
    args.all_target_dataset = [determine_target_dataset(s.strip()) for s in
                               args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []

    plot_correlation_metric(args.xaxis, args.all_method, args.all_target_dataset, args.metric)


if __name__ == "__main__":
    main()
