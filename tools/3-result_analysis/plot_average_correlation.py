import argparse
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from transfergraph.config import get_root_path_string
from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric


def shorten_method_name(method_name):
    method_name = method_name.replace("xgb_homoGATConv_all_normalize_without_transfer", "Our approach")

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

        return result.replace('_', ',').replace(',,', ',')
    return method_name


def create_color_mapping(method_names):
    unique_methods = sorted(set(method_names))  # Sort to ensure consistent order
    palette = sns.color_palette("colorblind", n_colors=len(unique_methods))  # You can choose any palette
    color_map = {method: color for method, color in zip(unique_methods, palette)}
    return color_map


def plot_correlation_metric(task_type, all_method, all_target_dataset, metric):
    directory_results = f"{get_root_path_string()}/resources/experiments/{task_type}/result_analysis"
    correlation_file = f"{directory_results}/{metric.value}_correlation.csv"

    # Load the CSV file
    df = pd.read_csv(correlation_file, index_col=0)

    # Select only the specified target datasets
    if all_target_dataset:
        df = df[all_target_dataset]

    # Filter for the specified methods if provided and shorten method names
    df = df.loc[df.index.isin(all_method)]
    all_method_name_shortened = [shorten_method_name(name) for name in df.index]
    df.index = all_method_name_shortened

    # Calculate average across the specified columns
    df['average'] = df.mean(axis=1)
    df.sort_values(by='average', ascending=False, inplace=True)

    # Prepare plot
    plt.figure(figsize=(10, 6))
    # Set palette
    color_map = create_color_mapping(all_method_name_shortened)
    colors = [color_map[name] for name in df.index]  # Retrieve colors for current methods
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.barplot(y=df.index, x='average', data=df, palette=colors, orient='h', width=0.65)
    ax.set_xlabel("Best predicted accuracy", fontsize=24, labelpad=20)
    ax.set_ylabel('Strategy', fontsize=24)
    ax.tick_params(axis='y', labelsize=20)  # Set the y-axis labels
    ax.tick_params(axis='x', labelsize=20)  # Set the x-axis labels

    # Setting x-axis limits to focus on narrow range of similar values
    # xmin = df['average'].min() - 0.01  # slight offset to lower end
    xmin = 0
    xmax = df['average'].max() + 0.1  # increased right margin for annotations
    plt.xlim(xmin, xmax)

    # Add text labels for the exact values
    for p in ax.patches:
        ax.annotate(
            format(p.get_width(), '.3f'),
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha='left', va='center',
            xytext=(4, 0),  # 4 points horizontal offset
            textcoords='offset points',
            fontsize=16
            )  # Match the font size with tick labels

    plots_directory = f"{directory_results}/plots"
    if not os.path.isdir(plots_directory):
        os.makedirs(plots_directory)
    fig_path = f'{plots_directory}/barplot_pred_model_method_average.pdf'
    plt.savefig(fig_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def determine_target_dataset(target_dataset):
    if "glue" in target_dataset:
        target_dataset = "glue/" + target_dataset.split("glue_")[1]
    elif "tweet_eval" in target_dataset:
        target_dataset = "tweet_eval/" + target_dataset.split("tweet_eval_")[1]
    return target_dataset


def main():
    parser = argparse.ArgumentParser(description='Process transferability experiments.')
    parser.add_argument('--task_type', required=True, help='Task type to process (e.g., image_classification)')
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

    plot_correlation_metric(args.task_type, args.all_method, args.all_target_dataset, args.metric)


if __name__ == "__main__":
    main()
