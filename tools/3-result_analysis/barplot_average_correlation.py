import argparse
import os.path
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from transfergraph.config import get_root_path_string
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod
from transfergraph.transferability_estimation.correlation_utils import TransferabilityCorrelationMetric


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


def create_hatch_mapping(method_names):
    patterns = itertools.cycle(['/', '\\', '|', '-'])
    tg_hatches = {}
    hatches_map = {}

    for method in sorted(set(method_names)):
        if method.startswith('TG:'):
            second_part = method.split(':')[1]
            if second_part not in tg_hatches:
                tg_hatches[second_part] = next(patterns)
            hatches_map[method] = tg_hatches[second_part]
        else:
            hatches_map[method] = ''

    return hatches_map


def plot_correlation_metric(task_type, all_method, all_target_dataset, metric):
    directory_results = f"{get_root_path_string()}/resources/experiments/{task_type}/result_analysis"
    correlation_file = f"{directory_results}/{metric.value}_correlation.csv"

    # Load the CSV file
    df = pd.read_csv(correlation_file, index_col=0)

    # Select only the specified target datasets
    if all_target_dataset:
        df = df[all_target_dataset]

    # Filter for the specified methods if provided and shorten method names
    if len(all_method) != 0:
        pattern = '|'.join(all_method)
        df = df.loc[df.index.str.contains(pattern)]

    all_method_name_shortened = [shorten_method_name(name) for name in df.index]
    df.index = all_method_name_shortened

    # Calculate average across the specified columns
    df['average'] = df.mean(axis=1)
    df.sort_values(by='average', ascending=False, inplace=True)

    # Prepare plot
    plt.figure(figsize=(10, 6))
    # Set palette and hatches
    color_map = create_color_mapping(all_method_name_shortened)
    hatch_map = create_hatch_mapping(all_method_name_shortened)
    colors = [color_map[name] for name in df.index]  # Retrieve colors for current methods
    hatches = [hatch_map[name] for name in df.index]  # Retrieve hatches for current methods

    sns.set_theme()
    sns.set_context("talk")
    sns.set_style("white")
    ax = sns.barplot(y=df.index, x='average', data=df, palette=colors, orient='h', width=0.65)
    ax.set_xlabel("Accuracy", fontsize=24, labelpad=20)
    ax.set_ylabel('Strategy', fontsize=24)
    ax.tick_params(axis='y', labelsize=20)  # Set the y-axis labels
    ax.tick_params(axis='x', labelsize=20)  # Set the x-axis labels

    # Setting x-axis limits to focus on narrow range of similar values
    xmin = 0
    xmax = df['average'].max() + 0.15  # increased right margin for annotations
    plt.xlim(xmin, xmax)

    # Apply different hatches to each bar
    for bar, hatch in zip(ax.patches, hatches):
        bar.set_hatch(hatch)

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
