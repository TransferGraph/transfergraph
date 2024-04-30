import argparse
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from transfergraph.config import get_root_path_string


def plot_max_accuracy(task_type):
    # Construct the path to the dataset
    directory_experiments = f"{get_root_path_string()}/resources/experiments"
    data_path = f"{directory_experiments}/{task_type}/records.csv"

    # Read the data
    actual_performances = pd.read_csv(data_path)

    # Group by dataset and model to get the maximum eval_accuracy for each combination
    actual_performances = actual_performances.groupby(['finetuned_dataset', 'model'])['eval_accuracy'].max().reset_index()
    actual_performances = actual_performances[~actual_performances['finetuned_dataset'].isin(['0.1', 'none', 'lora', 'dbpedia_14'])]

    # Filter by fine-tuning method
    if 'peft_method' in actual_performances.columns:
        actual_performances = actual_performances[pd.isna(actual_performances['peft_method'])]

    # Set the aesthetic style of the plots
    sns.set_theme()
    sns.set_context("paper")

    # Create a color palette
    palette = sns.color_palette("muted", n_colors=len(actual_performances['finetuned_dataset'].unique()))

    # Create the horizontal boxplot for the uploaded data
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(
        data=actual_performances,
        x='eval_accuracy',
        y='finetuned_dataset',
        orient='h',
        hue='finetuned_dataset',  # Assign 'hue' to utilize the palette effectively
        palette=palette,
        dodge=False  # Set dodge to False to avoid separation by hue
    )

    ax.set_xlabel('Fine-tune Accuracy', fontsize=24, labelpad=20)
    ax.set_ylabel('Target Dataset', fontsize=24)
    ax.tick_params(axis='y', labelrotation=45, labelsize=20)  # Tilt the y-axis labels
    ax.tick_params(axis='x', labelsize=20)  # Tilt the y-axis labels

    plt.tight_layout()

    # Save the figure as a PDF for inclusion
    plots_directory = f"{directory_experiments}/{task_type}/result_analysis/plots"

    if not os.path.isdir(plots_directory):
        os.makedirs(plots_directory)

    fig_path = f'{plots_directory}/boxplot_accuracies_horizontal.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    # Show plot
    plt.show()

    # Print path to the saved figure
    print(f"Boxplot saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate boxplot for maximum evaluation accuracy.')
    parser.add_argument('--task_type', required=True, help='Task type to process (e.g., image_classification)')

    args = parser.parse_args()

    plot_max_accuracy(args.task_type)


if __name__ == "__main__":
    main()
