import argparse
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from transfergraph.config import get_root_path_string
sns.set_theme()
sns.set_context("talk", font_scale=1.3)
sns.set_style("whitegrid")

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

def plot_max_accuracy(task_type, all_target_dataset):
    # Construct the path to the dataset
    directory_experiments = f"{get_root_path_string()}/resources/experiments"
    data_path = f"{directory_experiments}/{task_type}/records.csv"

    # Read the data
    actual_performances = pd.read_csv(data_path)

    # Filter the actual_performances by all_target_dataset
    actual_performances = actual_performances[actual_performances['finetuned_dataset'].isin(all_target_dataset)]

    # Update dataset names
    actual_performances['finetuned_dataset'] = actual_performances['finetuned_dataset'].apply(shorten_dataset_name)

    # Group by dataset and model to get the maximum eval_accuracy for each combination
    actual_performances = actual_performances.groupby(['finetuned_dataset', 'model'])['eval_accuracy'].max().reset_index()
    actual_performances = actual_performances[~actual_performances['finetuned_dataset'].isin(['0.1', 'none', 'lora', 'dbpedia_14'])]

    # Filter by fine-tuning method
    if 'peft_method' in actual_performances.columns:
        actual_performances = actual_performances[pd.isna(actual_performances['peft_method'])]

    # Create a color palette
    palette = sns.color_palette("Set2", n_colors=len(actual_performances['finetuned_dataset'].unique()))

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

    ax.set_xlabel('Fine-tune Accuracy')
    ax.set_ylabel('Target Dataset')
    ax.tick_params(axis='y', labelrotation=45)  # Tilt the y-axis labels
    ax.tick_params(axis='x')  # Tilt the y-axis labels

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


def determine_target_dataset(target_dataset):
    if "glue" in target_dataset:
        target_dataset = "glue/" + target_dataset.split("glue_")[1]
    elif "tweet_eval" in target_dataset:
        target_dataset = "tweet_eval/" + target_dataset.split("tweet_eval_")[1]
    return target_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate boxplot for maximum evaluation accuracy.')
    parser.add_argument('--task_type', required=True, help='Task type to process (e.g., image_classification)')
    parser.add_argument('--all_target_dataset', type=str, required=False, default=None)

    args = parser.parse_args()
    args.all_target_dataset = [determine_target_dataset(s.strip()) for s in args.all_target_dataset.split(",")] if args.all_target_dataset is not None else []

    plot_max_accuracy(args.task_type, args.all_target_dataset)


if __name__ == "__main__":
    main()
