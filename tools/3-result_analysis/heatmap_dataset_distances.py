import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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

def determine_target_dataset(target_dataset):
    if "glue" in target_dataset:
        target_dataset = "glue/" + target_dataset.split("glue_")[1]
    elif "tweet_eval" in target_dataset:
        target_dataset = "tweet_eval/" + target_dataset.split("tweet_eval_")[1]
    return target_dataset

def generate_heatmap(task_type, datasets_to_include=None):
    # Construct the CSV file path based on the task type
    if task_type == 'sequence_classification':
        model = 'EleutherAI_gpt-neo-125m_gpt'
    elif task_type == 'image_classification':
        model = 'google_vit_base_patch16_224_imagenet'
    else:
        raise Exception(f'Unknown task type {task_type}')

    csv_file = f"{get_root_path_string()}/resources/experiments/{task_type}/corr_domain_similarity_{model}.csv"

    # Read the CSV file
    df = pd.read_csv(csv_file, index_col=0)

    # Clean the index and columns
    df.index = df.index.str.strip('"')
    df.columns = df.columns.str.strip('"')

    # If datasets_to_include is provided, filter the dataframe
    if datasets_to_include:
        df = df.loc[datasets_to_include, datasets_to_include]

    # Apply the shorten_dataset_name function to index and columns
    df.index = df.index.map(shorten_dataset_name)
    df.columns = df.columns.map(shorten_dataset_name)

    # Generate the heatmap without annotating the actual values
    plt.figure(figsize=(12, 10))
    inverted_cividis_red_blue_cmap = LinearSegmentedColormap.from_list(
        "inverted_cividis_red_blue",
        ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    )
    ax = sns.heatmap(df, cmap='Spectral_r', cbar=True)

    # Rotate x-axis labels, position them at the top, and remove the plot title
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor', fontsize=24)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=24)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=24)
    ax.set_title('')  # Remove the plot title
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Adjust layout to fit labels

    # Save the plot to the specified directory
    plot_dir = f"{get_root_path_string()}/resources/experiments/{task_type}/result_analysis/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = f"{plot_dir}/heatmap_{model}.png"
    plt.savefig(plot_file)

    plt.show()


if __name__ == "__main__":
    # Argument parser for command line execution
    parser = argparse.ArgumentParser(description='Generate heatmap for dataset distances.')
    parser.add_argument('--task_type', type=str, help='Task type to construct the path to the CSV file.')
    parser.add_argument(
        '--all_dataset',
        type=str,
        help='Comma-separated list of datasets to include. Include all if not provided.',
        default=None
        )

    args = parser.parse_args()

    datasets_to_include = args.all_dataset.split(',') if args.all_dataset else None
    datasets_to_include = [determine_target_dataset(target_dataset) for target_dataset in datasets_to_include]

    generate_heatmap(args.task_type, datasets_to_include)
