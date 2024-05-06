import os
import time

import numpy as np
import pandas as pd
from transformers import PreTrainedModel

from transfergraph.config import get_root_path_string
from transfergraph.dataset.base_dataset import BaseDataset
from transfergraph.dataset.embed_utils import DatasetEmbeddingMethod
from transfergraph.dataset.task import TaskType
from transfergraph.transferability_estimation.feature_utils import extract_features_without_labels


class DatasetEmbedder:
    def __init__(self, probe_model: PreTrainedModel, dataset: BaseDataset, embedding_method: DatasetEmbeddingMethod, task_type: TaskType):
        self.probe_model = probe_model
        self.dataset = dataset
        self.embedding_method = embedding_method
        self.task_type = task_type
        self.time_start = time.time()

    def embed(self):
        directory = determine_directory_embedded_dataset(self.probe_model.name_or_path, self.task_type, self.embedding_method)

        if self.embedding_method == DatasetEmbeddingMethod.DOMAIN_SIMILARITY:
            features_tensor, features_dimension = extract_features_without_labels(self.dataset.train_loader, self.probe_model)

            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(determine_file_name_embedded_dataset(directory, self.dataset.name), features_tensor)
        else:
            raise Exception(f"Unexpected DatasetEmbeddingMethod: {self.embedding_method.name}")

        self.save_runtime(directory)

    def save_runtime(self, directory):
        runtime_dict = {
            "dataset": self.dataset.name,
            "embedding_method": self.embedding_method.value,
            "probe_model": self.probe_model.name_or_path,
            "number_of_samples": len(self.dataset.train_loader.dataset),
            "batch_size": self.dataset.train_loader.batch_size,
            "time_total": time.time() - self.time_start
        }
        runtime_dir = f"{directory}/runtime"
        os.makedirs(runtime_dir, exist_ok=True)
        runtime_file = f"{runtime_dir}/runtime_results.csv"
        if not os.path.exists(runtime_file):
            runtime_df = pd.DataFrame(
                columns=["dataset", "embedding_method", "probe_model", "number_of_samples", "batch_size", "time_total"]
            )
        else:
            runtime_df = pd.read_csv(runtime_file, index_col=0)
        runtime_df = pd.concat([runtime_df, pd.DataFrame(runtime_dict, index=[0])], ignore_index=True)
        runtime_df.to_csv(runtime_file)


def determine_file_name_embedded_dataset(directory: str, dataset_name: str):
    dataset_name_sanitized = dataset_name.replace('/', '_').replace(' ', '-')

    return os.path.join(directory, dataset_name_sanitized + f'_feature.npy')


def determine_directory_embedded_dataset(probe_model_name: str, task_type: TaskType, embedding_method: DatasetEmbeddingMethod):
    model_name_sanitized = probe_model_name.replace('/', '_')
    return os.path.join(
        get_root_path_string(),
        "resources",
        "experiments",
        task_type.value,
        'embedded_dataset/',
        embedding_method.value,
        model_name_sanitized
    )
