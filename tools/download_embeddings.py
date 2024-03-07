import argparse

from huggingface_hub import HfApi

from transfergraph.dataset.embed_utils import DatasetEmbeddingMethod
from transfergraph.dataset.embedder import determine_directory_embedded_dataset
from transfergraph.dataset.task import TaskType


def main(args: argparse.Namespace):
    api = HfApi()
    model_name_sanitized = args.model_name.replace("/", "_")
    api.snapshot_download(
        repo_id=f"TransferGraph/embedded_dataset_{args.embedding_method.value}_{model_name_sanitized}",
        local_dir=determine_directory_embedded_dataset(args.model_name, args.task_type, args.embedding_method),
        repo_type="dataset",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to download pre-computed dataset embeddings.')
    parser.add_argument('--model_name', required=True, type=str, help='pretrained model identifier.')
    parser.add_argument('--embedding_method', required=True, type=DatasetEmbeddingMethod, help='the type of embedding method.')
    parser.add_argument('--task_type', type=TaskType, required=True, help='the type of task.')

    args = parser.parse_args()

    main(args)
