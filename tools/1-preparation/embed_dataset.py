import argparse
import logging

import torch
from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification, AutoTokenizer, AutoModelForSequenceClassification

from transfergraph.dataset.embed_utils import DatasetEmbeddingMethod
from transfergraph.dataset.embedder import DatasetEmbedder
from transfergraph.dataset.hugging_face.dataset import HuggingFaceDatasetImage, HuggingFaceDatasetText
from transfergraph.dataset.task import TaskType


def main(args: argparse.Namespace):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.task_type == TaskType.IMAGE_CLASSIFICATION:
        dataset = HuggingFaceDatasetImage.load(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            image_processor=AutoImageProcessor.from_pretrained(args.model_name)
        )
        config = AutoConfig.from_pretrained(
            args.model_name,
            finetuning_task=args.task_type,
        )
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name,
            config=config,
        )
    elif args.task_type == TaskType.SEQUENCE_CLASSIFICATION:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # Check if the tokenizer does not have a pad token
        if tokenizer.pad_token is None:
            # Set the pad token to eos_token (end-of-sequence token) if it's not set.
            # You can also choose another token as the pad token if you prefer.
            tokenizer.pad_token = tokenizer.eos_token
        dataset = HuggingFaceDatasetText.load(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            max_train_samples=args.max_train_samples
        )
        config = AutoConfig.from_pretrained(
            args.model_name,
            finetuning_task=args.task_type,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            config=config,
        )
        # Ensure the model's pad token id is updated
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        raise Exception(f"Unexpected task_type: {args.task_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda
    embedder = DatasetEmbedder(
        model.to(device),
        dataset,
        args.embedding_method,
        args.task_type
    )

    embedder.embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to fine-tune image and text models.')
    parser.add_argument('--model_name', required=True, type=str, help='pretrained model identifier.')
    parser.add_argument('--dataset_path', required=True, type=str, help='dataset path.')
    parser.add_argument('--dataset_name', required=False, type=str, help='dataset name.')
    parser.add_argument('--task_type', type=TaskType, help='the type of task.')
    parser.add_argument('--embedding_method', required=True, type=DatasetEmbeddingMethod, help='the type of embedding method.')
    parser.add_argument('--max_train_samples', type=int, required=False, default=1000000, help='The maximum number of samples.')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the dataloaders.",
    )

    args = parser.parse_args()

    main(args)
