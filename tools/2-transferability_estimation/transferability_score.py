import argparse
import logging

import torch
from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification, AutoTokenizer, AutoModelForSequenceClassification

from transfergraph.dataset.hugging_face.dataset import HuggingFaceDatasetImage, HuggingFaceDatasetText
from transfergraph.dataset.task import TaskType
from transfergraph.transferability_estimation.baseline.methods.estimator import TransferabilityEstimatorFeatureBased
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod


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
            num_labels=len(dataset.all_class),
            i2label={label: str(i) for i, label in enumerate(dataset.all_class)},
            label2id={str(i): label for i, label in enumerate(dataset.all_class)},
            finetuning_task=args.task_type.value,
        )
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    elif args.task_type == TaskType.SEQUENCE_CLASSIFICATION:
        dataset = HuggingFaceDatasetText.load(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            tokenizer=AutoTokenizer.from_pretrained(args.model_name)
        )
        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=len(dataset.all_class),
            i2label={label: str(i) for i, label in enumerate(dataset.all_class)},
            label2id={str(i): label for i, label in enumerate(dataset.all_class)},
            finetuning_task=args.task_type.value,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    else:
        raise Exception(f"Unexpected task_type: {args.task_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda
    estimator = TransferabilityEstimatorFeatureBased(
        dataset,
        model.to(device),
        args.metric,
        args,
    )
    estimator.score()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to fine-tune image and text models.')
    parser.add_argument('--model_name', required=True, type=str, help='pretrained model identifier.')
    parser.add_argument('--dataset_path', required=True, type=str, help='dataset path.')
    parser.add_argument('--dataset_name', required=False, type=str, help='dataset name.')
    parser.add_argument('--task_type', type=TaskType, required=True, help='the type of task.')
    parser.add_argument('--metric', required=True, type=TransferabilityMethod, help='the type of metric.')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the dataloaders.",
    )

    args = parser.parse_args()

    main(args)
