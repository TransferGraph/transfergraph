import math
import os

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Lambda, Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, \
    Resize, CenterCrop
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from transformers.image_processing_utils import BaseImageProcessor

from transfergraph.config import get_root_path_string
from transfergraph.dataset.base_dataset import ALL_DATASET_CONFIG


class HuggingFaceDatasetBuilder:
    def __init__(self, dataset_path: str, batch_size: int, dataset_name: str = None, max_train_samples: int = None):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.max_train_samples = max_train_samples

        self.config = self.determine_dataset_config()

        if self.dataset_name is None:
            self.dataset_full_name = self.dataset_path
        else:
            self.dataset_full_name = f'{self.dataset_path}/{self.dataset_name}'

        if os.getenv("HF_HOME") is None and os.getenv("HF_DATASETS_CACHE") is None:
            if self.dataset_name is None:
                self.cache_directory = os.path.join(get_root_path_string(), "data/hugging_face/", self.dataset_path)
            else:
                self.cache_directory = os.path.join(get_root_path_string(), "data/hugging_face/", self.dataset_path, self.dataset_name)
        else:
            self.cache_directory = None

        self.label_key = self.config.get("label_key", "label")

    def determine_dataset_config(self):
        if self.dataset_name is None:
            if self.dataset_path in ALL_DATASET_CONFIG:
                return ALL_DATASET_CONFIG[self.dataset_path]
            else:
                raise Exception(
                    f"No configuration defined for {self.dataset_path}, please add it to transfergraph/dataset/base_dataset.py."
                    )
        else:
            if self.dataset_path in ALL_DATASET_CONFIG and self.dataset_name in ALL_DATASET_CONFIG[self.dataset_path]['tasks']:
                return ALL_DATASET_CONFIG[self.dataset_path]['tasks'][self.dataset_name]
            else:
                raise Exception(
                    f"No configuration defined for {self.dataset_path}/{self.dataset_name}, please add it to transfergraph/dataset/base_dataset.py."
                    )

    def _load_dataset(self):
        source = self.config.get("source", "huggingface")

        if source == "huggingface":
            if "alias" in self.config:
                all_dataset = load_dataset(self.config["alias"], cache_dir=self.cache_directory)
            else:
                all_dataset = load_dataset(self.dataset_path, self.dataset_name, cache_dir=self.cache_directory)
        elif source == 'local':
            file_type = self.config["type"]
            train_path = os.path.join(get_root_path_string(), self.config["train_path"])
            validation_path = os.path.join(get_root_path_string(), self.config["validation_path"])

            all_dataset = {
                "train": load_dataset(file_type, data_files=train_path, split="train"),
                "validation": load_dataset(file_type, data_files=validation_path, split="train")
            }
        else:
            raise Exception(f"Unexpected source {source}")

        if self.max_train_samples is None:
            return all_dataset
        else:
            ratio = None

            num_samples = len(all_dataset["train"])

            if num_samples > self.max_train_samples:
                ratio = self.max_train_samples / num_samples

            for dataset_key in all_dataset:
                if ratio is not None:
                    num_samples = len(all_dataset[dataset_key])

                    all_dataset[dataset_key] = all_dataset[dataset_key].select(range(math.floor(num_samples * ratio)))

            return all_dataset

    @staticmethod
    def determine_validation_split_key(datasets):
        for split_key in ["validation", "eval", "test"]:
            if split_key in datasets:
                return split_key

        return None


class HuggingFaceDatasetBuilderText(HuggingFaceDatasetBuilder):
    def __init__(
            self,
            dataset_path: str,
            tokenizer: PreTrainedTokenizer,
            batch_size: int,
            dataset_name: str = None,
            max_train_samples: int = None
    ):
        super().__init__(dataset_path, batch_size, dataset_name, max_train_samples)

        self.tokenizer = tokenizer
        self.all_feature_key = self.config.get("all_feature_key", ["sentence"])

    def download_and_preprocess(self):
        datasets = self._load_dataset()
        processed_datasets = self._preprocess(datasets)
        data_collator = DataCollatorWithPadding(self.tokenizer)

        train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, batch_size=self.batch_size, collate_fn=data_collator)

        if "validation" in processed_datasets:
            validation_dataloader = DataLoader(processed_datasets["validation"], batch_size=self.batch_size, collate_fn=data_collator)
        else:
            validation_dataloader = None

        if "labels" in processed_datasets["train"].column_names:
            label_list = processed_datasets["train"].unique("labels")
            label_list.sort()
        else:
            label_list = list()

        from transfergraph.dataset.hugging_face.dataset import HuggingFaceDatasetText
        return HuggingFaceDatasetText(
            self.dataset_full_name,
            train_dataloader,
            validation_dataloader,
            label_list,
            self.tokenizer
        )

    def _preprocess(
            self,
            datasets: DatasetDict,
    ) -> dict:
        def preprocess_function(examples):
            all_input_key = self.all_feature_key
            all_input_text = []

            for intput_key in all_input_key:
                all_input_text.append(examples[intput_key])

            texts = (
                (*all_input_text,)
            )
            result = self.tokenizer(*texts, truncation=True)

            if self.label_key in examples:
                result['labels'] = examples[self.label_key]

            return result

        train_dataset_tokenized = datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
            desc=f"Running tokenizer on train dataset",
        )
        validation_split_key = self.determine_validation_split_key(datasets)

        if validation_split_key is None:
            return {"train": train_dataset_tokenized}
        else:
            validation_dataset_tokenized = datasets[validation_split_key].map(
                preprocess_function,
                batched=True,
                remove_columns=datasets[validation_split_key].column_names,
                desc=f"Running tokenizer on {validation_split_key} dataset",
            )

            return {"train": train_dataset_tokenized, "validation": validation_dataset_tokenized}


class HuggingFaceDatasetBuilderImage(HuggingFaceDatasetBuilder):
    def __init__(
            self,
            dataset_path: str,
            image_processor: BaseImageProcessor,
            batch_size: int,
            dataset_name: str = None,
            max_train_samples: int = None
    ):
        super().__init__(dataset_path, batch_size, dataset_name, max_train_samples)

        self.image_processor = image_processor
        self.feature_key = self.config.get("feature_key", "img")

    def download_and_preprocess(self):
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example[self.label_key] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        datasets = self._load_dataset()
        processed_datasets = self._preprocess(datasets)

        if "validation" in processed_datasets:
            validation_loader = DataLoader(processed_datasets["validation"], batch_size=self.batch_size, collate_fn=collate_fn)
        else:
            validation_loader = None

        from transfergraph.dataset.hugging_face.dataset import HuggingFaceDatasetImage
        return HuggingFaceDatasetImage(
            self.dataset_full_name,
            DataLoader(processed_datasets["train"], shuffle=True, batch_size=self.batch_size, collate_fn=collate_fn),
            validation_loader,
            processed_datasets["train"].unique(self.label_key),
        )

    def _preprocess(self, datasets: DatasetDict) -> dict:
        if "shortest_edge" in self.image_processor.size:
            size = self.image_processor.size["shortest_edge"]
        else:
            size = (self.image_processor.size["height"], self.image_processor.size["width"])
        normalize = (
            Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
            if hasattr(self.image_processor, "image_mean") and hasattr(self.image_processor, "image_std")
            else Lambda(lambda x: x)
        )
        train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
        val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

        def preprocess_train(example_batch):
            """Apply _train_transforms across a batch."""
            example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch[self.feature_key]]

            return example_batch

        def preprocess_val(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch[self.feature_key]]

            return example_batch

        train_dataset = datasets["train"].with_transform(preprocess_train)

        validation_split_key = self.determine_validation_split_key(datasets)

        if validation_split_key is None:
            return {"train": train_dataset}
        else:
            eval_dataset = datasets[validation_split_key].with_transform(preprocess_val)

            return {"train": train_dataset, "validation": eval_dataset}
