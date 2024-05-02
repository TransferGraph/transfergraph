from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from transfergraph.dataset.base_dataset import BaseDataset
from transfergraph.dataset.hugging_face.builder import HuggingFaceDatasetBuilderImage, HuggingFaceDatasetBuilderText


class HuggingFaceDatasetText(BaseDataset):
    def __init__(
            self,
            name: str,
            train_loader: DataLoader,
            eval_loader: DataLoader,
            all_class: list,
            tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(name, train_loader, eval_loader, all_class)

        self.tokenizer = tokenizer

    @classmethod
    def load(
            cls,
            dataset_path: str,
            tokenizer: PreTrainedTokenizer,
            batch_size: int,
            dataset_name: str = None,
            max_train_samples: int = None
    ):
        builder = HuggingFaceDatasetBuilderText(dataset_path, tokenizer, batch_size, dataset_name, max_train_samples)

        return builder.download_and_preprocess()


class HuggingFaceDatasetImage(BaseDataset):
    @classmethod
    def load(
            cls,
            dataset_path: str,
            image_processor: BaseImageProcessor,
            batch_size: int,
            dataset_name: str = None
    ):
        builder = HuggingFaceDatasetBuilderImage(dataset_path, image_processor, batch_size, dataset_name)

        return builder.download_and_preprocess()
