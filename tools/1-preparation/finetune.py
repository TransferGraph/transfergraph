import argparse
import logging
import os.path

from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, AutoImageProcessor, \
    AutoModelForImageClassification, AutoConfig, SchedulerType

from transfergraph.config import get_root_path_string
from transfergraph.dataset.hugging_face.dataset import HuggingFaceDatasetImage, HuggingFaceDatasetText
from transfergraph.dataset.task import TaskType
from transfergraph.trainer.accelerate_trainer import AccelerateTrainer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


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

        if args.peft_method is not None:
            if args.peft_method == 'lora':
                peft_config = LoraConfig(
                    r=args.lora_dimension_attention,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias=args.lora_bias,
                    target_modules=["query", "value"],
                    modules_to_save=["classifier"],
                )
                model = get_peft_model(model, peft_config)
                print_trainable_parameters(model)
            else:
                raise Exception(f"Unexpected PEFT method {args.peft_method}")
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

        if args.peft_method is not None:
            if args.peft_method == 'lora':
                peft_config = LoraConfig(
                    task_type=PeftTaskType.SEQ_CLS,
                    r=args.lora_attention_dimension,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias=args.lora_bias,
                    target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[config.model_type],
                )
                model = get_peft_model(model, peft_config)
                print_trainable_parameters(model)
            else:
                raise Exception(f"Unexpected PEFT method {args.peft_method}")
    else:
        raise Exception(f"Unexpected task_type: {args.task_type}")

    all_training_argument = TrainingArguments(
        output_dir=os.path.join(get_root_path_string(), "models"),
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=False,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        push_to_hub_organization=args.push_to_hub_organization,
    )
    trainer = AccelerateTrainer(model, dataset, all_training_argument, args.task_type, args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to fine-tune image and text models.')
    parser.add_argument('--model_name', type=str, help='pretrained model identifier.')
    parser.add_argument('--dataset_path', required=True, type=str, help='dataset path.')
    parser.add_argument('--dataset_name', required=False, type=str, help='dataset name.')
    parser.add_argument('--task_type', type=TaskType, help='the type of task.')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the dataloaders.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--push_to_hub_organization", type=str, help="Organization to push to if applicable.")
    parser.add_argument("--peft_method", required=False, type=str, help="PEFT method to use.", choices=['lora'])
    parser.add_argument("--lora_attention_dimension", type=int, default=1, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", type=int, default=1, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=int, default=0.1, help="LoRA alpha.")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "lora_only", "all"])
    args = parser.parse_args()

    os.environ["WANDB_NAME"] = f"{args.model_name}-{args.dataset_name}"

    main(args)
