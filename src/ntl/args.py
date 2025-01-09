from dataclasses import dataclass, field
from typing import Optional, Literal

from transformers import (
    Seq2SeqTrainingArguments, MODEL_WITH_LM_HEAD_MAPPING
)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
    NOTE: Expanding TrainingArguments class from transformers with custom arguments.
    """
    do_only_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Only evaluate the model."
        },
    )
    trial: Optional[str] = field(
        default=None,
        metadata={
            "help": "Version name for this run"
        },
    )
    special_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Special name for this run"
        },
    )
    language_modelling: Literal["clm", "mlm"] = field(
        default="clm",
        metadata={
            "help": "Choose either clm or mlm for language modelling"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    name: Optional[str] = field(
        default=None,
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
                    + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    number_encoding: Optional[str] = field(
        default="rt",
        metadata={
            "help": "Choose either xval or rt or None, or none_regression_head for number encodings"
        },
    )
    number_token_loss: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Adds NumberTokenLoss object"
        },
    )
    number_token_loss_with_wasserstein: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Adds NumberTokenLoss object with Wasserstein distance"
        },
    )
    number_token_loss_weight: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Weight of the number_token_loss in reference to other loss"
        },
    )
    number_token_loss_function: Optional[str] = field(
        default="mse",
        metadata={
            "help": "Loss function for number token loss. Allowed: mse, huber, mae."
        },
    )
    log_scale_embeddings: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log scale the embeddings. Only applicable for xval and rt."
        },
    )
    xval_bigger_language_head: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use a bigger language head for xval."
        },
    )


@dataclass
class DatasetArguments:
    train_with_augmented_data: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Train with augmented data."
        },
    )

    dataset_name: str = field(
        default="mathematics_dataset",
        metadata={
            "help": "Name of the dataset. Allowed: mathematics_dataset, gsm8k, multiplication, arithmetic"
        },
    )

    mode: Optional[str] = field(
        default="interpolate_extrapolate",
        metadata={
            "help": "Whether we combine mathematics datasets in testing, or test individually. Allowed: interpolate_extrapolate, dataset_comparison"
        },
    )
