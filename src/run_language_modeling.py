#!/usr/bin/env python3
"""
Language modeling adapted from Huggingface transformers.

The file is an adaptation of https://github.com/huggingface/transformers/blob/v3.1.0/examples/language-modeling/run_language_modeling.py

"""
import sys
sys.path.append("..")

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    HfArgumentParser,
    set_seed,
)
from transformers.training_args import TrainingArguments

from src.data import load_txt_dataset
from src.encoding_decoding.rt_encoding_decoding import CustomMaskingCollator
from src.tokenizer.rt_tokenizer import RtTokenizer
from src.tokenizer.xval_tokenizer import XvalTokenizer
from src.trainer import CustomTrainer, get_trainer_dict
from src.transformer_backbone.t5 import T5RegressionModel

transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_latest_checkpoint(model_path: str, must_contain: str = "best") -> str:
    """
    Given a path to the model folder it searches the latest saved checkpoint
    and returns the path to it.
    Args:
        model_path (str): Path to model folder. Has to contain folders called
            'checkpoint-best-STEP' and 'checkpoint-latest-STEP' where STEP is
            a positive integer.
        must_contain (str, optional): Subselect checkpoints that contain a
            certain query. Defaults to 'best'.
    Returns:
        str: Path to latest checkpoint
    """

    # Finding checkpoints
    checkpoints = [f for f in os.listdir(model_path) if f.startswith("checkpoint")]
    if must_contain is not None:
        checkpoints = list(filter(lambda x: must_contain in x, checkpoints))

    if len(checkpoints) == 0:
        logger.warning(
            f"No checkpoints found that contain {must_contain} in {model_path}."
        )
        # Relax criteria and retry
        next_try = "checkpoint" if must_contain != "checkpoint" else ""
        return get_latest_checkpoint(model_path, must_contain=next_try)

    # Sorting
    try:
        idx = np.argsort([int(c.split("-")[-1]) for c in checkpoints])[-1]
    except ValueError:
        raise ValueError(f"Checkpoints dont seem to follow format: {checkpoints}.")

    return os.path.join(model_path, checkpoints[idx])


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    NOTE: Expanding TrainingArguments class from transformers with custom arguments.

    eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
    """

    # Was introduced only in transformers 3.4.0
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of predictions steps to accumulate before moving the tensors to the CPU."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

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
            "help": "Chose either xval or rt for number encodings"
        }
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Switch off comet
    os.environ["COMET_MODE"] = "DISABLED"

    # Switch off WandB
    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser(
        (ModelArguments, CustomTrainingArguments)
    )
    model_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    if model_args.config_name:
        # if file exists load it otherwise just use config name
        if os.path.isfile(model_args.config_name):
            with open(model_args.config_name, "r") as f:
                model_params = json.load(f)
        else:
            model_params = {}

        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            mem_len=model_params.get("mem_len", 1024),
        )

    elif model_args.model_name_or_path:
        if "checkpoint" not in model_args.model_name_or_path:
            model_args.model_name_or_path = get_latest_checkpoint(
                model_args.model_name_or_path,
                must_contain="best",
            )

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        model_params = config.__dict__

    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        model_params = config.__dict__
        logger.warning("You are instantiating a new config instance from scratch.")

    # load tokenizer    
    if model_args.number_encoding == "rt":
        if model_args.tokenizer_name:
            tokenizer = RtTokenizer.from_pretrained(
                model_args.tokenizer_name, cache_dir=model_args.cache_dir
            )

        elif model_args.model_name_or_path:
            tokenizer = RtTokenizer.from_pretrained(
                model_args.model_name_or_path, cache_dir=model_args.cache_dir
            )
        elif model_args.config_name:
            tokenizer = RtTokenizer.from_pretrained(
                model_args.config_name, cache_dir=model_args.cache_dir
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )
    elif model_args.number_encoding == "xval":
        if model_args.tokenizer_name:
            tokenizer = XvalTokenizer.from_pretrained(
                model_args.tokenizer_name, cache_dir=model_args.cache_dir
            )

        elif model_args.model_name_or_path:
            tokenizer = XvalTokenizer.from_pretrained(
                model_args.model_name_or_path, cache_dir=model_args.cache_dir
            )
        elif model_args.config_name:
            tokenizer = XvalTokenizer.from_pretrained(
                model_args.config_name, cache_dir=model_args.cache_dir
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )

    if model_args.model_name_or_path:

        # Restore checkpoint if available
        if "checkpoint" not in model_args.model_name_or_path:
            model_args.model_name_or_path = get_latest_checkpoint(
                model_args.model_name_or_path,
                must_contain="best",
            )
        config.vocab_size = len(tokenizer)  # Update vocab size
        model = T5RegressionModel.from_pretrained( 
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        logger.info("Model restored")

        # Get min loss so far
        try:
            loss_df = pd.read_csv(
                os.path.join(model_args.model_name_or_path, "training_log.csv"),
                index_col=0,
            )
            model_params.update({"training_logs": list(loss_df.T.to_dict().values())})
            logger.info("Restored training loss history.")
        except Exception:
            logger.warning(
                "Could not find loss history, might overwrite good checkpoints."
            )

    else:
        logger.info("Training new model from scratch")
        model = T5RegressionModel(config=config)

    logger.info(f"PyTorch version: {torch.__version__}")
    model.resize_token_embeddings(len(tokenizer))

    # Set number embeddings for RT encoding
    if model_args.number_encoding == "rt":
        model.set_number_embeds(len(tokenizer), tokenizer.get_vocab())

    # Get datasets
    data_path = 'data/mathematics_dataset-v1.0/mathematics_dataset-v1.0/train-easy/algebra__linear_1d_small.txt'
    train_dataset = load_txt_dataset(data_path)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")


    # Conditional Generation Training
    data_collator = CustomMaskingCollator(
        tokenizer=tokenizer
    )

    #custom_trainer_params = get_trainer_dict(model_params)

    # Initialize our Trainer
    """trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        use_numerical_encodings=True,
        d_model=model.config.hidden_size,
        model_type="t5-small", # TODO as parameter
        vmax=10,
        **custom_trainer_params,
    )"""
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer        
    )

    # Training
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None
           and os.path.isdir(model_args.model_name_or_path)
        else None
    )
    trainer.train(model_path=model_path)
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.state.is_world_process_zero:
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()


