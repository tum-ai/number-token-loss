#!/usr/bin/env python3
"""
Language modeling adapted from Huggingface transformers.

The file is an adaptation of https://github.com/huggingface/transformers/blob/v3.1.0/examples/language-modeling/run_language_modeling.py

"""
import sys

from src.trainer import CustomSeq2SeqTrainer
from src.transformer_backbone.t5.t5_vanilla_for_number_token_loss import T5VanillaForNumberTokenLoss

sys.path.append("..")

import json
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass, field
from typing import Optional, Literal
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback, T5ForConditionalGeneration, Seq2SeqTrainingArguments
)

from src.data import load_txt_dataset, load_json_dataset
from src.collators.rt_question_answer_collator import RtQuestionAnswerCLMCollator
from src.collators.xval_question_answer_collator import XvalQuestionAnswerCLMCollator
from src.collators.vanilla_question_answer_collator import VanillaQuestionAnswerCLMCollator
from src.tokenizer.rt_tokenizer import RtTokenizer
from src.tokenizer.xval_tokenizer import XvalTokenizer
from src.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from src.transformer_backbone.t5.t5_rt import T5RegressionModelRT
from src.transformer_backbone.t5.t5_xval import T5RegressionModelXval
from src.evaluation import CustomMetrics
from src.number_token_loss import NumberTokenLoss

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
class CustomTrainingArguments(Seq2SeqTrainingArguments):
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

    do_only_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Only evaluate the model."
        },
    )

    train_with_augmented_data: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Train with augmented data."
        },
    )

    dataset_name: Literal["mathematics_dataset", "gsm8k"] = field(
        default="mathematics_dataset",
        metadata={
            "help": "Name of the dataset."
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
            "help": "Chose either xval or rt or None for number encodings"
        },
    )
    number_token_loss: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Adds NumberTokenLoss object"
        },
    )
    number_token_loss_weight: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Weight of the number_token_loss in reference to other loss"
        },
    )
    number_token_loss_function: Optional[Literal["mse", "huber", "mae"]] = field(
        default="mse",
        metadata={
            "help": "Sets the order of the NTL. For example 2 -> MSE, 3 -> Mean Cubic Error etc."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Switch off comet
    os.environ["COMET_MODE"] = "DISABLED"

    # Switch off WandB
    os.environ["WANDB_DISABLED"] = "false"

    parser = HfArgumentParser(
        (ModelArguments, CustomTrainingArguments)
    )
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set generation arguments
    training_args.predict_with_generate = True
    if model_args.number_encoding != "xval":
        training_args.generation_num_beams = 4
        logger.info("Setting generation_num_beams to 4")
    else:
        logger.info("Setting generation_num_beams to 1 for xval")

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
    logger.info("Training on dataset: %s", training_args.dataset_name)
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

    if model_args.number_encoding == "rt":
        model_class = T5RegressionModelRT
        tokenizer_class = RtTokenizer
    elif model_args.number_encoding == "xval":
        model_class = T5RegressionModelXval
        tokenizer_class = XvalTokenizer
    elif model_args.number_encoding.lower() == "none":
        if model_args.number_token_loss:
            model_class = T5VanillaForNumberTokenLoss
            tokenizer_class = T5Custom_Tokenizer
        else:
            model_class = T5ForConditionalGeneration
            tokenizer_class = transformers.AutoTokenizer
    else:
        raise ValueError(f"Unknown number encoding: {model_args.number_encoding}")

    # load tokenizer    
    if model_args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    elif model_args.config_name:
        tokenizer = tokenizer_class.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.number_encoding != "none" or model_args.number_token_loss:
        n_new_tokens = len(tokenizer) - len(transformers.AutoTokenizer.from_pretrained("t5-small"))
        logger.info(f"Number of new tokens: {n_new_tokens}")
        logger.info(f"Old vocab size: {config.vocab_size}")
        config.vocab_size = config.vocab_size + n_new_tokens
        logger.info(f"New vocab size: {config.vocab_size}")

        config.added_vocab = tokenizer.get_added_vocab()

    if model_args.number_encoding == "xval":
        model_init_kwargs = {"tokenizer": tokenizer}
    else:
        model_init_kwargs = {}

    if model_args.number_encoding == "xval" and model_args.number_token_loss:
        raise Exception("Xval does not accept NumberTokenLoss")

    if model_args.number_token_loss:
        if model_args.number_token_loss_function == "mse":
            loss_function = F.mse_loss
        elif model_args.number_token_loss_function == "huber":
            loss_function = F.huber_loss
        elif model_args.number_token_loss_function == "mae":
            loss_function = F.l1_loss
        else:
            raise ValueError(f"Unknown loss function: {model_args.number_token_loss_function}")

        model_init_kwargs["number_token_loss"] = NumberTokenLoss(
            tokenizer,
            vocab_size=config.vocab_size,
            device=training_args.device,
            loss_function=loss_function,
            weight=model_args.number_token_loss_weight
        )

    if model_args.model_name_or_path:

        # delete generation config, as not deleting leads to model not being loadable
        if os.path.isdir(model_args.model_name_or_path) and os.path.exists(os.path.join(model_args.model_name_or_path, "generation_config.json")):
            os.remove(os.path.join(model_args.model_name_or_path, "generation_config.json"))

        # Restore checkpoint if available
        # if "checkpoint" not in model_args.model_name_or_path:
        #     model_args.model_name_or_path = get_latest_checkpoint(
        #         model_args.model_name_or_path,
        #         must_contain="best",
        #     )
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            ignore_mismatched_sizes=True,
            **model_init_kwargs,
        )

        if model_args.number_encoding == "xval":
            model.initialize_num_head_weights()

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
        model = model_class(config=config, **model_init_kwargs)

    logger.info(f"PyTorch version: {torch.__version__}")

    # Get datasets

    if training_args.dataset_name == "gsm8k":
        if training_args.train_with_augmented_data:
            train_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/train_t_clean_with_augmented.jsonl'
        else:
            train_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/train_t_clean.jsonl'
        eval_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/val_t_clean.jsonl'
        test_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/test_clean.jsonl'
        train_dataset = load_json_dataset(train_data_path)
        eval_dataset = load_json_dataset(eval_data_path)
        test_dataset = load_json_dataset(test_data_path)
    elif training_args.dataset_name == "mathematics_dataset":
        train_data_path = 'data/mathematics_dataset-v1.0/train.txt'
        eval_data_path = 'data/mathematics_dataset-v1.0/val.txt'
        test_interpolate_data_path = 'data/mathematics_dataset-v1.0/test_interpolate.txt'
        test_extrapolate_data_path = 'data/mathematics_dataset-v1.0/test_extrapolate.txt'

        train_dataset = load_txt_dataset(train_data_path)
        eval_dataset = load_txt_dataset(eval_data_path)
        test_interpolate_dataset = load_txt_dataset(test_interpolate_data_path)
        test_extrapolate_dataset = load_txt_dataset(test_extrapolate_data_path)
    else:
        raise ValueError(f"Unknown dataset: {training_args.dataset_name}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")

    # Conditional Generation Training
    if model_args.number_encoding == "rt":
        data_collator = RtQuestionAnswerCLMCollator(
            tokenizer=tokenizer
        )
    elif model_args.number_encoding == "xval":
        data_collator = XvalQuestionAnswerCLMCollator(
            tokenizer=tokenizer
        )
    elif model_args.number_encoding.lower() == "none":
        # Rt collator can be used for default T5 as well
        data_collator = VanillaQuestionAnswerCLMCollator(
            tokenizer=tokenizer
        )

    # Custom Metric
    custom_metrics = CustomMetrics(
        tokenizer=tokenizer,
        number_encoding=model_args.number_encoding,
        output_dir=training_args.output_dir,
        save_all_output=True,
    )

    # Early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001)

    # custom_trainer_params = get_trainer_dict(model_params)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # callbacks=[early_stopping_callback],
        compute_metrics=custom_metrics,
    )

    # Training
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None
           and os.path.isdir(model_args.model_name_or_path)
        else None
    )

    if not training_args.do_only_eval:
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.state.is_world_process_zero:
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        logger.info("Skipping training.")



        # logger.info("*** Evaluate on training data ***")
        # eval_results = trainer.evaluate(eval_dataset=train_dataset)
        # logger.info(f"eval_results training data: {eval_results}")

    logger.info("*** Evaluate on validation data ***")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info(f"eval_results validation data: {eval_results}")

    if training_args.dataset_name == "gsm8k":
        logger.info("*** Evaluate on test set ***")
        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"eval_results test data: {eval_results}")
    elif training_args.dataset_name == "mathematics_dataset":
        logger.info("*** Evaluate on interpolate data ***")
        eval_results = trainer.evaluate(eval_dataset=test_interpolate_dataset)
        logger.info(f"eval_results interpolate data: {eval_results}")

        logger.info("*** Evaluate on extrapolate data ***")
        eval_results = trainer.evaluate(eval_dataset=test_extrapolate_dataset)
        logger.info(f"eval_results extrapolate data: {eval_results}")


if __name__ == "__main__":
    main()