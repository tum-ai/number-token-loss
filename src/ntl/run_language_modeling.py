#!/usr/bin/env python3
"""
Language modeling adapted from Huggingface transformers.

The file is an adaptation of https://github.com/huggingface/transformers/blob/v3.1.0/examples/language-modeling/run_language_modeling.py

"""
import sys
import os
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    set_seed,
    EarlyStoppingCallback, T5ForConditionalGeneration, T5ForSequenceClassification
)
import hydra
from omegaconf import DictConfig, OmegaConf

from ntl.trainer import CustomSeq2SeqTrainer
from ntl.evaluation import CustomMetrics
from ntl.args import ModelArguments, TrainingArguments, DatasetArguments
from ntl.data.data import load_txt_dataset, load_json_dataset
from ntl.collators.question_answer_clm.xval_question_answer_collator import XvalQuestionAnswerCLMCollator
from ntl.collators.question_answer_clm.vanilla_question_answer_collator import VanillaQuestionAnswerCLMCollator
from ntl.collators.question_answer_mlm.regression_head_question_answer_collator import RegressionHeadQuestionAnswerCollator
from ntl.collators.question_answer_mlm.xval_mask_question_collator import XvalMaskedQuestionAnswerCollator
from ntl.collators.question_answer_mlm.vanilla_mlm_question_answer_collator import VanillaMaskedQuestionAnswerCollator
from ntl.transformer_backbone.t5.t5_vanilla_for_number_token_loss import T5VanillaForNumberTokenLoss
from ntl.transformer_backbone.t5.t5_rt import T5RegressionModelRT
from ntl.transformer_backbone.t5.t5_xval import T5RegressionModelXval
from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from ntl.tokenizer.rt_tokenizer import RtTokenizer
from ntl.tokenizer.xval_tokenizer import XvalTokenizer
from ntl.loss_functions.number_token_loss import NumberTokenLoss
from ntl.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss

from ntl.loss_functions.number_token_loss import NumberTokenSelector
from ntl.utils.label_smoother import GaussianLabelSmoother


transformers.logging.set_verbosity_info()
logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    os.environ["COMET_MODE"] = "DISABLED"

    store_config(cfg)

    model_args = ModelArguments(**cfg.model_args)
    training_args = TrainingArguments(**cfg.training_args)
    dataset_args = DatasetArguments(**cfg.dataset_args)

    run_language_modeling(model_args, training_args, dataset_args)


def run_language_modeling(model_args: ModelArguments, training_args: TrainingArguments, dataset_args: DatasetArguments):
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

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

    logger.info("Training on dataset: %s", dataset_args.dataset_name)
    logger.info("Training/evaluation parameters %s", training_args)

    if model_args.number_encoding != "xval":
        training_args.generation_num_beams = 4
        logger.info("Setting generation_num_beams to 4")
    else:
        logger.info("Setting generation_num_beams to 1 for xval")

    if model_args.log_scale_embeddings:
        if model_args.number_encoding in ["xval", "rt", "none_regression_head"]:
            logger.info("Log scaling embeddings")
        else:
            raise ValueError("Log scaling only supported for xval, rt and none_regression_head")
    else:
        if model_args.number_encoding in ["xval", "rt", "none_regression_head"]:
            logger.info("Not log scaling embeddings")

    if model_args.xval_bigger_language_head:
        if model_args.number_encoding == "xval":
            logger.info("Using bigger language head for xval")
        else:
            raise ValueError("Bigger language head only supported for xval")

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

    if training_args.language_modelling == "clm":
        # Set generation arguments
        training_args.predict_with_generate = True
        logger.info("Setting predict_with_generate to True")
    else:
        training_args.predict_with_generate = False
        logger.info("Setting predict_with_generate to False")

    if model_args.number_encoding == "rt":
        model_class = T5RegressionModelRT
        tokenizer_class = RtTokenizer
    elif model_args.number_encoding == "xval":
        model_class = T5RegressionModelXval
        tokenizer_class = XvalTokenizer
    elif model_args.number_encoding.lower() == "none":
        if model_args.number_token_loss or model_args.gaussian_label_smoother:
            model_class = T5VanillaForNumberTokenLoss
            tokenizer_class = T5Custom_Tokenizer
        else:
            model_class = T5ForConditionalGeneration
            tokenizer_class = transformers.AutoTokenizer
    elif model_args.number_encoding.lower() == "none_regression_head":
        config.num_labels = 1
        model_class = T5ForSequenceClassification
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
        pad_to_multiple_of = 64 if torch.cuda.is_available() else 1
        config.vocab_size = len(tokenizer) + pad_to_multiple_of - (len(tokenizer) % pad_to_multiple_of)
        config.added_vocab = tokenizer.get_added_vocab()
        logger.info("Size of new tokenizer: %s", len(tokenizer))
        logger.info(f"New vocab size: {config.vocab_size}")

    if model_args.number_encoding == "xval":
        model_init_kwargs = {"tokenizer": tokenizer, "bigger_language_head": model_args.xval_bigger_language_head}
    else:
        model_init_kwargs = {}

    if model_args.number_encoding in ["rt", "xval"]:
        model_init_kwargs["log_scale_embeddings"] = model_args.log_scale_embeddings

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
            raise ValueError(
                f"Unknown number_token_loss_function: {model_args.number_token_loss_function}. Allowed: mse, huber, mae.")

        if model_args.number_token_loss_with_wasserstein:
            logger.info("Using Wasserstein distance for number token loss")
            model_init_kwargs["number_token_loss"] = WassersteinNumberTokenLoss(
                tokenizer,
                vocab_size=config.vocab_size,
                device=training_args.device,
                order_numbers=model_args.number_encoding != "rt",
                loss_function=loss_function,
                weight=model_args.number_token_loss_weight
            )
        else:
            logger.info("Using normal number token loss")
            model_init_kwargs["number_token_loss"] = NumberTokenLoss(
                tokenizer,
                vocab_size=config.vocab_size,
                device=training_args.device,
                loss_function=loss_function,
                weight=model_args.number_token_loss_weight
            )

    if model_args.gaussian_label_smoother: 
        selector = NumberTokenSelector(tokenizer, vocab_size=config.vocab_size, device=training_args.device) 
        label_smoother = GaussianLabelSmoother(
            sigma=model_args.label_smoother_sigma,           
            ignore_index=-100,   
            selector=selector    
        )
    else: 
        label_smoother = None

    if model_args.model_name_or_path:

        # delete generation config, as not deleting leads to model not being loadable
        if os.path.isdir(model_args.model_name_or_path) and os.path.exists(
                os.path.join(model_args.model_name_or_path, "generation_config.json")):
            os.remove(os.path.join(model_args.model_name_or_path, "generation_config.json"))

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            ignore_mismatched_sizes=True,
            **model_init_kwargs,
        )

        if model_args.number_encoding == "xval" and "checkpoint" not in model_args.model_name_or_path:
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

    if dataset_args.dataset_name == "gsm8k":
        if dataset_args.train_with_augmented_data:
            train_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/train_t_clean_with_augmented.jsonl'
        else:
            train_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/train_t_clean.jsonl'
        eval_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/val_t_clean.jsonl'
        test_data_path = 'data/grade-school-math/grade_school_math/data/preprocessed/test_clean.jsonl'
        train_dataset = load_json_dataset(train_data_path)
        eval_dataset = load_json_dataset(eval_data_path)
        test_dataset = load_json_dataset(test_data_path)
    elif dataset_args.dataset_name == "mathematics_dataset":
        train_data_path = 'data/mathematics_dataset-v1.0/train.txt'
        eval_data_path = 'data/mathematics_dataset-v1.0/val.txt'
        test_interpolate_data_path = 'data/mathematics_dataset-v1.0/test_interpolate.txt'
        test_extrapolate_data_path = 'data/mathematics_dataset-v1.0/test_extrapolate.txt'

        train_dataset = load_txt_dataset(train_data_path)
        eval_dataset = load_txt_dataset(eval_data_path)
        test_interpolate_dataset = load_txt_dataset(test_interpolate_data_path)
        test_extrapolate_dataset = load_txt_dataset(test_extrapolate_data_path)
    elif dataset_args.dataset_name == "arithmetic":
        logger.info("Training on arithmetic dataset")
        train_data_path = 'data/mathematics_dataset-v1.0/arithmetic_train.txt'
        eval_data_path = 'data/mathematics_dataset-v1.0/arithmetic_val.txt'
        test_interpolate_data_path = 'data/mathematics_dataset-v1.0/arithmetic_test_interpolate.txt'
        test_extrapolate_data_path = 'data/mathematics_dataset-v1.0/arithmetic_test_extrapolate.txt'

        train_dataset = load_txt_dataset(train_data_path)
        eval_dataset = load_txt_dataset(eval_data_path)
        test_interpolate_dataset = load_txt_dataset(test_interpolate_data_path)
        test_extrapolate_dataset = load_txt_dataset(test_extrapolate_data_path)
    elif dataset_args.dataset_name == "multiplication":
        train_data_path = 'data/digit-multiplication/data/train.jsonl'
        eval_data_path = 'data/digit-multiplication/data/val.jsonl'
        test_data_path = 'data/digit-multiplication/data/test.jsonl'
        train_dataset = load_json_dataset(train_data_path)
        eval_dataset = load_json_dataset(eval_data_path)
        test_dataset = load_json_dataset(test_data_path)
    elif dataset_args.dataset_name == "multirc":
        train_data_path = 'data/multirc/data/preprocessed/train_clean.jsonl'
        eval_data_path = 'data/multirc/data/preprocessed/val_clean.jsonl'
        test_data_path = 'data/multirc/data/preprocessed/test_clean.jsonl'
        train_dataset = load_json_dataset(train_data_path)
        eval_dataset = load_json_dataset(eval_data_path)
        test_dataset = load_json_dataset(test_data_path)
    elif dataset_args.dataset_name == "debug":
        train_data_path = 'data/mathematics_dataset-v1.0/mathematics_dataset-v1.0/train-easy/algebra__linear_1d_small.txt'
        eval_data_path = 'data/mathematics_dataset-v1.0/mathematics_dataset-v1.0/train-easy/algebra__linear_1d_small.txt'
        test_data_path = 'data/mathematics_dataset-v1.0/mathematics_dataset-v1.0/train-easy/algebra__linear_1d_small.txt'
        train_dataset = load_txt_dataset(train_data_path)
        eval_dataset = load_txt_dataset(eval_data_path)
        test_dataset = load_txt_dataset(test_data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_args.dataset_name}. Allowed: gsm8k, mathematics_dataset, multiplication")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters {num_params} of type {type(model)}")

    data_collator = get_data_collator(model_args, tokenizer, training_args)

    # Custom Metric
    custom_metrics = CustomMetrics(
        tokenizer=tokenizer,
        number_encoding=model_args.number_encoding,
        output_dir=training_args.output_dir,
        save_all_output=True,
        log_scale=model_args.log_scale_embeddings,
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
        label_smoother=label_smoother,
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
        start_time = time.time()
        trainer.train(model_path=model_path)
        end_time=time.time()
        logger.info("Elapsed time:")
        logger.info(end_time - start_time)
    else:
        logger.info("Skipping training.")

    if dataset_args.mode != "dataset_comparison":
        logger.info("*** Evaluate on validation data ***")
        eval_results_val = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"eval_results validation data: {eval_results_val}")

        if dataset_args.dataset_name in ["arithmetic", "mathematics_dataset"]:
            logger.info("*** Evaluate on interpolation data for arithmetic ***")
            eval_results_test_interpolate = trainer.evaluate(eval_dataset=test_interpolate_dataset)
            logger.info(f"eval_results interpolate data: {eval_results_test_interpolate}")

            logger.info("*** Evaluate on extrapolation data for arithmetic ***")
            eval_results_test_extrapolate = trainer.evaluate(eval_dataset=test_extrapolate_dataset)
            logger.info(f"eval_results extrapolate data: {eval_results_test_extrapolate}")

            return eval_results_val, eval_results_test_interpolate, eval_results_test_extrapolate, model
        else:
            logger.info("*** Evaluate on test set ***")
            eval_results_test = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"eval_results test data: {eval_results_test}")
            return eval_results_val, eval_results_test, model


    else:
        if dataset_args.dataset_name != "mathematics_dataset":
            raise ValueError("dataset_args.mode=dataset_comparison only supported for mathematics_dataset")

        logger.info("*** Comparing loss on individuals datasets ***")

        # interpolation
        int_categories = [
            "algebra__linear_1d.txt",
            "algebra__linear_1d_composed.txt",
            "algebra__linear_2d.txt",
            "algebra__linear_2d_composed.txt",
            "algebra__sequence_next_term.txt",
            "arithmetic__add_or_sub.txt",
            "arithmetic__add_sub_multiple.txt",
            "arithmetic__mul.txt",
            "numbers__div_remainder.txt",
            "numbers__div_remainder_composed.txt",
            "numbers__place_value.txt",
            "numbers__round_number.txt",
            "numbers__round_number_composed.txt",
        ]
        int_results = []
        for name in int_categories:
            path = 'data/mathematics_dataset-v1.0/mathematics_dataset-v1.0/interpolate/' + name
            test_dataset = load_txt_dataset(path)
            logger.info("*** Testing interpolation on " + name + " data ***")
            result = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"Test results: {result}")
            int_results.append(result)

        #extrapolation
        ext_categories = [
        "arithmetic__add_or_sub_big.txt",
        "arithmetic__add_sub_multiple_longer.txt",
        "arithmetic__mixed_longer.txt",
        "arithmetic__mul_big.txt",
        "arithmetic__mul_div_multiple_longer.txt",
        "numbers__place_value_big.txt",
        "numbers__round_number_big.txt",
        ]
        ext_results = []
        for name in ext_categories:
            path = 'data/mathematics_dataset-v1.0/mathematics_dataset-v1.0/extrapolate/' + name
            test_dataset = load_txt_dataset(path)
            logger.info("*** Testing extrapolation on " + name + " data ***")
            result = trainer.evaluate(eval_dataset=test_dataset)
            logger.info(f"Test results: {result}")
            ext_results.append(result)

        return int_results, ext_results


def get_data_collator(model_args: ModelArguments, tokenizer, training_args: TrainingArguments) -> transformers.DataCollator:
    # Conditional Generation Training
    if training_args.language_modelling == "clm":
        logger.info("Using CLM collator")
        if model_args.number_encoding == "rt":
            data_collator = VanillaQuestionAnswerCLMCollator(
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
        elif model_args.number_encoding.lower() == "none_regression_head":
            raise ValueError("Regression head not supported for CLM")

    # Masked Language Modeling
    else:
        logger.info("Using MLM collator")
        if model_args.number_encoding == "rt":
            data_collator = VanillaMaskedQuestionAnswerCollator(
                tokenizer=tokenizer
            )
        elif model_args.number_encoding == "xval":
            data_collator = XvalMaskedQuestionAnswerCollator(
                tokenizer=tokenizer
            )
        elif model_args.number_encoding.lower() == "none":
            # Rt collator can be used for default T5 as well
            data_collator = VanillaMaskedQuestionAnswerCollator(
                tokenizer=tokenizer
            )
        elif model_args.number_encoding.lower() == "none_regression_head":
            data_collator = RegressionHeadQuestionAnswerCollator(
                tokenizer=tokenizer, log_scale=model_args.log_scale_embeddings
            )
    return data_collator


def store_config(cfg: DictConfig):
    output_dir = cfg.training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


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


if __name__ == "__main__":
    main()
