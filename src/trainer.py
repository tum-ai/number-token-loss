"""
Taken from https://github.com/huggingface/transformers/blob/v4.33.3/src/transformers/trainer.py
"""

import collections
import gc
import json
import os
import shutil
import warnings
from random import random
from time import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from .encoding_decoding.numerical_encodings import FloatEncoding, IntEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)

NON_MODEL_KEYS = ["real_property", "sample_weights"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENCODING_FACTORY = {"float": FloatEncoding, "int": IntEncoding}

MODEL_TO_EMBEDDING_FN = {
    "albert": "model.albert.embeddings",
    "xlnet": "self.model.transformer.word_embedding",
}


def get_trainer_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to take out a subset of a dictionary with keys that are
    important for `CustomTrainer` but cant be passed down to `Trainer`.

    Args:
        dictionary (dict): Dict with keyword arguments for `CustomTrainer` constructor.

    Returns:
        dict: Dict with keyword arguments for `CustomTrainer` that cant be passed to
            childclass constructor (`Trainer`).
    """
    keys_to_keep = [
        "verbose_evaluation",
        "numerical",
        "d_model",
        "vocab_size",
        "vmax",
        "model_type",
        "mem_len",
        "training_logs",
        "train_config",
        "alternating_collator",
    ]
    keep_dict = {}
    for keep_key in keys_to_keep:
        for key, val in dictionary.items():
            if re.search(keep_key, key) is not None:
                keep_dict[key] = val
    return keep_dict


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):

        # logger.info(f'ARGS are\n{kwargs}\n{args}')
        # Remove keyword arguments unwanted by parent class
        child_kwargs = get_trainer_dict(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k not in child_kwargs}

        # Call parent class constructor
        super().__init__(*args, **kwargs)

        # Extract custom arguments
        self.verbose_evaluation = child_kwargs.get("verbose_evaluation", True)
        logger.info(f"Verbose evaluation {self.verbose_evaluation}")

        # Restore the logged parameters (training)
        self.logs = child_kwargs.get("training_logs", [])
        self.eval_logs = []

        # Will safe RMSE and Pearson of every epoch
        try:
            tokens = self.data_collator.property_tokens
        except AttributeError:
            tokens = [None]

        if self.logs != []:
            self.min_loss = pd.DataFrame(self.logs)["loss"].min()
            if child_kwargs.get("train_config", {}).get("reset_training_loss", False):
                self.min_loss = 10e5
            logger.info(f"Current minimal loss {self.min_loss}")
        else:
            self.min_loss = 10e5

        self.use_numerical_encodings = child_kwargs.get(
            "use_numerical_encodings", False
        )

        if self.use_numerical_encodings:
            logger.info("Attempting to use numerical encodings.")
            self.numerical_encodings_type = child_kwargs.get(
                "numerical_encodings_type", "float"
            )
            self.numerical_encodings_format = child_kwargs.get(
                "numerical_encodings_format", "sum"
            )
            self.numerical_encodings_dim = child_kwargs.get(
                "numerical_encodings_dim", 16
            )

            if self.numerical_encodings_format == "concat":

                if self.numerical_encodings_dim > child_kwargs["d_model"]:
                    raise ValueError(
                        "Numerical encoding size cant be bigger than embedding size"
                    )

                self.combine_embed = self.overwrite_embed

            elif self.numerical_encodings_format == "sum":
                self.numerical_encodings_dim = child_kwargs["d_model"]

                self.combine_embed = self.sum_embed

            else:
                raise ValueError(
                    f"Unknown float encoding format {self.numerical_encodings_format}."
                )

            self.numerical_encoder = NUM_ENCODING_FACTORY[
                self.numerical_encodings_type
            ](
                num_embeddings=child_kwargs["vocab_size"],
                embedding_dim=self.numerical_encodings_dim,
                vocab=self.tokenizer.vocab,
                vmax=child_kwargs.get("vmax", None),
            )

            self.model_embed = eval(
                MODEL_TO_EMBEDDING_FN[child_kwargs.get("model_type", "xlnet")]
            )

        # self.search = SEARCH_FACTORY[child_kwargs.get("eval_search", "greedy")](
        #     child_kwargs.get("eval_search_args", {})
        # )
        self.save_attention = child_kwargs.get("save_attention", False)

    def sum_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        return e + num_e

    def overwrite_embed(self, e: torch.Tensor, num_e: torch.Tensor) -> torch.Tensor:
        e[:, :, -self.numerical_encodings_dim:] = num_e
        return e

    def save_attention(self, inputs: torch.Tensor, attention: torch.Tensor):
        """
        Save the attention weights for the current batch.

        Args:
            inputs (torch.Tensor): input_ids
            attention (torch.Tensor): attention tensor

        """

        for idx, a in enumerate(attention):
            for i, aa in enumerate(a):
                np.save(
                    f"batch_{self.counter}_layer_{idx}_tup_{i}", aa.detach().numpy()
                )

        for i, inp in enumerate(inputs):
            tokens = self.tokenizer.convert_ids_to_tokens(inp.tolist())
            with open(f"batch_{self.counter}_sample_{i}.txt", "w") as f:
                f.write(str(tokens))
        self.counter += 1

    def feed_model(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Forward pass of `inputs` through `model`. This function handles the numerical
        encodings if applicable.

        Args:
            model (nn.Module): The model to consume data.
            inputs (Dict[str, Union[torch.Tensor, Any]]): A dict that can be understood
                by model.__call__. Keys should include `input_ids`, `perm_mask`,
                `labels` and `target_mapping`.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Output from model
        """
        model_inputs = inputs  # shallow copy

        if self.use_numerical_encodings:
            model_inputs = inputs.copy()
            # Pop keys unused by model
            [model_inputs.pop(k, None) for k in NON_MODEL_KEYS]
            embeddings = self.model_embed(inputs["input_ids"])
            numerical_embeddings = self.numerical_encoder(inputs["input_ids"])
            embeddings = self.combine_embed(embeddings, numerical_embeddings)
            model_inputs.pop("input_ids", None)

            if not self.save_attention:
                outputs = model(inputs_embeds=embeddings, **model_inputs)
            else:
                # Attention config
                outputs = model(
                    inputs_embeds=embeddings,
                    **model_inputs,
                    output_attentions=True,
                    output_hidden_states=False,
                )
                self.save_attention(inputs["input_ids"], outputs[-1])

        else:
            [model_inputs.pop(k, None) for k in NON_MODEL_KEYS]
            outputs = model(**model_inputs)

        return outputs

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            gnore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        NOTE: Overwritten here to enable custom embeddings + for moinitoring purposes.

        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """

        has_labels = any(
            inputs.get(k) is not None
            for k in ["labels", "lm_labels", "masked_lm_labels"]
        )

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # NOTE: Overwritten with custom embeddings
            outputs = self.feed_model(model, inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().detach()
            else:
                loss = None
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[
                    self.args.past_index if has_labels else self.args.past_index - 1
                ]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        logits = logits.detach()

        # # NOTE: Overwritten for moinitoring purposes (will print occassionally)
        # if self.verbose_evaluation and random() < 0.00001:

        #     try:
        #         # TODO: Only fill the masked tokens
        #         prediction = (
        #             self.search(logits[1, :, :].unsqueeze(0))
        #             .detach()
        #             .cpu()
        #             .squeeze()
        #             .tolist()
        #         )
        #         gt_seq, gt_dict = self.tokenizer.aggregate_tokens(
        #             self.tokenizer.get_sample_label(
        #                 self.tokenizer.convert_ids_to_tokens(labels[0]),
        #                 self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        #             ),
        #             label_mode=True,
        #         )

        #         p_seq, p_dict = self.tokenizer.aggregate_tokens(
        #             self.tokenizer.convert_ids_to_tokens(prediction), label_mode=False
        #         )

        #         logger.info(f"\nPredicted: {p_seq} \t, {p_dict.get('qed', -1)}")
        #         logger.info(f"Ground truth {gt_seq} \t {gt_dict.get('qed', -1)}")
        #     except Exception:
        #         logger.info("Error occurred in converting logits to sequence.")

        return loss, logits, labels

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        NOTE: Overwritten to call custom feed_model method for own embedding methods.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # NOTE: Overwritten to maintain custom embeddings and alternative losses.
        outputs = self.feed_model(model, inputs)
        loss = outputs[0]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        NOTE: Overwritten to save best model alongside some metrics

        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        super().log(logs)

        if "eval_loss" in logs.keys():
            logger.info(f"Evaluation {logs}")
            self.eval_logs.append({"eval_loss": logs["eval_loss"]})
            if "epoch" in logs.keys():
                self.eval_logs[-1].update(
                    {"epoch": logs["epoch"], "step": self.global_step}
                )

        # Custom logging
        if "loss" in logs.keys():
            # In case of training logging
            if self.epoch is not None:
                logs["epoch"] = self.epoch
                output = {**logs, **{"step": self.global_step}}
                self.logs.append(output)

            # Save new best model
            if logs["loss"] < self.min_loss:
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-best-{self.global_step}"
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.min_loss = logs["loss"]
                self.save_model(output_dir)
                # Save optimizer and scheduler
                if self.is_world_master():
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                pd.DataFrame(self.logs).to_csv(
                    os.path.join(output_dir, "training_log.csv")
                )
                pd.DataFrame(self.eval_logs).to_csv(
                    os.path.join(output_dir, "eval_log.csv")
                )

                checkpoint_prefix = f"{PREFIX_CHECKPOINT_DIR}-best"

                if self.is_world_process_zero():
                    self._rotate_checkpoints(prefix=checkpoint_prefix)

    def _rotate_checkpoints(
            self, use_mtime: bool = False, prefix: str = PREFIX_CHECKPOINT_DIR
    ) -> None:
        """NOTE: Overwritten to enable passing down checkpoint prefix for deletion."""
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime, checkpoint_prefix=prefix
        )

        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - self.args.save_total_limit
        )
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                    checkpoint
                )
            )
            shutil.rmtree(checkpoint)
