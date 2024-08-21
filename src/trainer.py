from typing import Dict, Optional, Union, List, Any, Tuple

import torch
from torch import nn
from transformers import Seq2SeqTrainer
from transformers.integrations import is_deepspeed_zero3_enabled


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer class that inherits from transformers' Seq2SeqTrainer.
    It overrides the prediction_step method in order to
        - additionally return next_token_prediction_logits for calculating perplexity
        - also handle the number_predictions from xval as they are an additional output of the model
    """

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        :return: Tuple of (next_token_prediction_loss, (next_token_prediction_logits, generated_tokens), labels)
        Where next_token_prediction_loss is the loss of the next token prediction,
        next_token_prediction_logits are the logits of the next token prediction,
        generated_tokens are the generated tokens, unless for xval where the generated tokens is a tuple of (generated_tokens, generated_numbers),
        labels are the token labels, unless for xval where the labels is a tuple of (token_labels, number_labels)
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        ########################
        # Customized code start
        ########################
        # set the max_length to the length of the labels + 10 to ensure that the model can generate the full sequence
        self.model.generation_config.max_length = inputs["labels"].shape[-1] + 10 if "labels" in inputs else self.model.generation_config.max_length
        ########################
        # Customized code end
        ########################

        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False


        ########################
        # Customized code start
        ########################
        # Check if the model returns a tuple - if yes, the second element is the number prediction from xval
        if isinstance(generated_tokens, tuple):
            # only true for xval
            generated_tokens, generated_numbers = generated_tokens
        else:
            generated_numbers = None

        # remove the first padding token
        generated_tokens = generated_tokens[:, 1:]

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        # if xval number predictions, we have to handle them exactly like the generated tokens
        if generated_numbers is not None:
            generated_numbers = generated_numbers[:, 1:]
            if generated_numbers.shape[-1] < gen_config.max_length:
                generated_numbers = self._pad_numbers_to_max_len(generated_numbers, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and generated_numbers.shape[-1] < gen_config.max_new_tokens + 1:
                generated_numbers = self._pad_numbers_to_max_len(generated_numbers, gen_config.max_new_tokens + 1)
            generated_tokens = (generated_tokens, generated_numbers)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                    next_token_prediction_logits = outputs.logits
                if self.label_smoother is not None:
                    next_token_prediction_loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    next_token_prediction_loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                next_token_prediction_loss = None

        if self.args.prediction_loss_only:
            return next_token_prediction_loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
            # if number labels, add them to the label output
            if "number_labels" in inputs:
                number_labels = inputs["number_labels"]
                if number_labels.shape[-1] < gen_config.max_length:
                    number_labels = self._pad_numbers_to_max_len(number_labels, gen_config.max_length)
                elif gen_config.max_new_tokens is not None and number_labels.shape[-1] < gen_config.max_new_tokens + 1:
                    number_labels = self._pad_numbers_to_max_len(number_labels, gen_config.max_new_tokens + 1)
                labels = (labels, number_labels)
        else:
            labels = None


        return next_token_prediction_loss, (next_token_prediction_logits, generated_tokens), labels

    def _pad_numbers_to_max_len(self, number_tensor, max_length):
        """
        Pad a sequence of numbers to max_length with padding number of 1.
        :param number_tensor: The tensor of numbers to pad
        :param max_length: The maximum length to pad to.
        :return: The padded tensor.
        """
        pad_number = 1

        padded_tensor = pad_number * torch.ones(
            (number_tensor.shape[0], max_length), dtype=number_tensor.dtype, device=number_tensor.device
        )
        padded_tensor[:, : number_tensor.shape[-1]] = number_tensor
        return padded_tensor

    ########################
    # Customized code end
    ########################