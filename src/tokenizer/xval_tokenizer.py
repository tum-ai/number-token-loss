import itertools
import logging
import re
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import torch
from torch import TensorType
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding, TextInput, TextInputPair, \
    PreTokenizedInput, PreTokenizedInputPair, EncodedInput, EncodedInputPair
from transformers.utils import PaddingStrategy

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer, NUMBER_REGEX


class XvalTokenizer(NumberEncodingTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_token = "[NUM]"
        self.add_tokens([num_token])
        self.num_token = num_token
        self.num_token_id = self.convert_tokens_to_ids(num_token)
        self.model_input_names.append("number_embeddings")

    def get_num_token_ids(self):
        return [self.num_token_id]

    def get_num_tokens(self):
        return [self.num_token]

    def decode_number_token(self, token, number: float = None):
        return number

    def decode_into_human_readable(
            self,
            ids: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
            numbers: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"] = None
    ) -> Tuple[List[str], int, int]:
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        if isinstance(numbers, torch.Tensor):
            numbers = numbers.cpu().numpy()

        numbers = list(map(lambda x: list(map(lambda y: self.tokenize(str(y)), x)), numbers))
        decoded_ids = np.array(list(map(lambda sample: self.convert_ids_to_tokens(sample), ids)))
        count_no_number_prediction_at_all = np.sum(np.all(ids != 32100, axis=1))

        def replace_number_tokens_with_numbers(id, number, decoded_id):
            return number if id in self.get_num_token_ids() else decoded_id

        def flatten(lst):
            flat_list = []
            for item in lst:
                if isinstance(item, list):
                    flat_list.extend(flatten(item))
                else:
                    flat_list.append(item)
            return flat_list

        decoded_ids = [
            list(map(lambda id, number, decoded_id: replace_number_tokens_with_numbers(id, number, decoded_id), ids_row,
                     numbers_row, decoded_ids_row))
            for ids_row, numbers_row, decoded_ids_row in zip(ids, numbers, decoded_ids)
        ]
        decoded_ids = list(map(flatten, decoded_ids))

        # Remove padding tokens
        decoded_ids = [list(filter(lambda x: x not in self.all_special_tokens, decoded_id)) for decoded_id in
                       decoded_ids]

        try:
            decoded_ids = list(
                map(lambda sample: self.convert_tokens_to_string(sample) if len(sample) else "", decoded_ids))
        except Exception as e:
            logging.error(f"Error converting tokens to string: {e} for tokens {decoded_ids}")
            decoded_ids = ["" for _ in range(len(decoded_ids))]

        decoded_ids = list(
            map(lambda sample: self.convert_tokens_to_string(sample) if len(sample) else "", decoded_ids))

        return decoded_ids, 0, count_no_number_prediction_at_all

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                ############################
                # Custom Code Start
                ############################
                nonum_text, numbers = extract(text, num_token=self.num_token)
                tokens = self.tokenize(nonum_text, **kwargs)
                ############################
                # Custom Code End
                ############################
                return self.convert_tokens_to_ids(tokens), numbers
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                tokens = []
                numbers = []
                if is_split_into_words:
                    for t in text:
                        ############################
                        # Custom Code Start
                        ############################
                        nonum_text, number_text = extract(t, num_token=self.num_token)
                        tokenized_text = self.tokenize(nonum_text, is_split_into_words=True, **kwargs)
                        tokens.extend(tokenized_text)
                        numbers.extend(number_text)
                        ############################
                        # Custom Code End
                        ############################
                    return self.convert_tokens_to_ids(tokens), numbers
                else:
                    raise ValueError("Not supported in our implementation")
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                raise ValueError("Not supported in our implementation 2")
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when"
                        " `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids, first_numbers = get_input_ids(text)
        second_ids, second_numbers = get_input_ids(text_pair) if text_pair is not None else None, None

        return self.prepare_for_model(
            first_ids,
            first_numbers=first_numbers,
            pair_ids=second_ids,
            pair_numbers=second_numbers,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                ############################
                # Custom Code Start
                ############################
                nonum_text, numbers = extract(text, num_token=self.num_token)
                tokens = self.tokenize(nonum_text, **kwargs)
                ############################
                # Custom Code End
                ############################
                return self.convert_tokens_to_ids(tokens), numbers
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                tokens = []
                numbers = []
                if is_split_into_words:
                    for t in text:
                        ############################
                        # Custom Code Start
                        ############################
                        nonum_text, number_text = extract(t, num_token=self.num_token)
                        tokenized_text = self.tokenize(nonum_text, is_split_into_words=True, **kwargs)
                        tokens.extend(tokenized_text)
                        numbers.extend(number_text)
                        ############################
                        # Custom Code End
                        ############################
                    return self.convert_tokens_to_ids(tokens), numbers
                else:
                    raise ValueError("Not supported in our implementation")
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                raise ValueError("Not supported in our implementation 2")
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        input_numbers = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids, first_numbers = get_input_ids(ids)
            second_ids, second_numbers = get_input_ids(pair_ids) if pair_ids is not None else None, None
            input_ids.append((first_ids, second_ids))
            input_numbers.append((first_numbers, second_numbers))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            input_numbers,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    def _batch_prepare_for_model(
            self,
            batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
            batch_number_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
            verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for ((first_ids, second_ids), (first_numbers, second_numbers)) in zip(batch_ids_pairs, batch_number_pairs):
            outputs = self.prepare_for_model(
                first_ids,
                first_numbers=first_numbers,
                pair_ids=second_ids,
                pair_numbers=second_numbers,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def prepare_for_model(
            self,
            ids: List[int],
            first_numbers: List[int],
            pair_ids: Optional[List[int]] = None,
            pair_numbers: List[int] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence

        ############################
        # Custom Code Start
        ############################
        number_locs = np.array(sequence) == self.num_token_id
        num_embed = np.ones(len(sequence)).astype(np.float32)  # Use float32 instead of float16

        if num_embed[number_locs].shape[0] < len(first_numbers):
            # trunctuate first numbers
            first_numbers = first_numbers[:len(num_embed[number_locs])]
        
        num_embed[number_locs] = first_numbers + pair_numbers if pair_numbers else first_numbers
        encoded_inputs["number_embeddings"] = num_embed.tolist()
        ############################
        # Custom Code End
        ############################

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                            encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                ############################
                # Custom Code Start
                ############################
                encoded_inputs["number_embeddings"] = encoded_inputs["number_embeddings"] + [1] * difference
                ############################
                # Custom Code End
                ############################
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
                ############################
                # Custom Code Start
                ############################
                encoded_inputs["number_embeddings"] = [1] * difference + encoded_inputs["number_embeddings"]
                ############################
                # Custom Code End
                ############################
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs


def extract(text, num_token="[NUM]"):
    import re

    # this regular expression is intended to match numerical values in various forms
    # like integers, floating-point numbers, or scientific notation, while avoiding
    # matching numbers that are part of strings.

    numbers = []

    def replace(match):
        numbers.append(match.group())
        return "¬"

    nonum_text = re.sub(NUMBER_REGEX, replace, text)
    return compress_matrix(nonum_text).replace("¬", num_token), numbers


def compress_matrix(text):
    text = (
        text.replace("¬, ¬", "¬¬")
        .replace("¬, ¬", "¬¬")
        .replace("¬,¬", "¬¬")
        .replace("¬,¬", "¬¬")
    )
    return text
