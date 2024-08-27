from typing import List, Tuple, Dict
import math
import functools
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number
from src.utils.helper_functionality import print_structure, write_debug_log
import evaluate
import nltk

PADDING_TOKEN = -100
MASKED_OUT = -1
MALFORMED_RT_TOKEN = 100000
NON_NUMERIC_TOKEN = 10000
SURELY_NUMERIC_TOKEN_BOUND = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_token_to_check_validity(token: str) -> float:
    """
    Validates a token and extracts its numeric value if valid. Uses number encoding to allow usage with numpy
    
    Args:
        token (str): The token to be validated and converted.
    
    Returns:
        float: The extracted numeric value if the token is valid, otherwise a predefined error code.
    """
    if token.startswith("_") and token.endswith("_"):
        parts = token.strip("_").split("_")
        if len(parts) == 2 and parts[0].isdigit():
            return int(parts[1])
        else:
            return MALFORMED_RT_TOKEN
    else:
        return NON_NUMERIC_TOKEN


def is_valid_numeric_token_sequence(validation_array: np.ndarray) -> bool:
    """
    Checks if the validation array contains a valid decreasing sequence of numeric tokens.
    
    Args:
        validation_array (np.ndarray): Array of numeric values derived from tokens.
    
    Returns:
        bool: True if the array contains a valid decreasing sequence, otherwise False.
    """
    numeric_token_indices = np.where(validation_array < SURELY_NUMERIC_TOKEN_BOUND)[0]
    if len(numeric_token_indices) == 0:
        return False
    # Numeric tokens are a) consequitive without interruption, b) encoded position form decreasing sequence
    valid_sequence = np.all(np.diff(numeric_token_indices) == 1) and np.all(
        np.diff(validation_array[numeric_token_indices]) == -1)
    return valid_sequence


def convert_tokens_to_num_rt(token_array: np.ndarray) -> np.ndarray:
    """
    Converts an array of tokens into numeric values, checking for validity and applying transformations.
    
    Args:
        token_array (np.ndarray): Array of tokens to be converted.
    
    Returns:
        tuple: A tuple containing:
            - A numpy array with the numeric values or NaNs for invalid sequences.
            - A boolean mask indicating which rows contain valid sequences.
    """
    validation_array = np.vectorize(convert_token_to_check_validity)(token_array)

    all_tokens_valid_per_row = np.all(validation_array != MALFORMED_RT_TOKEN, axis=1)
    valid_numeric_token_sequence_per_row = np.apply_along_axis(is_valid_numeric_token_sequence, 1, validation_array)
    valid_mask = all_tokens_valid_per_row & valid_numeric_token_sequence_per_row

    # We use a trick by converting non numeric tokens to zero and assuming there is only one numeric token
    float_array = np.vectorize(functools.partial(encoding_to_number, invalid_strict=False))(token_array)
    value_array = np.sum(float_array, axis=1)

    # Currently, this leaves multiple [NEG] predictions a possibility. Fixing not trivial. Maybe apply simplification rules like https://github.com/PolymathicAI/xVal/blob/main/xval/preprocess.py
    contains_neg = np.any(token_array == "[NEG]", axis=1)

    value_array = np.where(contains_neg, -value_array, value_array)

    value_array_with_nans = np.where(valid_mask, value_array, np.nan)

    return value_array_with_nans, valid_mask


class CustomMetrics:
    """
    Compute custom metrics for the model with access to the vocab to compute MSE
    """

    def __init__(self, tokenizer: NumberEncodingTokenizer, number_encoding: str):
        self.tokenizer = tokenizer
        self.index_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        self.number_encoding = number_encoding
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("sacrebleu")
        nltk.download('punkt_tab')

        if self.number_encoding == "none":
            # ▁ is necessary as T5 Tokenizes white spaces like this and it has tokens for 1 and ▁1
            self.numeric_token_pattern = re.compile(r"(\+|\-|▁)?(\d+)(\.)?(\d+)?")
            self.numeric_token_ids = set(
                v for k, v in tokenizer.get_vocab().items() if self.numeric_token_pattern.fullmatch(k)
            )
            self.numeric_token_tensor = torch.tensor(list(self.numeric_token_ids), device=DEVICE)

        self.batch_stats = []

    def parse_rt(self, predictions):
        if predictions.is_cuda:
            predictions = predictions.cpu()

        predictions_np = predictions.numpy()

        parsed_tokens = np.vectorize(lambda x: self.index_to_token.get(x, '<pad>'))(predictions_np)

        converted_numbers, _ = convert_tokens_to_num_rt(parsed_tokens)
        return converted_numbers

    def calculate_mse_rt(self, predicted: List[float], groundtruth: List[float]) -> Tuple[float, int]:
        """
            Calculates the mean squared error for valid predicted-ground truth pairs.
            
            Args:
                predicted: A list of predicted float numbers.
                groundtruth: A list of ground truth float numbers.
            
            Returns:
                The mean squared error for valid pairs and the count of valid pairs.
            """
        mse = 0.0
        valid_count = 0

        for pred, gt in zip(predicted, groundtruth):
            if not math.isnan(pred) and not math.isnan(gt):
                mse += (pred - gt) ** 2
                valid_count += 1

        if valid_count == 0:
            return float('nan'), 0
        return mse / valid_count, valid_count

    def calculate_mse_xval(self,predictions, number_predictions, token_labels, number_labels):
        num_mask = torch.isin(token_labels, torch.tensor(self.tokenizer.get_num_token_ids(), device=DEVICE))
        mse = F.mse_loss(
            number_predictions[num_mask],
            number_labels[num_mask].reshape(-1, 1),
            reduction="mean",
        )
        predicted_num_mask = predictions == 32100
        true_num_mask = num_mask
        # 1. Predict number but no number in true values
        sum_incorrectly_predicted_nums = np.sum(predicted_num_mask & ~true_num_mask)
        sum_incorrectly_predicted_text = np.sum(~predicted_num_mask & true_num_mask)

        return mse.cpu().numpy(), sum_incorrectly_predicted_nums, sum_incorrectly_predicted_text

    def perplexity(self, logits, labels):
        # Mask to ignore panumeric_tokening tokens (-100)
        mask = labels != -100

        # Apply mask to predictions and labels
        masked_logits = logits[mask]
        masked_labels = labels[mask]

        # Compute negative log likelihood
        nll = F.cross_entropy(masked_logits, masked_labels, reduction='mean')

        # Calculate perplexity
        perplexity = torch.exp(nll)

        return perplexity.item()

    def decode_ids(self, ids, numbers=None):
        """
        Decodes the given ids to strings using the tokenizer and the number encoding.
        Thereby the ids are converted to strings and number tokens are replaced by the corresponding numbers, if
        numbers are given. Additionally special tokens are removed.

        Example 1:
        ids = [0, 32000, 15, 1]
        numbers = [1, -6, 1, 1]

        decoded ids = ["<pad>", "[NUM], ";", "</s>"]
        returns: ["-6", ";"]

        Example 2:
        ids = [0, 32001, 15, 1]
        numbers = None

        decoded ids = ["<pad>", "_1_0_", ";", "</s>"]
        returns: ["_1_0_", ";"]

        :param ids: the ids to decode
        :param numbers: the numbers to replace the number tokens with
        :return: the decoded ids
        """
        ids = ids.cpu().numpy()
        ids = np.where(ids != PADDING_TOKEN, ids, self.tokenizer.pad_token_id)

        if numbers is None:
            decoded_ids = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        else:
            numbers = numbers.cpu().numpy()
            decoded_ids_array = np.vectorize(lambda x: self.index_to_token.get(x, '<pad>'))(ids)
            decoded_ids = np.where(np.isin(ids, self.tokenizer.get_num_token_ids()), numbers, decoded_ids_array)
            # Remove padding tokens
            decoded_ids = [list(filter(lambda x: x not in self.tokenizer.all_special_tokens, decoded_id)) for decoded_id in decoded_ids]
            decoded_ids = [" ".join(decoded_id) for decoded_id in decoded_ids]

        return decoded_ids

    def compute_rouge(self, decoded_preds, decoded_labels):
        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result

    def compute_bleu(self, decoded_preds, decoded_labels):
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Compute BLEU
        result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    def __call__(self, pred: EvalPrediction, compute_result: bool) -> Dict[str, float]:
        """
            While EvalPrediction declares to send 2- or 3-Tupel of np.arrays, we actually receive a 2-Tupel of tupels!
            The number of elements in model_output differs based on the number_encoding choosen.
            The shapes of the contained tensors differ for model_output and labels:
            Use print_structure to analyse.
            rt Args: 
                model_output (2-tupel of torch.Tensors) 
                labels 2-tupel of torch.Tensors: token_labels, number_labels
            xval Args:
                model_output (5-tupel of torch.Tensors)
                labels 2-tupel of torch.Tensors: token_labels, number_labels
            general Args:
                compute_result (bool): We calculate metrics in batches. Set to True during final batch to calculate overall results 

            Returns:
                Overall results if compute_result else None 
        
        """
        # Extract predictions and labels from pred tuple
        model_output, labels = pred
        logits, predictions = model_output

        if self.number_encoding in ["xval", "rt"]:
            token_labels, number_labels = labels
        else:
            token_labels, number_labels = labels, None

        if self.number_encoding == "xval":
            predictions, predicted_numbers = predictions
            decoded_preds = self.decode_ids(predictions, predicted_numbers)
            decoded_labels = self.decode_ids(token_labels, number_labels)
        else:
            decoded_preds = self.decode_ids(predictions)
            decoded_labels = self.decode_ids(token_labels)

        # compute perplexity
        perplexity_value = self.perplexity(logits, token_labels[:, :logits.size(1)])

        bleu = self.compute_bleu(decoded_preds, decoded_labels)
        rouge = self.compute_rouge(decoded_preds, decoded_labels)

        # Mask to ignore panumeric_tokening tokens (-100)
        mask = token_labels != PADDING_TOKEN

        # Apply mask to predictions and labels
        masked_predictions = torch.where(mask, predictions, MASKED_OUT)
        masked_labels = torch.where(mask, token_labels, MASKED_OUT)

        # compute whole number accuracy and token accuracy
        correct_predictions_w = torch.all(masked_predictions == masked_labels, dim=1)
        accuracy_w = torch.mean(correct_predictions_w.float()).item()
        correct_predictions = (predictions == token_labels) & mask
        accuracy = (torch.sum(correct_predictions) / torch.sum(mask)).item() if torch.sum(mask) > 0 else 0

        if self.number_encoding == "xval":
            mse = self.calculate_mse_xval(predictions, predicted_numbers, token_labels, number_labels)
            nan_count = -1  # TODO
        elif self.number_encoding == "rt":
            predicted_numbers = self.parse_rt(predictions)
            groundtruth_numbers = self.parse_rt(token_labels)
            mse, _ = self.calculate_mse_rt(predicted_numbers, groundtruth_numbers)
            nan_count = sum(math.isnan(num) for num in predicted_numbers)
        elif self.number_encoding.lower() == "none":
            num_mask = torch.isin(predictions, self.numeric_token_tensor).cpu().numpy()

            predictions = predictions.cpu().numpy()
            token_labels = token_labels.cpu().numpy()

            # We have to remove ▁ instead of _ because T5 Tokenizer uses unicode for some reason and encode whitespaces like this
            convert_to_string = np.vectorize(lambda x: self.index_to_token[x].strip("▁"))
            predicted_text = convert_to_string(predictions)
            groundtruth_text = convert_to_string(token_labels)

            predicted_text = np.where(num_mask, predicted_text, "")
            groundtruth_text = np.where(num_mask, groundtruth_text, "")

            collapsed_predicted_text = np.apply_along_axis(''.join, 1, predicted_text)
            collapsed_groundtruth_text = np.apply_along_axis(''.join, 1, groundtruth_text)

            def try_convert(s):
                try:
                    return float(s)
                except ValueError:
                    return np.nan

            vectorized_conversion = np.vectorize(try_convert)
            predicted_numbers = vectorized_conversion(collapsed_predicted_text)
            groundtruth_numbers = vectorized_conversion(collapsed_groundtruth_text)

            mse, _ = self.calculate_mse_rt(predicted_numbers, groundtruth_numbers)
            nan_count = sum(math.isnan(num) for num in predicted_numbers)
        else:
            raise NotImplementedError("Requesting evaluation for not supported number_encoding: {self.number_encoding}")

        self.batch_stats.append({
            'token_accuracy_whole': accuracy_w,
            'token_accuracy': accuracy,
            'MSE': mse,
            'nan_count': nan_count,
            'token_perplexity': perplexity_value,
            'bleu': bleu['score'],
            'rouge': rouge['rougeLsum']
        })

        if compute_result:
            computed_metrics = {
                'token_accuracy_whole': np.mean([stat['token_accuracy_whole'] for stat in self.batch_stats]),
                'token_accuracy': np.mean([stat['token_accuracy'] for stat in self.batch_stats]),
                'MSE': np.mean([stat['MSE'] for stat in self.batch_stats]),
                'nan_count': np.sum([stat['nan_count'] for stat in self.batch_stats]),
                'token_perplexity': np.mean([stat['token_perplexity'] for stat in self.batch_stats]),
                "bleu": np.mean([stat['bleu'] for stat in self.batch_stats]),
                "rouge": np.mean([stat['rouge'] for stat in self.batch_stats]),
            }
            self.batch_stats = []
            return computed_metrics
