import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number
from typing import List, Tuple, Dict
import math
import functools

import numpy as np

PAnumeric_tokenING_TOKEN = -100
MASKED_OUT = -1
MALFORMED_RT_TOKEN = 100000
NON_NUMERIC_TOKEN = 10000
SURELY_NUMERIC_TOKEN_BOUND = 5000

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
    #Numeric tokens are a) consequitive without interruption, b) encoded position form decreasing sequence
    valid_sequence = np.all(np.diff(numeric_token_indices) == 1) and np.all(np.diff(validation_array[numeric_token_indices]) == -1)
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

    #We use a trick by converting non numeric tokens to zero and assuming there is only one numeric token
    float_array = np.vectorize(functools.partial(encoding_to_number, invalid_strict=False))(token_array)
    value_array = np.sum(float_array, axis=1)

    #Currently, this leaves multiple [NEG] predictions a possibility. Fixing not trivial. Maybe apply simplification rules like https://github.com/PolymathicAI/xVal/blob/main/xval/preprocess.py
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
        self.batch_stats = []

    def parse_rt(self, predictions):
        if predictions.is_cuda:
            predictions = predictions.cpu()
        
        predictions_np = predictions.numpy()
        
        parsed_tokens = np.vectorize(lambda x: self.index_to_token.get(x, '<pad>'))(predictions_np)
        
        converted_numbers, _ = convert_tokens_to_num_rt(parsed_tokens)
        return converted_numbers
    def calculate_mse(self, predicted: List[float], groundtruth: List[float]) -> Tuple[float, int]:
        """
            Calculates the mean squared error for valid predicted-ground truth pairs.
            
            Parameters:
            - predicted: A list of predicted float numbers.
            - groundtruth: A list of ground truth float numbers.
            
            Returns:
            - The mean squared error for valid pairs and the count of valid pairs.
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

    def __call__(self, pred: EvalPrediction, compute_result: bool) -> Dict[str, float]:
        # Extract predictions and labels
        model_output, labels = pred

        #While EvalPrediction uses np.arrays we actually use torch.Tensors
        logits = model_output[0]
        token_labels, number_labels = labels
        
        # compute perplexity
        perplexity_value = self.perplexity(logits, token_labels)

        predictions = torch.argmax(logits, dim=2)

        if self.number_encoding == "xval":
            # TODO .reshape(-1) not correct, need to apply mask
            predicted_numbers = model_output[-1].detach().cpu().numpy().reshape(-1)
            groundtruth_numbers = number_labels.detach().cpu().numpy().reshape(-1)
        else:
            predicted_numbers = self.parse_rt(predictions)
            groundtruth_numbers = self.parse_rt(token_labels)

        # Mask to ignore panumeric_tokening tokens (-100)
        mask = token_labels != PAnumeric_tokenING_TOKEN

        # Apply mask to predictions and labels
        masked_predictions = torch.where(mask, predictions, MASKED_OUT)
        masked_labels = torch.where(mask, token_labels, MASKED_OUT)

        # compute whole number accuracy and token accuracy
        correct_predictions_w = torch.all(masked_predictions == masked_labels, dim=1)
        accuracy_w = torch.mean(correct_predictions_w.float()).item()
        correct_predictions = (predictions == token_labels) & mask
        accuracy = (torch.sum(correct_predictions) / torch.sum(mask)).item() if torch.sum(mask) > 0 else 0

        mse, _ = self.calculate_mse(predicted_numbers, groundtruth_numbers)
        nan_count = sum(math.isnan(num) for num in predicted_numbers)

        self.batch_stats.append({
            'token_accuracy_whole': accuracy_w,
            'token_accuracy': accuracy,
            'MSE': mse,
            'nan_count': nan_count,
            'token_perplexity': perplexity_value
        })
        if compute_result:
            computed_metrics = {
                'token_accuracy_whole': np.mean([stat['token_accuracy_whole'] for stat in self.batch_stats]),
                'token_accuracy': np.mean([stat['token_accuracy'] for stat in self.batch_stats]),
                'MSE': np.mean([stat['MSE'] for stat in self.batch_stats]),
                'nan_count': np.sum([stat['nan_count'] for stat in self.batch_stats]),
                'token_perplexity': np.mean([stat['token_perplexity'] for stat in self.batch_stats])
            }
            self.batch_stats = []
            return computed_metrics
