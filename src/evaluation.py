import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number
from typing import List, Tuple, Dict
import math


def is_valid_number_sequence(tokens: List[str]) -> List[bool]:
    """
    Validates the tokens based on the specified rules.
    
    Parameters:
    - tokens: A list of string tokens representing numbers.
    
    Returns:
    - A list of booleans indicating whether each token is valid.
    """
    is_valid = []

    for sublist in tokens:
        valid_sublist = True
        last_pos = None
        for token in sublist:
            if token.startswith("_") and token.endswith("_"):
                try:
                    parts = token.strip("_").split("_")
                    if len(parts) != 2:
                        valid_sublist = False
                        break

                    x, y = parts
                    if not (len(x) == 1 and x.isdigit()):
                        valid_sublist = False
                        break

                    y = int(y)

                    if last_pos is not None and y != last_pos - 1:
                        valid_sublist = False
                        break

                    last_pos = y
                except ValueError:
                    valid_sublist = False
                    break
            elif token in ['[NEG]', '</s>', '<pad>']:
                continue
            else:
                valid_sublist = False
                break

        is_valid.append(valid_sublist)

    return is_valid


def convert_to_number_rt(parsed_tokens: List[List[str]]) -> List[float]:
    """
    Converts parsed tokens to a list of floats after validating and transforming them.
    
    Parameters:
    - parsed_tokens: A list of lists containing string tokens.
    
    Returns:
    - A list of floats, where invalid conversions are represented by NaN.
    """
    results = []
    valid_tokens = is_valid_number_sequence(parsed_tokens)

    for idx, sublist in enumerate(parsed_tokens):
        if not valid_tokens[idx]:
            results.append(float('nan'))
            continue

        transformed = []

        for token in sublist:
            if token == "[NEG]":
                transformed.append("-")
            elif token in ["</s>", "<pad>"]:
                continue
            elif token.startswith("_") and token.endswith("_"):
                parts = token.strip("_").split("_")
                x, y = parts
                y = int(y)
                if y == -1:
                    transformed.append(f".{x}")
                else:
                    transformed.append(x)
            else:
                raise ValueError("Validation missed non number")

        try:
            number_str = "".join(transformed)
            number = float(number_str)
            results.append(number)
        except ValueError:
            results.append(float('nan'))

    return results


PADDING_TOKEN = -100
MASKED_OUT = -1


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
        print(predictions)
        parsed_tokens = [[self.index_to_token.get(index, '<pad>') for index in seq] for seq in predictions]
        converted_numbers = convert_to_number_rt(parsed_tokens)
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
        # Mask to ignore padding tokens (-100)
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

        # Mask to ignore padding tokens (-100)
        mask = token_labels != PADDING_TOKEN

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
