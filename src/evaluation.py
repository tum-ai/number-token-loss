import copy
from typing import List, Tuple, Dict
import math
import functools
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer, NUMBER_REGEX
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
    # Make a copy of the array to work with
    output = np.array(validation_array, dtype=float)
    output = np.where(output == NON_NUMERIC_TOKEN, np.nan, output)

    # Find indices of non-NaN numbers
    valid_mask = ~np.isnan(output)

    # Compute differences where possible
    differences = np.diff(output, prepend=np.nan)

    # Valid decreases are exactly -1
    valid_decreases = ((differences == -1) | np.isnan(differences)) & (~np.isnan(output))

    # We need to propagate invalids across all involved in a sequence
    # To do this, take valid_mask and wherever valid_decreases is False,
    # mark the next value as invalid (if it was originally valid)
    invalid_transitions = (valid_mask & ~valid_decreases)

    # Count invalid transitions
    invalid_count = np.sum(invalid_transitions)

    # Set invalid sequence starts to NaN
    output[invalid_transitions] = np.nan

    return ~np.isnan(output), invalid_count


def sum_sequences(arr):
    # Create an array to hold the result, initialized with NaNs
    result = np.full_like(arr, np.nan)

    # Identify where the array is not NaN
    not_nan = ~np.isnan(arr)

    # Find where sequences start (changes in the not_nan array)
    changes = np.diff(not_nan.astype(int))
    starts = np.where(changes == 1)[0] + 1  # Shift by 1 because diff reduces index by 1
    ends = np.where(changes == -1)[0]

    # Handle cases where the array starts or ends with non-NaN values
    if not_nan[0]:
        starts = np.insert(starts, 0, 0)
    if not_nan[-1]:
        ends = np.append(ends, len(arr) - 1)

    # Sum the sequences and assign the sums to the correct positions in the result array
    for start, end in zip(starts, ends):
        sequence_sum = np.sum(arr[start:end+1])
        result[end] = sequence_sum

    return result


def convert_tokens_to_num_rt(token_array: np.ndarray, tokenizer: NumberEncodingTokenizer):
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

    valid_sequence_mask, invalid_count = is_valid_numeric_token_sequence(validation_array)

    number_token_array = copy.deepcopy(token_array)
    number_token_array[~valid_sequence_mask] = np.nan
    number_token_array = np.vectorize(functools.partial(encoding_to_number, invalid_strict=False))(number_token_array)
    number_token_array = np.where(~valid_sequence_mask, np.nan, number_token_array)
    number_token_array = np.array(list(map(sum_sequences, number_token_array)))

    result = []

    for row in range(valid_sequence_mask.shape[0]):
        result.append([])
        is_negative = False
        for idx in range(valid_sequence_mask.shape[1]):
            if valid_sequence_mask[row][idx]:
                if np.isnan(number_token_array[row][idx]):
                    continue
                else:
                    result[row].extend(tokenizer.t5_tokenize(str(number_token_array[row][idx] * (-1 if is_negative else 1))))
                    is_negative = False
            else:
                token = token_array[row][idx]
                is_negative = token == "[NEG]"
                result[row].append(token)

    return result, invalid_count


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

        converted_numbers, invalid_count = convert_tokens_to_num_rt(parsed_tokens, self.tokenizer)
        converted_numbers = [list(filter(lambda x: x not in self.tokenizer.all_special_tokens, decoded_id)) for decoded_id in converted_numbers]
        decoded = [self.tokenizer.convert_tokens_to_string(tokens) if len(tokens) else "" for tokens in converted_numbers]

        return decoded, invalid_count

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

    def calculate_mse_xval(self, token_predictions, number_predictions, token_labels, number_labels):
        gt_num_mask = torch.isin(token_labels, torch.tensor(self.tokenizer.get_num_token_ids(), device=DEVICE))
        predicted_num_mask = torch.isin(token_predictions, torch.tensor(self.tokenizer.get_num_token_ids(), device=DEVICE))

        mse = F.mse_loss(
            number_predictions[gt_num_mask],
            number_labels[gt_num_mask].reshape(-1, 1),
            reduction="mean",
        )

        # 1. Predict number but no number in true values
        sum_incorrectly_predicted_nums = torch.sum(predicted_num_mask & ~gt_num_mask).item()
        # 2. No number predicted but number in true values
        sum_incorrectly_predicted_text = torch.sum(~predicted_num_mask & gt_num_mask).item()

        return mse.cpu().numpy(), sum_incorrectly_predicted_nums, sum_incorrectly_predicted_text

    def calculate_result_mse(self, prediction: List[str], label: List[str]):
        return np.nanmean([self.calculate_result_mse_per_sample(prediction[i], label[i]) for i in range(len(prediction))])

    def calculate_result_mse_per_sample(self, prediction: str, label: str):
        # Extract the last number of both strings and compare them
        prediction_number = re.findall(NUMBER_REGEX, prediction)
        if len(prediction_number) == 0:
            return np.nan
        prediction_number = "".join(prediction_number[-1])
        label_number = "".join(re.findall(NUMBER_REGEX, label)[-1])

        # Convert the strings to floats
        prediction_number = float(prediction_number)
        label_number = float(label_number)

        # Calculate the mean squared error
        mse = (prediction_number - label_number) ** 2
        return mse


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

        if self.number_encoding == "none":
            decoded_ids = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        elif self.number_encoding == "xval":
            numbers = numbers.cpu().numpy()
            numbers = list(map(lambda x: list(map(lambda y: self.tokenizer.tokenize(str(y)), x)), numbers))

            decoded_ids = np.array(list(map(lambda sample: self.tokenizer.convert_ids_to_tokens(sample), ids)))

            def replace_number_tokens_with_numbers(id, number, decoded_id):
                return number if id in self.tokenizer.get_num_token_ids() else decoded_id

            def flatten(lst):
                flat_list = []
                for item in lst:
                    if isinstance(item, list):
                        flat_list.extend(flatten(item))
                    else:
                        flat_list.append(item)
                return flat_list

            decoded_ids = [
                list(map(lambda id, number, decoded_id: replace_number_tokens_with_numbers(id, number, decoded_id), ids_row, numbers_row, decoded_ids_row))
                for ids_row, numbers_row, decoded_ids_row in zip(ids, numbers, decoded_ids)
            ]
            decoded_ids = list(map(flatten, decoded_ids))

            # Remove padding tokens
            decoded_ids = [list(filter(lambda x: x not in self.tokenizer.all_special_tokens, decoded_id)) for decoded_id in decoded_ids]
            decoded_ids = list(map(lambda sample: self.tokenizer.convert_tokens_to_string(sample) if len(sample) else "", decoded_ids))
        elif self.number_encoding == "rt":
            return self.parse_rt(torch.tensor(ids))[0]
        else:
            raise NotImplementedError(f"Requesting evaluation for not supported number_encoding: {self.number_encoding}")

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
            _, sum_incorrectly_predicted_nums, sum_incorrectly_predicted_text = self.calculate_mse_xval(predictions, predicted_numbers, token_labels, number_labels)
            mse = self.calculate_result_mse(decoded_preds, decoded_labels)
            count_invalid_numbers = 0
        elif self.number_encoding == "rt":
            decoded_labels, invalid_number_tokens_gt = self.parse_rt(token_labels)
            if invalid_number_tokens_gt > 0:
                raise ValueError(f"Found {invalid_number_tokens_gt} invalid number tokens in ground truth")
            decoded_preds, count_invalid_numbers = self.parse_rt(predictions)
            mse = self.calculate_result_mse(decoded_preds, decoded_labels)

            predicted_num_mask = torch.isin(predictions.cpu(), torch.tensor(self.tokenizer.get_num_token_ids())).cpu().numpy()
            gt_num_mask = torch.isin(token_labels.cpu(), torch.tensor(self.tokenizer.get_num_token_ids())).cpu().numpy()
            sum_incorrectly_predicted_nums = np.sum(predicted_num_mask & ~gt_num_mask)
            sum_incorrectly_predicted_text = np.sum(~predicted_num_mask & gt_num_mask)
        elif self.number_encoding.lower() == "none":
            gt_num_mask = torch.isin(token_labels, self.numeric_token_tensor).cpu().numpy()
            predicted_num_mask = torch.isin(predictions, self.numeric_token_tensor).cpu().numpy()

            predictions = predictions.cpu().numpy()
            token_labels = token_labels.cpu().numpy()

            # We have to remove ▁ instead of _ because T5 Tokenizer uses unicode for some reason and encode whitespaces like this
            convert_to_string = np.vectorize(lambda x: self.index_to_token[x].strip("▁"))
            predicted_text = convert_to_string(predictions)
            groundtruth_text = convert_to_string(token_labels)

            predicted_text = np.where(gt_num_mask, predicted_text, "")
            groundtruth_text = np.where(gt_num_mask, groundtruth_text, "")

            collapsed_predicted_text = np.apply_along_axis(''.join, 1, predicted_text)
            collapsed_groundtruth_text = np.apply_along_axis(''.join, 1, groundtruth_text)

            def try_convert(s):
                try:
                    return float(s)
                except ValueError:
                    return np.nan

            vectorized_conversion = np.vectorize(try_convert)
            predicted_numbers = vectorized_conversion(collapsed_predicted_text)

            mse = self.calculate_result_mse(decoded_preds, decoded_labels)
            count_invalid_numbers = sum(math.isnan(num) for num in predicted_numbers)
            sum_incorrectly_predicted_nums = np.sum(predicted_num_mask & ~gt_num_mask).item()
            sum_incorrectly_predicted_text = np.sum(~predicted_num_mask & gt_num_mask).item()
        else:
            raise NotImplementedError("Requesting evaluation for not supported number_encoding: {self.number_encoding}")

        self.batch_stats.append({
            'token_accuracy_whole': accuracy_w,
            'token_accuracy': accuracy,
            'MSE': mse,
            'count_invalid_numbers': count_invalid_numbers,
            "count_incorrectly_predicted_nums": sum_incorrectly_predicted_nums,
            "count_incorrectly_predicted_text": sum_incorrectly_predicted_text,
            'token_perplexity': perplexity_value,
            'bleu': bleu['score'],
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
        })

        if compute_result:
            computed_metrics = {
                'token_accuracy_whole': np.mean([stat['token_accuracy_whole'] for stat in self.batch_stats]),
                'token_accuracy': np.mean([stat['token_accuracy'] for stat in self.batch_stats]),
                'MSE': np.nanmean([stat['MSE'] for stat in self.batch_stats]),
                'count_invalid_numbers': np.sum([stat['count_invalid_numbers'] for stat in self.batch_stats]),
                "count_incorrectly_predicted_nums": np.sum([stat['count_incorrectly_predicted_nums'] for stat in self.batch_stats]),
                "count_incorrectly_predicted_text": np.sum([stat['count_incorrectly_predicted_text'] for stat in self.batch_stats]),
                'token_perplexity': np.mean([stat['token_perplexity'] for stat in self.batch_stats]),
                "bleu": np.mean([stat['bleu'] for stat in self.batch_stats]),
                "rouge1": np.mean([stat['rouge1'] for stat in self.batch_stats]),
                "rouge2": np.mean([stat['rouge2'] for stat in self.batch_stats]),
                "rougeL": np.mean([stat['rougeL'] for stat in self.batch_stats]),
            }
            self.batch_stats = []
            return computed_metrics
