import numpy as np
import torch
import torch.nn.functional as F
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

def convert_to_number_xval(parsed_tokens: List[List[str]]) -> List[float]:
    """
    Converts a list of lists of strings to a list of floats.
    
    Parameters:
    - parsed_tokens: A list of lists containing string xval representations of numbers.

    Returns:
    - A list of floats converted from the provided strings. Invalid numbers are represented as NANs
    """
    raise NotImplementedError("Xval not supported yet")
    #return [float('nan')]

PADDING_TOKEN = -100
MASKED_OUT = -1

class CustomMetrics:
    '''
    Compute custom metrics for the model with access to the vocab to compute MSE
    '''

    def __init__(self, tokenizer: NumberEncodingTokenizer, number_encoding: str):
        self.tokenizer = tokenizer
        self.index_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        self.number_encoding = number_encoding
        self.converter = convert_to_number_rt if number_encoding =="rt" else convert_to_number_xval
    
    def parse(self, predictions):
        print(predictions)
        parsed_tokens = [[self.index_to_token.get(index, '<pad>') for index in seq] for seq in predictions]
        converted_numbers = self.converter(parsed_tokens)
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

        # Convert to torch tensors
        masked_logits = torch.tensor(masked_logits)
        masked_labels = torch.tensor(masked_labels)

        # Compute negative log likelihood
        nll = F.cross_entropy(masked_logits, masked_labels, reduction='mean')

        # Calculate perplexity
        perplexity = torch.exp(nll)

        return perplexity.item()

    def __call__(self, pred):
        # Extract predictions and labels
        model_output, labels = pred
        logits = model_output[0]
        if self.number_encoding == "xval":
            number_predictions = model_output[-1]
        else:
            number_predictions = None

        token_labels, number_labels = labels

        # compute perplexity
        perplexity_value = self.perplexity(logits, token_labels)

        predictions = np.argmax(logits, axis=2)

        # Mask to ignore padding tokens (-100)
        mask = token_labels != -100 # TODO use globl PADDING_TOKEN variable (see above)

        # Apply mask to predictions and labels
        masked_predictions = np.where(mask, predictions, -1)  # Set masked-out positions to -1 ' TODO MASKED_OUT variable
        masked_labels = np.where(mask, token_labels, -1)

        # compute whole number accuracy and token accuracy
        correct_predictions_w = np.all(masked_predictions == masked_labels, axis=1)
        accuracy_w = np.mean(correct_predictions_w)
        correct_predictions = (predictions == token_labels) & mask
        accuracy = np.sum(correct_predictions) / np.sum(mask) if np.sum(mask) > 0 else 0 # TODO ?
        
        predicted_numbers = self.parse(predictions)
        groundtruth_numbers = self.parse(token_labels)


        #mse = self.mse(predictions, number_predictions, token_labels, number_labels)


        mse, _ = self.calculate_mse(predicted_numbers, groundtruth_numbers)
        nan_count = sum(math.isnan(num) for num in predicted_numbers)

        # TODO mask out invalid numbers (tokens do not form a valid number)

        return {
            'token_accuracy_whole': accuracy_w,
            'token_accuracy': accuracy,
            'MSE': mse,
            'nan_count': nan_count,
            'token_perplexity': perplexity_value
        }
