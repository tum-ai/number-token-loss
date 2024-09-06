import logging
import math
import re
from typing import List, Dict, Tuple

import evaluate
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer, NUMBER_REGEX
from src.tokenizer.t5custom_tokenizer import check_number_predictions

PADDING_TOKEN = -100
MASKED_OUT = -1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomMetrics:
    """
    Compute custom metrics for the model with access to the vocab to compute MSE
    """

    def __init__(self, tokenizer: NumberEncodingTokenizer, number_encoding: str, output_dir: str):
        self.tokenizer = tokenizer
        self.index_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        self.number_encoding = number_encoding
        self.output_dir = output_dir
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("sacrebleu")
        nltk.download('punkt_tab')
        nltk.download("punkt")

        if self.number_encoding == "none":
            # ▁ is necessary as T5 Tokenizes white spaces like this and it has tokens for 1 and ▁1
            self.numeric_token_pattern = re.compile(r"(\+|\-|▁)?(\d+)(\.)?(\d+)?")
            self.numeric_token_ids = set(
                v for k, v in tokenizer.get_vocab().items() if self.numeric_token_pattern.fullmatch(k)
            )
            self.numeric_token_tensor = torch.tensor(list(self.numeric_token_ids), device=DEVICE)

        self.batch_stats = []

        self.eval_count = 0

    def calculate_result_mse(self, prediction: List[str], label: List[str]) -> List[float]:
        return [self.calculate_result_number_loss_per_sample(prediction[i], label[i], order=2) for i in range(len(prediction))]

    def calculate_result_mae(self, prediction: List[str], label: List[str]) -> List[float]:
        return [self.calculate_result_number_loss_per_sample(prediction[i], label[i], order=1) for i in range(len(prediction))]

    def calculate_result_number_loss_per_sample(self, prediction: str, label: str, order: int):
        # Extract the last number of both strings and compare them
        # TODO only valid for this dataset, remove for other datasets

        prediction_number = re.findall(r"#\s*[+-]?\s*(\d+)(\.\d+)?", prediction)
        if len(prediction_number) == 0:
            return np.nan
        prediction_number = "".join(prediction_number[-1])
        label_number = "".join(re.findall(r"#\s*[+-]?\s*(\d+)(\.\d+)?", label)[-1])

        # Convert the strings to floats
        prediction_number = float(prediction_number)
        label_number = float(label_number)

        # Calculate the mean squared error
        try:
            mse = np.abs(prediction_number - label_number) ** order
        except Exception as e:
            logging.error(f"Error calculating MSE: {e} with numbers {prediction_number} and {label_number}")
            mse = np.nan
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
        if not self.number_encoding.lower() in ["xval", "rt", "none"]:
            raise NotImplementedError(
                f"Requesting evaluation for not supported number_encoding: {self.number_encoding}")

        # Extract predictions and labels from pred tuple
        model_output, labels = pred
        logits, predictions = model_output

        if self.number_encoding in ["xval", "rt"]:
            token_labels, number_labels = labels

        else:
            token_labels, number_labels = labels, None

        if self.number_encoding == "xval":
            predictions, predicted_numbers = predictions
            decoded_preds, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(
                predictions, predicted_numbers)
            decoded_labels, sanity_invalid_number_prediction, sanity_no_number_prediction = self.tokenizer.decode_into_human_readable(
                token_labels, number_labels)
        else:
            if hasattr(self.tokenizer, "decode_into_human_readable"):
                decoded_preds, count_invalid_number_prediction, count_no_number_prediction = self.tokenizer.decode_into_human_readable(
                    predictions)
                decoded_labels, sanity_invalid_number_prediction, sanity_no_number_prediction = self.tokenizer.decode_into_human_readable(
                    token_labels)
            else:
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                count_invalid_number_prediction, count_no_number_prediction = check_number_predictions(
                    decoded_preds)
                decoded_labels = self.tokenizer.batch_decode(token_labels, skip_special_tokens=True)
                sanity_invalid_number_prediction, sanity_no_number_prediction = check_number_predictions(
                    decoded_labels)

        # We should never observe invalid numbers and mostly likely never no number for gt
        if max(sanity_invalid_number_prediction, sanity_no_number_prediction) > 0:
            print(sanity_invalid_number_prediction)
            print(sanity_no_number_prediction)

        if compute_result:
            # save decoded predictions and labels for debugging
            with open(f"{self.output_dir}/decoded_preds_{self.eval_count}.txt", "w") as f:
                for idx in range(len(decoded_preds)):
                    f.write(f"Prediction {idx}: {decoded_preds[idx]}\n")
                    f.write(f"Label {idx}: {decoded_labels[idx]}\n")
            self.eval_count += 1

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

        mse = self.calculate_result_mse(decoded_preds, decoded_labels)
        mae = self.calculate_result_mae(decoded_preds, decoded_labels)
        count_not_produced_valid_results = np.sum(np.isnan(mse))

        self.batch_stats.append({
            'token_accuracy_whole': accuracy_w,
            'token_accuracy': accuracy,
            'MSE': mse,
            'MAE': mae,
            "count_not_produced_valid_results": count_not_produced_valid_results,
            "total_count": predictions.shape[0],
            "count_invalid_number_prediction": count_invalid_number_prediction,
            "count_no_number_prediction": count_no_number_prediction,
            'token_perplexity': perplexity_value,
            'bleu': bleu['score'],
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
        })

        if compute_result:
            total_count = np.sum([stat['total_count'] for stat in self.batch_stats])
            computed_metrics = {
                'token_accuracy_whole': np.mean([stat['token_accuracy_whole'] for stat in self.batch_stats]),
                'token_accuracy': np.mean([stat['token_accuracy'] for stat in self.batch_stats]),
                'MSE': np.nanmean(np.concatenate([stat['MSE'] for stat in self.batch_stats])),
                'MAE': np.nanmean(np.concatenate([stat['MAE'] for stat in self.batch_stats])),
                "count_not_produced_valid_results": np.sum(
                    [stat['count_not_produced_valid_results'] for stat in self.batch_stats]),
                "average_count_not_produced_valid_results": np.sum(
                    [stat['count_not_produced_valid_results'] for stat in self.batch_stats]) / total_count,
                "count_invalid_number_prediction": np.sum(
                    [stat['count_invalid_number_prediction'] for stat in self.batch_stats]),
                "count_no_number_prediction": np.sum(
                    [stat['count_no_number_prediction'] for stat in self.batch_stats]),
                "average_invalid_number_prediction": np.sum(
                    [stat['count_invalid_number_prediction'] for stat in self.batch_stats]) / total_count,
                "average_no_number_prediction": np.sum(
                    [stat['count_no_number_prediction'] for stat in self.batch_stats]) / total_count,
                'token_perplexity': np.mean([stat['token_perplexity'] for stat in self.batch_stats]),
                "bleu": np.mean([stat['bleu'] for stat in self.batch_stats]),
                "rouge1": np.mean([stat['rouge1'] for stat in self.batch_stats]),
                "rouge2": np.mean([stat['rouge2'] for stat in self.batch_stats]),
                "rougeL": np.mean([stat['rougeL'] for stat in self.batch_stats]),
            }
            self.batch_stats = []
            return computed_metrics
