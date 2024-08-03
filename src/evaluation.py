import numpy as np
import torch
import torch.nn.functional as F
from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from src.encoding_decoding.numerical_encodings import encoding_to_number


class CustomMetrics:
    '''
    Compute custom metrics for the model with access to the vocab to compute MSE
    '''

    def __init__(self, tokenizer: NumberEncodingTokenizer, number_encoding: str):
        self.tokenizer = tokenizer
        self.number_encoding = number_encoding

    def mse(self, predictions, number_predictions, token_labels, number_labels):
        num_mask = np.isin(token_labels, self.tokenizer.get_num_token_ids())

        if self.number_encoding == "rt":
            shape = predictions.shape
            prediction_tokens = np.array(self.tokenizer.convert_ids_to_tokens(predictions.reshape(-1))).reshape(shape)
            prediction_num_mask = np.isin(predictions, self.tokenizer.get_num_token_ids())
            converted_numbers = [encoding_to_number(token) for token in prediction_tokens[prediction_num_mask]]
            number_predictions = np.full_like(predictions, 100) # TODO what do we want to happen, if the model should predict a number token but does not do so?
            number_predictions[prediction_num_mask] = converted_numbers

        mse = F.mse_loss(
            torch.tensor(number_predictions[num_mask]),
            torch.tensor(number_labels[num_mask].reshape(-1, 1)),
            reduction="mean",
        )

        return mse

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
        mask = token_labels != -100

        # Apply mask to predictions and labels
        masked_predictions = np.where(mask, predictions, -1)  # Set masked-out positions to -1
        masked_labels = np.where(mask, token_labels, -1)

        # compute whole number accuracy and token accuracy
        correct_predictions_w = np.all(masked_predictions == masked_labels, axis=1)
        accuracy_w = np.mean(correct_predictions_w)
        correct_predictions = (predictions == token_labels) & mask
        accuracy = np.sum(correct_predictions) / np.sum(mask) if np.sum(mask) > 0 else 0

        # compute MSE
        mse = self.mse(predictions, number_predictions, token_labels, number_labels)

        # TODO mask out invalid numbers (tokens do not form a valid number)

        return {
            'token_accuracy_whole': accuracy_w,
            'token_accuracy': accuracy,
            'MSE': mse,
            'token_perplexity': perplexity_value
        }
