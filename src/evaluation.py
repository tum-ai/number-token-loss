import numpy as np
import torch


class CustomMetrics():
    '''
    Compute custom metrics for the model with access to the vocab to compute MSE
    '''

    def __init__(self, vocab):
        self.index_to_token = {v: k for k, v in vocab.items()}

    def parse(self, predictions):
        parsed_tokens = [[self.index_to_token.get(index, 0) for index in seq] for seq in predictions]
        # TODO parse to get MSE loss
        return parsed_tokens

    def __call__(self, pred):
        # Extract predictions and labels
        predictions, labels = pred
        predictions = np.argmax(predictions[0], axis=2)

        # Mask to ignore padding tokens (-100)
        mask = labels != -100
        
        # Apply mask to predictions and labels
        masked_predictions = np.where(mask, predictions, -1)  # Set masked-out positions to -1
        masked_labels = np.where(mask, labels, -1)
        
        # compute whole number accuracy and token accuracy
        correct_predictions_w = np.all(masked_predictions == masked_labels, axis=1)
        accuracy_w = np.mean(correct_predictions_w)
        correct_predictions = (predictions == labels) & mask
        accuracy = np.sum(correct_predictions) / np.sum(mask) if np.sum(mask) > 0 else 0

        # compute MSE
        numbers = self.parse(predictions)

        # mask out invalid numbers (tokens do not form a valid number)


        return {
            'accuracy_whole': accuracy_w,
            'accuracy': accuracy,
            'MSE': 1
        }