import torch
import torch.nn.functional as F
from torch._tensor import Tensor
import logging
from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class WassersteinNumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, order_numbers: bool,
                 loss_function=F.mse_loss, weight=0.5):
        self.tokenizer = tokenizer
        self.loss_function = loss_function
        self.weight = weight
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())

        # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(token, ignore_order=False)

        # Extract indices and values of number tokens
        number_token_mask = ~torch.isnan(self.nvocab)
        self.number_token_indices = torch.nonzero(number_token_mask, as_tuple=False).squeeze()

        if order_numbers:
            logging.info("Sorting number tokens by numerical value...")
            # Sort the number tokens by their numerical values
            self.number_token_values = self.nvocab[self.number_token_indices]
            sorted_values, sorted_indices = torch.sort(self.number_token_values)
            self.number_token_values = sorted_values
            self.number_token_indices = self.number_token_indices[sorted_indices]

    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        # Extract the logits for the number tokens, ordered by numerical value
        logits = logits[:, :, self.number_token_indices]

        # Compute the softmax over the logits
        softmaxed = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, num_number_tokens)

        labels = labels.masked_fill(labels == -100, 0)

        # Get the target numbers
        y = self.nvocab[labels]  # Shape: (batch_size, seq_len)

        # if y is nan, produce one hot with just nans, else produce one hot
        valid_positions = ~torch.isnan(y)
        target_distributions = F.one_hot(labels, num_classes=len(self.nvocab)).float()

        nan_mask = ~valid_positions.unsqueeze(-1).expand_as(target_distributions)
        target_distributions = target_distributions.masked_fill(nan_mask, float('nan'))

        target_distributions = target_distributions[:, :, self.number_token_indices]

        wasserstein_distance = self._calculate_1d_wasserstein_dist(softmaxed, target_distributions)

        return torch.mean(wasserstein_distance[valid_positions])

    def _calculate_1d_wasserstein_dist(self, predictions, labels):
        """
        Compute the 1D Wasserstein distance (Earth Mover's Distance) between two distributions X and Y.

        Parameters:
        - X: Tensor of shape [n], distribution X (e.g., softmax probabilities or normalized histograms).
        - Y: Tensor of shape [n], distribution Y (e.g., target distribution).

        Returns:
        - Wasserstein distance (scalar).
        """
        # Ensure X and Y have the same shape
        if predictions.shape != labels.shape:
            raise ValueError("Expecting equal shapes for X and Y!")

        # Compute the cumulative distributions (CDFs)
        cdf_X = torch.cumsum(predictions, dim=-1)
        cdf_Y = torch.cumsum(labels, dim=-1)

        # Compute the Wasserstein distance as the L1 distance between the two CDFs
        wasserstein_dist = torch.sum(torch.abs(cdf_X - cdf_Y), dim=-1)

        return wasserstein_dist
