from typing import Literal

import torch
import torch.nn.functional as F
from torch._tensor import Tensor
from transformers import PreTrainedTokenizer


class NumberTokenLoss:
    """Class for NTL."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        NTL Constructor. Setting up attributes needed by NT loss

        Args:
            tokenizer: A PreTrainedTokenizer instance.
        """
        self.tokenizer = tokenizer

        # Add digits to vocab if not there yet.
        self.tokenizer.add_tokens(list(map(str, range(10))))
        self.nt_vals = torch.full((len(self.tokenizer.get_vocab()),), float("nan"))

        # Try to convert each token to a float after stripping the space prefix and filter out super- and subscript tokens
        for token, id in self.tokenizer.get_vocab().items():
            if token.strip().isdigit() and token.strip() and token.strip().isascii():
                # NOTE: This check ensures number token value only occurs for digits, not for multi-digit numbers (123)
                # This stabilizes training with NTL.
                if -1 <= float(token) <= 9 and len(token.lstrip(" ")) == 1:
                    self.nt_vals[id] = float(token)

        self.is_number_token = ~torch.isnan(self.nt_vals)
        self.nt_vals_dense = self.nt_vals[self.is_number_token]
        print(f"Found {len(self.nt_vals_dense)} number tokens in vocab: {self.nt_vals_dense.tolist()}")

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

    def forward(self, logits: Tensor, labels: Tensor, reduction: Literal["none", "mean", "sum"] = "mean") -> Tensor:
        """
        Computes the NTL.

        Args:
            logits: Tensor of shape BS x T x V.
            labels: Tensor of shape BS x T.
            reduction: Reduction scheme for loss aggregation.

        Returns:
            Loss tensor.

        """
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        y = self.nt_vals[labels]
        valid_positions = ~torch.isnan(y)

        # apply softmax and get number labels
        logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(logits, dim=-1)

        # compute absolute difference between the true numbers and all possible number values
        abs_diff = torch.abs(y[valid_positions].unsqueeze(-1) - self.nt_vals_dense)
        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[valid_positions]).sum(axis=-1)

        if reduction == "none":
            return loss
        elif reduction == "sum":
            return torch.sum(loss)
        elif reduction == "mean":
            return torch.mean(loss)
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'.")


class CEWithNTL:
    """Class for CEWithNTL."""

    def __init__(self, tokenizer: PreTrainedTokenizer, ntl_weight: float = 0.5):
        """
        CEWithNTL Constructor. Setting up attributes needed by CEWithNTL loss

        Args:
            tokenizer: A PreTrainedTokenizer instance.
            ntl_weight: Weighting factor for CE and NTL.
        """
        self.tokenizer = tokenizer
        self.ntl = NumberTokenLoss(tokenizer)
        self.ntl_weight = ntl_weight

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tensor:
        """
        Computes the CEWithNTL.

        Args:
            logits: Tensor of shape BS x T x V.
            labels: Tensor of shape BS x T.
            reduction: Reduction scheme for loss aggregation.

        Returns:
            Loss tensor.

        """
        ce_loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction=reduction)
        ntl_loss = self.ntl(logits, labels, reduction=reduction)
        return ce_loss + self.ntl_weight * ntl_loss