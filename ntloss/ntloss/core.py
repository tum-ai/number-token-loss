


from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch._tensor import Tensor
from transformers import PreTrainedTokenizer

from .utils import is_number


class AbstractNTLoss(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer):
        """
        NTL constructor.

        Args:
            tokenizer: standard HF tokenizer
            
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.setup_number_tokens()

    def setup_number_tokens(self):
        """Setting up attributes needed by NT loss"""

        # Add digits to vocab if not there yet.
        self.tokenizer.add_tokens(list(map(str, range(10))))
        vocab = self.tokenizer.get_vocab()
        self.number_values = torch.full((len(vocab),), float("nan"))

        # Try to convert each token to a float after stripping the space prefix
        for token, id in vocab.items():
            if is_number(token, finite=True):
                # NOTE: This check ensures number token value only occurs for digits, not for multi-digit numbers (123)
                # This stabilizes training with NTL. Can be altered though, see paper experiments.
                if -1 <= float(token) <= 9 and len(token.lstrip(" ")) == 1:
                    self.number_values[id] = float(token)

        self.is_number_token = ~torch.isnan(self.number_values)
        self.number_values_dense = self.number_values[self.is_number_token]
        logger.info(
            f"Found {len(self.number_values_dense)} number tokens in vocab: {self.number_values_dense.tolist()}"
        )

    @abstractmethod
    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> Tensor:
        ...

    def __call__(self, *args, **kwargs):
        """Alias to self.forward"""
        return self.forward(*args, **kwargs)

class NTLossDotProduct(AbstractNTLoss):
    """Class for NTL-MSE. """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer, loss_function: Callable = F.mse_loss
    ):
        """
        NTL constructor.

        Args:
            tokenizer: NTLTokenizer with necessary attributes like is_number_token etc.
            loss_function: Function to apply on the delta between the ground truth number
                and the obtained dot product (nt-probs * token-values).
            
        """
        super().__init__(tokenizer=tokenizer)
        self.loss_function = loss_function


    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
        
    ) -> Tensor:
        """
        Computes the NTL based on the dot product between token values and their probs.

        Args:
            logits: Tensor of shape BS x T x V
            labels: Tensor of shape BS x T
            loss_mask: Optional tensor of BS x T
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".

        Returns:
            Loss tensor
                0-D if reduction=="mean"|"sum"
                BS x T if reduction=="none"
        """
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLossMSE are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLossMSE are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        y = self.number_values[labels]
        valid_positions = ~torch.isnan(y)

        # Apply the loss_mask to lower importance of number tokens before the final answer
        label_mask = (
            loss_mask[valid_positions]
            if loss_mask is not None
            else torch.ones(y.size(), dtype=logits.dtype, device=labels.device)[valid_positions]
        )

        # If no digit tokens in batch, or total of the relevant loss_mask is zero, no need for upcoming calculations
        if (torch.count_nonzero(valid_positions) == 0) or (
            torch.count_nonzero(label_mask) == 0
        ):
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=labels.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(valid_positions)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        bs, seq_len, _ = logits.size()
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # compute the weighted average of number tokens
        yhat = torch.sum(
            softmax_probs[valid_positions] * self.number_values_dense, dim=-1
        )

        # Apply specified loss function to y and yhat
        loss = self.loss_function(yhat, y[valid_positions], reduction="none")


        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(
                loss.flatten(), label_mask.flatten()
            ) / torch.count_nonzero(label_mask)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), label_mask.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(valid_positions.view(-1).size()).to(loss.device)
            loss_[valid_positions.view(-1)] = loss * label_mask
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~valid_positions]) == 0, (
                "NumberTokenLossMSE computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss




class NTLoss(AbstractNTLoss):
    """Class for Wasserstein-based NTLoss. This is the default as per our paper."""
    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_mask: Optional[Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100
    ) -> Tensor:
        """
        Computes the NTL.

        Args:
            logits: Tensor of shape BS x T x V
            labels: Tensor of shape BS x T
            loss_mask: Optional tensor of BS x T
            reduction: Optional string specifying the reduction to apply to the
                output. Defaults to "mean", options are "mean", "sum", "none".

        Returns:
            Loss tensor
                0-D if reduction=="mean"|"sum"
                BS x T if reduction=="none"

        """

        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        labels = labels.clone().masked_fill(labels == ignore_index, 0)

        # Create a mask to filter out non-digit tokens
        y = self.number_values[labels]
        valid_positions = ~torch.isnan(y)

        # Apply the loss_mask to lower importance of number tokens before the final answer
        label_mask = (
            loss_mask[valid_positions]
            if loss_mask is not None
            else torch.ones(y.size(), dtype=logits.dtype, device=labels.device)[valid_positions]
        )

        # If no digit tokens in batch, or total of the relevant loss_mask is zero, no need for upcoming calculations
        if (torch.count_nonzero(valid_positions) == 0) or (
            torch.count_nonzero(label_mask) == 0
        ):
            if (reduction == "mean") | (reduction == "sum"):
                loss = torch.tensor(0, dtype=labels.dtype, device=labels.device)
            elif reduction == "none":
                loss = torch.zeros_like(valid_positions)
            else:
                raise ValueError(f"{reduction} is not a valid value for reduction")

            return loss

        # apply softmax and get number labels
        bs, seq_len, _ = logits.size()
        nt_logits = logits[:, :, self.is_number_token]
        softmax_probs = F.softmax(nt_logits, dim=-1)

        # compute absolute difference between the true numbers and all possible number values
        abs_diff = torch.abs(
            y[valid_positions].unsqueeze(-1) - self.number_values_dense
        )

        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[valid_positions]).sum(axis=-1)

        if reduction == "mean":
            # Mean pooling (weighted by loss mask)
            loss = torch.dot(
                loss.flatten(), label_mask.flatten()
            ) / torch.count_nonzero(label_mask)
        elif reduction == "sum":
            loss = torch.dot(loss.flatten(), label_mask.flatten())
        elif reduction == "none":
            # Cast loss for number tokens back to Tensor of size BS x T
            loss_ = torch.zeros(valid_positions.view(-1).size()).to(loss.device)
            loss_[valid_positions.view(-1)] = loss * label_mask
            loss = loss_.view(bs, seq_len)

            assert torch.sum(loss[~valid_positions]) == 0, (
                "NumberTokenLoss computed for non-digit tokens!"
            )

        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        return loss


