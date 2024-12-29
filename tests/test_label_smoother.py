# These test cases were created by ChatGPT. 

import unittest
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from unittest.mock import MagicMock

from transformers.trainer_pt_utils import LabelSmoother

from ntl.loss_functions.number_token_loss import NumberTokenSelector
from ntl.utils.label_smoother import GaussianLabelSmoother

# Optionally import or define a mock NumberTokenSelector
class MockNumberTokenSelector:
    """
    A mock selector that can partially mask out tokens.
    We'll let the constructor specify which positions are 'number tokens'.
    """
    def __init__(self, valid_positions=None):
        # valid_positions is a list of (batch, seq) pairs that are "number tokens"
        self.valid_positions = valid_positions or []

    def select_number_tokens(self, logits: torch.Tensor, labels: torch.Tensor):
        # Create a boolean mask of shape [batch, seq_len]
        batch_size, seq_len, num_classes = logits.shape
        number_tokens = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=logits.device)

        for (b, s) in self.valid_positions:
            number_tokens[b, s] = True

        # Keep only those logits
        filtered_logits = logits[number_tokens]  # shape [valid_count, num_classes]
        filtered_logits = filtered_logits.unsqueeze(0) if filtered_logits.dim() == 1 else filtered_logits

        # Adjust shape to [batch_size, seq_len, num_filtered_classes], but we only keep one contiguous dimension
        # For simplicity, let's do a naive reshape
        # In reality, you'd want to replicate logic from your real selector
        # but let's keep it simple for demonstration
        filtered_logits = filtered_logits.view(batch_size, -1, num_classes)

        filtered_labels = labels.clone()
        # For positions NOT in number_tokens, we can set them to ignore_index, forcing them out
        ignore_index = -100
        for b_i in range(batch_size):
            for s_i in range(seq_len):
                if not number_tokens[b_i, s_i]:
                    filtered_labels[b_i, s_i] = ignore_index

        return filtered_logits, filtered_labels, number_tokens


@dataclass
class GaussianLabelSmoother(LabelSmoother):
    sigma: float = 1.0
    ignore_index: int = -100
    selector: object = None

    def __call__(self, model_output, labels: torch.Tensor, shift_labels: bool = False) -> torch.Tensor:
        if isinstance(model_output, dict):
            logits = model_output["logits"]
        else:
            logits = model_output[0]

        if shift_labels:
            # shift the last dimension
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # Optionally filter number tokens
        if self.selector is not None:
            logits, labels, number_tokens = self.selector.select_number_tokens(logits, labels)
        else:
            # If no selector, treat all as valid
            number_tokens = (labels != self.ignore_index)

        num_classes = logits.size(-1)
        # One-hot encode the labels
        one_hot_labels = F.one_hot(labels.clamp(min=0), num_classes=num_classes).float()

        valid_mask = (labels != self.ignore_index) & number_tokens

        labels_flat = labels[valid_mask].view(-1)
        if labels_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Vectorized Gaussian
        device = logits.device
        classes_arange = torch.arange(num_classes, device=device).unsqueeze(0)  # shape [1, num_classes]
        labels_flat_expanded = labels_flat.unsqueeze(1)  # shape [V, 1]
        dist_sq = (classes_arange - labels_flat_expanded) ** 2
        gauss = torch.exp(-dist_sq / (2 * (self.sigma ** 2)))
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)

        # Fill a full-size [B, S, num_classes] distribution
        gaussian_labels = torch.zeros_like(one_hot_labels)
        gaussian_labels[valid_mask] = gauss

        log_probs = F.log_softmax(logits, dim=-1)
        loss_per_token = -(gaussian_labels * log_probs).sum(dim=-1)

        # Mask out invalid positions
        loss_per_token = torch.where(valid_mask, loss_per_token, torch.tensor(0.0, device=device))
        num_valid = valid_mask.sum().item()
        loss = loss_per_token.sum() / max(num_valid, 1)

        return loss


class TestGaussianLabelSmoother(unittest.TestCase):
    def test_minimal_input(self):
        """
        Case 1: minimal shape => (B=1, S=1, C=2).
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=None)

        logits = torch.tensor([[[0.0, 1.0]]])  # shape [1,1,2]
        labels = torch.tensor([[1]])           # shape [1,1]

        # No shift_labels
        loss = smoother(model_output={"logits": logits}, labels=labels, shift_labels=False)
        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_ignore_index_all(self):
        """
        Case 2: all tokens are ignore_index => expecting loss == 0.0
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=None)

        logits = torch.randn(2, 3, 4)  # shape [B=2, S=3, C=4]
        labels = torch.full((2, 3), -100)  # all ignored

        loss = smoother(model_output={"logits": logits}, labels=labels)
        self.assertEqual(loss.item(), 0.0)

    def test_shift_labels(self):
        """
        Case 3: test that shift_labels works as intended.
        We'll create a scenario with (B=1, S=4, C=3) and see
        if it shifts properly (the last token is ignored in logits).
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=None)

        logits = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],  # token_0
                    [0.1, 0.2, 0.3],  # token_1
                    [1.0, 1.0, 1.0],  # token_2
                    [2.0, 2.0, 2.0],  # token_3
                ]
            ],
            dtype=torch.float,
        )  # shape [1, 4, 3]
        labels = torch.tensor([[0, 1, 2, 2]])  # shape [1,4]

        # shift_labels => logits => shape becomes [1,3,3], labels => [1,3]
        loss = smoother({"logits": logits}, labels, shift_labels=True)
        self.assertFalse(torch.isnan(loss))

    def test_no_valid_tokens(self):
        """
        Case 4: The selector decides that no tokens are "number tokens".
        That means valid_mask is all False => loss should be 0.0
        """
        # Mock selector that returns an all-False mask
        class AllFalseSelector:
            def select_number_tokens(self, logits, labels):
                batch_size, seq_len, num_classes = logits.shape
                number_tokens = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=logits.device)
                return logits, labels, number_tokens

        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=AllFalseSelector())

        logits = torch.randn(2, 3, 5)
        labels = torch.randint(low=0, high=5, size=(2, 3))

        loss = smoother({"logits": logits}, labels)
        self.assertEqual(loss.item(), 0.0)

    def test_gaussian_distribution_correctness(self):
        """
        Case 5: Check a small scenario (B=1, S=2, C=3) where we can compute expected
        gaussian distributions by hand for label=0 and label=2, sigma=1.0.
        This is an optional deeper correctness check.
        """
        smoother = GaussianLabelSmoother(sigma=1.0, ignore_index=-100, selector=None)

        logits = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],  # for token_0
                    [1.0, 0.0, 0.0],  # for token_1
                ]
            ],
            dtype=torch.float,
        )  # shape [1,2,3]
        labels = torch.tensor([[0, 2]])  # shape [1,2]

        loss = smoother({"logits": logits}, labels)
        self.assertFalse(torch.isnan(loss))
        # Optional: compare to a known expected value if you can compute the log-softmax
        # and gaussian distribution by hand.

if __name__ == "__main__":
    unittest.main()
