import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from transformers.trainer_pt_utils import LabelSmoother
from ntl.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss


@dataclass
class GaussianLabelSmoother:
    """
    A label smoother that applies Gaussian smoothing ONLY to number tokens, as
    selected by `NumberTokenSelector`. Non-number tokens remain untouched or masked out.
    If sigma=0, this label smoother behaves identically to standard cross-entropy loss.

    Args:
        sigma (float, optional, defaults to 1.0):
            The standard deviation for the Gaussian around the correct label.
        ignore_index (int, optional, defaults to -100):
            The index in the labels to ignore (e.g., padding or special tokens). Inherited from `LabelSmoother`. 
        selector (NumberTokenSelector, optional):
            A selector to filter out tokens that are not recognized as numbers. 
    """

    sigma: float = 1.0
    ignore_index: int = -100
    selector: object = None  # Instance of `NumberTokenSelector`
    eps = 1e-8  # epsilon
    number_token_loss: WassersteinNumberTokenLoss = None  # Only works with this instantiation of the Label Smoother

    def __call__(self, model_output, labels: Tensor, shift_labels: bool = False) -> Tensor:
        """
        Compute the Gaussian-smoothed cross-entropy loss.
        
        Parameters: 
            model_output: torch.Tensor or Dict[str, torch.Tensor]
                The model output logits or a dictionary containing the logits.
            labels: torch.Tensor of shape (batch_size, seq_len)
            shift_labels: bool
        """
        # Get logits from model output
        if isinstance(model_output, dict):
            logits = model_output["logits"] # (batch_size, seq_len, voc_size)
        else:
            logits = model_output[0] # (batch_size, seq_len, voc_size)
            
        # Handle empty logits or labels gracefully by returning zero loss
        if logits.numel() == 0 or labels.numel() == 0:
            # Return a zero that still has grad_fn
            print("requires_grad:", logits.requires_grad)
            return logits.sum() * 0.0


        # Shift labels if needed 
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # Select only number tokens for smoothing
        if self.selector is not None:
            if not hasattr(self.selector, 'nvocab'):
                raise AttributeError("The selector must have an attribute 'nvocab' representing the number of valid vocab tokens.")

            # Get the number of classes and the mask for number tokens
            tokens_encoding_numbers = torch.tensor(self.selector.number_token_indices)
            num_classes_numbers = tokens_encoding_numbers.shape[0]

            number_mask = torch.isin(labels, tokens_encoding_numbers)  # (batch_size, seq_len)

            # Get the decoded labels
            labels_dec = self.selector.nvocab[labels]

        else:
            # If no selector is given, throw an error
            raise ValueError("A NumberTokenSelector instance must be provided to select number tokens for smoothing.")

        # All labels that are not self.ignore_index
        valid_mask = labels != self.ignore_index  # (batch_size, seq_len)

        if not valid_mask.any():
            # If no valid tokens are present, return zero loss that still has grad_fn
            return logits.sum() * 0.0

        # Mask for valid number labels and non-padding tokens.
        assert number_mask.shape == valid_mask.shape, (
            "Number mask should have the same shape as valid_maks!"
        )  # (batch_size, seq_len) # should not change anything, as number_mask is already a subset of valid_mask
        non_number_mask = valid_mask * ~number_mask  # (batch_size, seq_len)

        # Compute log probabilities once for efficiency
        log_probs = F.log_softmax(logits, dim=-1)  # [B, S, C]

        # Initialize loss tensors
        loss_numbers = torch.zeros_like(labels_dec, dtype=logits.dtype, device=logits.device)  # (batch_size, seq_len)
        loss_non_numbers = torch.zeros_like(
            labels_dec, dtype=logits.dtype, device=logits.device
        )  # (batch_size, seq_len)

        # Compute loss for number tokens
        if number_mask.any():
            if self.sigma == 0.0:
                # When sigma is zero, use one-hot labels directly without smoothing.
                # To avoid F.one_hot error, all labels outside of valid_mask are set to 0
                number_labels_filled = labels_dec.clone()
                number_labels_filled = labels_dec.masked_fill(
                    ~number_mask, 0
                )  # All non-number tokens are filled with zero
                number_one_hot = F.one_hot(number_labels_filled, num_classes=num_classes_numbers).float()
                number_one_hot = number_one_hot * number_mask.unsqueeze(-1)  # Zero out non-number tokens

                # Compute the loss for number tokens
                loss_numbers = -(number_one_hot * log_probs[..., :num_classes_numbers]).sum(dim=-1)
                
            else:      
                # Gaussian smoothing for number tokens
                # Create a tensor of number values
                number_values = self.selector.number_token_values.unsqueeze(0).unsqueeze(
                    0
                )  # (1, 1, num_classes_numbers)

                # Expand labels to shape (batch_size, seq_length, 1).  Cast to float32 if necessary
                labels_dec_expanded = labels_dec.unsqueeze(-1).float()  # (batch_size, seq_length, 1)

                # Compute Gaussian distribution around each label index
                gaussian = torch.exp(
                    -0.5 * ((number_values - labels_dec_expanded) / self.sigma) ** 2
                )  # (batch_size, seq_len//number_outputs, num_classes_numbers)

                # Normalize to ensure each (batch, output) sums to 1. Prevent division by zero
                gaussian_probs = gaussian / (gaussian.sum(dim=2, keepdim=True) + self.eps) 

                # Apply mask to Gaussian probabilities
                gaussian_probs = gaussian_probs * number_mask.unsqueeze(-1)  # Zero out non-number tokens

                # Compute the loss for number tokens
                loss_numbers = -(gaussian_probs * log_probs[..., :num_classes_numbers]).sum(dim=-1) # (batch_size, seq_len)

        # Compute loss for non-number tokens
        if non_number_mask.any():
            # One-hot encoding for non-number tokens
            non_number_labels_filled = labels.clone()
            non_number_labels_filled = non_number_labels_filled.masked_fill(~non_number_mask, 0)  # Fill non-valid tokens with 0 # (batch_size, seq_len)
            one_hot_non_num = F.one_hot(non_number_labels_filled, num_classes=logits.size(-1)).float()
            one_hot_non_num = one_hot_non_num *  non_number_mask.unsqueeze(-1).expand(-1, -1, one_hot_non_num.size(-1)) # non_number_mask.unsqueeze(-1)  # Zero out non-number tokens

            # Compute the loss for non-number tokens
            loss_non_numbers = -(one_hot_non_num * log_probs).sum(dim=-1)
                
        # Combine the two losses into a single tensor
        loss_per_token = torch.where(number_mask, loss_numbers, loss_non_numbers)  # (batch_size, seq_len)

        # Average across the valid tokens.         
        num_valid = valid_mask.sum().float()
        loss = loss_per_token.sum() / torch.clamp(num_valid, min=1.0)

        # Compute additional number token loss if available
        if self.number_token_loss is not None:
            number_token_loss = self.number_token_loss.forward(logits, labels, gaussian_probs)
            return (loss, number_token_loss)
        else:
            return loss
        
