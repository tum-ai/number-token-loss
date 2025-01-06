import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from transformers.trainer_pt_utils import LabelSmoother


@dataclass
class GaussianLabelSmoother(LabelSmoother):
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

    def __call__(self, model_output, labels: Tensor, shift_labels: bool = False) -> Tensor:
        """
        Compute the Gaussian-smoothed cross-entropy loss.
        """
        # Get logits from model output
        if isinstance(model_output, dict):
            logits = model_output["logits"]
        else:
            logits = model_output[0]
            
        # Handle empty logits or labels gracefully by returning zero loss
        if logits.numel() == 0 or labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Shift labels if needed 
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # Select only number tokens for smoothing
        if self.selector is not None:
            if not hasattr(self.selector, 'nvocab'):
                raise AttributeError("The selector must have an attribute 'nvocab' representing the number of valid vocab tokens.")
    
            # Select number tokens
            logits, labels, number_tokens = self.selector.select_number_tokens(logits, labels)
            
            # Get the number of classes and the mask for number tokens
            tokens_encoding_numbers = self.selector.nvocab[number_tokens]  
            num_classes = tokens_encoding_numbers.shape[0]
            labels_mask = torch.isin(labels, tokens_encoding_numbers)

        else:
            # If no selector is given, assume all are number tokens
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
            num_classes = logits.size(-1)  # Dynamic determination of num_classes 
            # raise Exception("A NumberTokenSelector needs to be provided to the GaussianLabelSmoother.")

        # Mask for valid number labels and non-padding tokens. Potentially unnecessary, as number labels certainly do not include the ignore_index. Added for safety.
        valid_mask = (labels != self.ignore_index) & labels_mask
        
        # Validation to ensure that labels are within the valid range [0, num_classes - 1]
        valid_labels = (labels[valid_mask] >= 0) & (labels[valid_mask] < num_classes)
        if not torch.all(valid_labels):
            raise RuntimeError("Some labels are out of the valid range [0, num_classes - 1].")
        
        if self.sigma == 0.0:
            # When sigma is zero, use one-hot labels directly without smoothing. 
            # To avoid F.one_hot error, all labels outside of valid_mask are set to 0
            safe_labels = labels.clone()
            safe_labels = labels * valid_mask  
            labels_to_calculate_loss = F.one_hot(safe_labels, num_classes=num_classes).float()
            
            # Zero out the labels_to_calculate_loss where not valid
            labels_to_calculate_loss = labels_to_calculate_loss * valid_mask.unsqueeze(-1)
            
        else:                  
            # Check if there are any number tokens to smooth
            if valid_mask.any():
                # Create a tensor of class indices
                class_indices = torch.arange(num_classes, device=labels.device).view(1, 1, num_classes)  # (1, 1, num_classes)
                
                # Expand labels to shape (batch_size, seq_length, 1). Cast to float32 if necessary
                labels_expanded = labels.unsqueeze(-1).float()  #  (batch_size, seq_length, 1)
                
                # Gaussian distribution around each label index:
                #    Over [0..num_classes-1] for each label l_i: 
                #       dist_j = exp(-((j - l_i)^2 / (2*sigma^2)))

                # Calculate the Gaussian probability for each class
                gaussian = torch.exp(-0.5 * ((class_indices - labels_expanded) / self.sigma) ** 2)  # (batch_size, num_outputs, num_classes)
                
                # Normalize to ensure each (batch, output) sums to 1
                gaussian_probs = gaussian / gaussian.sum(dim=2 , keepdim=True)  # [B, S, C] 
                        
                # Apply the valid mask
                labels_to_calculate_loss = gaussian_probs * valid_mask.unsqueeze(-1)
                
            else: 
                # If no valid tokens, set labels_to_calculate_loss to zero
                labels_to_calculate_loss = torch.zeros_like(logits)
                
                
        # Compute cross-entropy using smoothed label distribution
        log_probs = F.log_softmax(logits, dim=-1)  # shape [B, S, C] 
        loss_per_token = -(labels_to_calculate_loss * log_probs).sum(dim=-1) # distribution = - sum_{j} (smoothed_label_j * log_probs_j)

        # Average across the valid tokens. Also works in the case that num_valid == 0. 
        # Invalid positions are replaced with zero, ensuring that the tensor remains connected to the graph
        loss_per_token = torch.where(valid_mask, loss_per_token, torch.zeros_like(loss_per_token))
        num_valid = valid_mask.sum().float()
        loss = loss_per_token.sum() / torch.clamp(num_valid, min=1.0)
            
        return loss
        
    
    
        