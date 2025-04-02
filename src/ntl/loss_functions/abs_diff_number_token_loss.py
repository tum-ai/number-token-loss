import torch
import torch.nn.functional as F
from torch._tensor import Tensor
import re

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class AbsDiffNumberTokenLoss:
    """
    Loss function for numberical tokens based on the weighted absolute difference between true and predicted number
    NOTE: This loss is equivalent to the Wasserstein distance as long as the ground truth distribution is one-hot
    """

    def __init__(
            self, 
            tokenizer: NumberEncodingTokenizer, 
            vocab_size: int, 
            device, 
            loss_function=F.mse_loss, 
            weight=0.5, 
            weight_by_probablility_mass: bool = False
    ):
        self.tokenizer = tokenizer
        self.loss_function = loss_function
        self.weight = weight
        self.weight_by_probablility_mass = weight_by_probablility_mass
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())

        # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(token, ignore_order=True)

        self.number_tokens = ~torch.isnan(self.nvocab)
        self.number_values = self.nvocab[self.number_tokens]

    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")


        labels = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)

        # Create a mask to filter out non-digit tokens
        y = self.nvocab[labels]
        valid_positions = ~torch.isnan(y)

        # Check that each sample has at least one valid position
        # batch_valid = valid_positions.any(dim=1)
        # if not batch_valid.all():
        #     raise ValueError("Each sample in the batch must have at least one valid number token position")
        
        # number_pattern = r'^-?\d+\.?\d*$'
        # for batch_idx in range(len(labels)):
        #     decoded_labels_list = [self.tokenizer.decode(label) for label in labels[batch_idx]]
        #     for sample_idx, label in enumerate(decoded_labels_list):
        #         if re.match(number_pattern, label):
        #             if not valid_positions[batch_idx, sample_idx]:
        #                 raise ValueError(f"Found number token in labels that was not marked as valid position: {label}")

        if self.weight_by_probablility_mass:
            # Calculate softmax probabilities for all tokens
            all_token_probs = F.softmax(logits, dim=-1)
            
            # Calculate softmax probabilities for number tokens
            number_token_probs = all_token_probs[:, :, self.number_tokens]
            
            # Calculate the ratio of probability mass assigned to number tokens
            # Sum across the number token dimension to get total probability per position
            number_mass = number_token_probs.sum(dim=-1)
            # Total probability mass is 1.0 per position (due to softmax)
            mass_ratio = number_mass.clone()
            mass_ratio[valid_positions] = number_mass[valid_positions]
        
        
        # Get the logits for just the number tokens
        logits_number = logits[:, :, self.number_tokens]
        softmax_probs = F.softmax(logits_number, dim=-1)

        # compute absolute difference between the true numbers and all possible number values
        abs_diff = torch.abs(y[valid_positions].unsqueeze(-1) - self.number_values)

        # loss is the absolute difference weighted by the softmax probs
        position_loss = (abs_diff * softmax_probs[valid_positions]).sum(axis=-1)
        
        if self.weight_by_probablility_mass:
            # Weight loss by the percentage of mass assigned to number tokens
            weighted_position_loss = position_loss * mass_ratio[valid_positions]
        else:
            weighted_position_loss = position_loss

        return torch.mean(weighted_position_loss)
