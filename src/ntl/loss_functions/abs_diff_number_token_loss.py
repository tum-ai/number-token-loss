import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class AbsDiffNumberTokenLoss:
    """
    Loss function for numberical tokens based on the weighted absolute difference between true and predicted number
    NOTE: This loss is equivalent to the Wasserstein distance as long as the ground truth distribution is one-hot
    """

    def __init__(
            self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, loss_function=F.mse_loss, weight=0.5
    ):
        self.tokenizer = tokenizer
        self.loss_function = loss_function
        self.weight = weight
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

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        y = self.nvocab[labels]
        valid_positions = ~torch.isnan(y)

        # apply softmax and get number labels
        logits = logits[:, :, self.number_tokens]
        softmax_probs = F.softmax(logits, dim=-1)

        # compute absolute difference between the true numbers and all possible number values
        abs_diff = torch.abs(y[valid_positions].unsqueeze(-1) - self.number_values)

        # loss is the absolute difference weighted by the softmax probs
        loss = (abs_diff * softmax_probs[valid_positions]).sum(axis=-1)

        return torch.mean(loss)
