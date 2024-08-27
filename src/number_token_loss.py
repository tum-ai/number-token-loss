import torch
import torch.nn.functional as F
from torch._tensor import Tensor
from src.encoding_decoding.numerical_encodings import encoding_to_number
from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class NumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, device, loss_order=2, weight=0.5):
        self.tokenizer = tokenizer
        self.order = loss_order
        self.weight = weight
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
        self.nvocab = torch.tensor(
            [self.tokenizer.decode_number_token(token) if token in hashed_num_tokens else float('nan') for token in self.tokenizer.get_vocab()],
            dtype=torch.float32,
            device=device
        )

    def forward(self, logits: Tensor, labels: Tensor):

        # Create a mask to filter out non-digit tokens
        number_tokens = ~torch.isnan(self.nvocab)
        logits = logits[:, :, number_tokens]

        # Compute the weighted average of number tokens (yhat)
        softmaxed = F.softmax(logits, dim=-1)
        yhat = torch.sum(softmaxed * self.nvocab[number_tokens], dim=-1)
        y = self.nvocab[labels]

        # Compute the final loss function
        loss = torch.nanmean((torch.abs(y - yhat) ** self.order)) ** (1 / self.order)
        return float(loss)
