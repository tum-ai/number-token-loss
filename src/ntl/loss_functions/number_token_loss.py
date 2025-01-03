import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from ntl.utils.number_token_selector import NumberTokenSelector


class NumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, loss_function=F.mse_loss, weight=0.5):
        self.loss_function = loss_function
        self.weight = weight
        
        # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        # self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        self.selector = NumberTokenSelector(tokenizer, vocab_size, device) # self.nvocab)
        self.nvocab = self.selector.nvocab # torch.full((vocab_size,), float("nan"), device=device) 



    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        logits, labels, number_tokens = self.selector.select_number_tokens(logits, labels)

        # Compute the weighted average of number tokens (yhat)
        softmaxed = F.softmax(logits, dim=-1)
        yhat = torch.sum(softmaxed * self.nvocab[number_tokens], dim=-1)
        y = self.nvocab[labels]

        loss = self.loss_function(yhat[~torch.isnan(y)], y[~torch.isnan(y)])
        return loss