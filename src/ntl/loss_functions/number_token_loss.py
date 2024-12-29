import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class NumberTokenSelector:
    '''
    Select number tokens 
    '''
    
    def __init__(self, tokenizer: NumberEncodingTokenizer, nvocab):
        self.tokenizer = tokenizer
        self.nvocab = nvocab
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
        
        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(token, ignore_order=True)


    def select_number_tokens(self, logits: Tensor, labels: Tensor):
        
        # Create a mask to filter out non-digit tokens and labels
        number_tokens = ~torch.isnan(self.nvocab)
        logits = logits[:, :, number_tokens] 
        labels = labels.masked_fill(labels == -100, 0)

        return logits, labels, number_tokens



class NumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, loss_function=F.mse_loss, weight=0.5):
        self.loss_function = loss_function
        self.weight = weight
        
        # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        self.selector = NumberTokenSelector(tokenizer, self.nvocab, device)



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