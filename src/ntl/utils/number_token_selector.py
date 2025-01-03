import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer

class NumberTokenSelector:
    '''
    Select number tokens 
    '''
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size, device): # nvocab):
        self.tokenizer = tokenizer
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

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


