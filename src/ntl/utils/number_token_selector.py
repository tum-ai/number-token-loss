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

        # Extract indices and values of number tokens
        self.number_token_mask = ~torch.isnan(self.nvocab)
        self.number_token_indices = torch.nonzero(self.number_token_mask, as_tuple=False).squeeze()

        self.number_token_values = self.nvocab[self.number_token_indices]

    def select_number_tokens(self, logits: Tensor):
        # Create a mask to filter out non-digit tokens and labels
        logits = logits[:, :, self.number_token_mask]
        return logits, self.number_token_mask


