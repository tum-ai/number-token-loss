import torch
import torch.nn as nn
from ntl.loss_functions.number_token_loss import NumberTokenLoss



class NumberTokenLossWrapper(nn.Module):
    """
    A wrapper class that adds number token loss functionality to any HuggingFace model.
    """
    def __init__(self, number_token_loss: NumberTokenLoss = None, model_is_decoder: bool = False):
        super().__init__()
        self.number_token_loss = number_token_loss
        self.model_is_decoder = model_is_decoder

    def forward(
        self, outputs, labels, num_items_in_batch
    ) -> torch.FloatTensor:
        # TODO double check this
        if self.model_is_decoder:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = outputs.logits
            shift_labels = labels
        number_token_loss = self.number_token_loss.forward(shift_logits, shift_labels)
        outputs["number_loss"] = number_token_loss
        outputs["token_loss"] = outputs.loss
        outputs.loss = outputs.loss + self.number_token_loss.weight * number_token_loss
    
        return outputs.loss
        
      
