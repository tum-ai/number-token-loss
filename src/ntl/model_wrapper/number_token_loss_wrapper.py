import torch
import torch.nn as nn
from ntl.loss_functions.number_token_loss import NumberTokenLoss



class NumberTokenLossWrapper(nn.Module):
    """
    A wrapper class that adds number token loss functionality to any HuggingFace model.
    """
    def __init__(self, number_token_loss: NumberTokenLoss = None):
        super().__init__()
        self.number_token_loss = number_token_loss

    def forward(
        self, outputs, labels, num_items_in_batch
    ) -> torch.FloatTensor:
        number_token_loss = self.number_token_loss.forward(outputs.logits, labels)
        
        outputs["number_loss"] = number_token_loss
        outputs["token_loss"] = outputs.loss
        outputs.loss = outputs.loss + self.number_token_loss.weight * number_token_loss
    
        return outputs.loss
        
      
