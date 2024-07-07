from torch import nn
from transformers import XLNetModel


class XLNetBackbone(nn.Module):
    def __init__(self, model_name):
        super(XLNetBackbone, self).__init__()
        self.transformer = XLNetModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size

    def forward(self, embeddings, attention_mask, token_type_ids):
        outputs = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        return sequence_output
