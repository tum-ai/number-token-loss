from torch import nn


class Numformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=6,
        num_layers=6,
        dim_feedforward=3072,
        dropout=0.1,
        activation=nn.GELU(),
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        transformer_bias=False,
        numhead_bias=True,
        context_length=1024,
        is_causal=False,
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            # bias=transformer_bias,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder, num_layers=num_layers, enable_nested_tensor=False
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(context_length, d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )
        self.num_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )
        self.is_causal = is_causal

    def forward(self, x, x_num):
        x = self.token_embed(x) * x_num.unsqueeze(-1)
        x = x + self.position_embed.weight[: x.shape[1]].unsqueeze(0)
        x = self.encoder_stack(x, is_causal=self.is_causal)
        logit_preds = self.lm_head(x)
        num_preds = self.num_head(x)
        return logit_preds, num_preds

