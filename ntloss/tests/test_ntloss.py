import math
import os

import numpy as np
import pytest
import torch
from torch import Tensor
from transformers import AutoTokenizer

from ntloss import NTLoss, NTLossDotProduct


def pytest_configure(config):
    # register custom mark
    config.addinivalue_line(
        "markers", "skip_on_ci: skip tests when running in GitHub Actions"
    )

@pytest.fixture(scope="session")
def t5_tokenizer():
    return AutoTokenizer.from_pretrained("t5-small")

@pytest.fixture(scope="session")
def vocab_size(t5_tokenizer):
    # config = AutoConfig.from_pretrained("t5-small")  # not actually used below
    return t5_tokenizer.vocab_size

@pytest.fixture
def skip_on_ci(request):
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipped on GitHub Actions due to high memory usage")

def make_logits(vocab_size, token_logit_value_dicts):
    """
    Build a (1 x T x V) tensor filled with -inf, 
    then set the logits specified in token_logit_value_dicts.
    """
    T = len(token_logit_value_dicts)
    logits = torch.full((1, T, vocab_size), -np.inf, dtype=torch.float32)
    for i, tok_dict in enumerate(token_logit_value_dicts):
        for tok_id, logit in tok_dict.items():
            logits[0, i, tok_id] = logit
    return logits

@pytest.mark.usefixtures("skip_on_ci")
@pytest.mark.parametrize("LossClass", [NTLoss, NTLossDotProduct])
@pytest.mark.parametrize("logits_dicts,label_tokens", [
    # positive logits scenario
    (
        [
            {"1": 1.0, "2": 1.2, "0": 0.5, "3": 1.5},
            {"1": 1.5, "2": 1.2, "0": 0.5, "3": 1.5},
            {"1": 1.0, "2": 1.2, "0": 0.5, "3": 1.5},
        ],
        ["1", "1", "a"],
    ),
    # mixed logits scenario
    (
        [
            {"0": -4.0, "1": 2.0, "2": -1.0},
            {"0": 1.5,  "1": 0.5, "2": 1.2},
            {"3": -2.0, "4": 1.0, "5": -2.5},
        ],
        ["2", "1", "3"],
    ),
])
def test_ntloss_variants(t5_tokenizer, vocab_size, LossClass, logits_dicts, label_tokens):
    # convert token strings to IDs
    convert = t5_tokenizer.convert_tokens_to_ids
    token_logit_value_dicts = {
        # map token strings to token IDs upfront
        i: {convert(tok): val for tok, val in logits_dicts[i].items()}
        for i in range(len(logits_dicts))
    }
    # build logits tensor
    # reorder into a list for our helper
    logits_list = [token_logit_value_dicts[i] for i in range(len(logits_dicts))]
    logits = make_logits(vocab_size, logits_list)
    
    # build labels tensor shape (1 x T)
    label_ids = [convert(tok) for tok in label_tokens]
    labels = torch.tensor([label_ids], dtype=torch.long)
    
    # instantiate and run
    loss_fn = LossClass(tokenizer=t5_tokenizer)
    loss = loss_fn(logits, labels)
    
    # assertions
    assert isinstance(loss, Tensor), "Loss should be a Python float"
    assert isinstance(loss.item(), float), "Loss should be a Python float"
    assert not math.isnan(loss), "Loss must not be NaN"
