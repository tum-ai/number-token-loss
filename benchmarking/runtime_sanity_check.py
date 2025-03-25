import time
import logging
import torch
import numpy as np
from ntl.loss_functions.base_number_token_loss import NumberTokenLoss, CEWithNTL
from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from torch.nn import CrossEntropyLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def everythink_as_number_token(ntl_loss: NumberTokenLoss, device: str) -> NumberTokenLoss:

    ntl_loss.nt_vals = torch.full((len(ntl_loss.tokenizer.get_vocab()),), float(1)).to(device)
    ntl_loss.is_number_token = ~torch.isnan(ntl_loss.nt_vals)
    ntl_loss.nt_vals_dense = ntl_loss.nt_vals[ntl_loss.is_number_token].to(device)

    return ntl_loss


def benchmark(loss_fn, time_steps, vocab_size, batch_size, device, permute = False):

    # Clear cache
    torch.cuda.empty_cache()

    # Generate random input
    logits = torch.randn(batch_size, time_steps, vocab_size).to(device)
    labels = torch.randint(vocab_size, (batch_size, time_steps)).to(device)

    # Permute shape for ce loss
    if permute: logits = logits.permute(0, 2, 1)

    # Sync and start timer
    if device == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()

    # Compute Loss
    loss = loss_fn(logits, labels)

    # Sync and stop timer
    if device == "cuda": torch.cuda.synchronize()
    end = time.perf_counter()

    # Ensure loss is not NaN
    assert not torch.isnan(loss).any(), "Loss is NaN!"

    return end - start

def runtime_measurement(time_steps, batch_size, num_batches):
    
    # Setup  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")
    vocab_size = len(tokenizer)
    
    ce_loss = CrossEntropyLoss(ignore_index=-100).to(device)
    ntl = NumberTokenLoss(tokenizer, device)
    ce_with_ntl = CEWithNTL(tokenizer, device)

    ntl = everythink_as_number_token(ntl, device)
    ce_with_ntl.ntl = everythink_as_number_token(ce_with_ntl.ntl, device)

    # Get number token ids
    # ntl_number_token_ids = (~torch.isnan(ntl.nt_vals)).nonzero().squeeze()
    # ce_with_ntl_number_token_ids = (~torch.isnan(ce_with_ntl.ntl.nt_vals)).nonzero().squeeze()

    ce_times = []
    ntl_times = []
    ce_with_ntl_times = []

    for _ in range(num_batches):

        ce_time = benchmark(ce_loss, time_steps, vocab_size, batch_size, device, permute=True)
        ntl_time = benchmark(ntl, time_steps, vocab_size, batch_size, device)
        ce_with_ntl_time = benchmark(ce_with_ntl, time_steps, vocab_size, batch_size, device)

        ce_times.append(ce_time)
        ntl_times.append(ntl_time)
        ce_with_ntl_times.append(ce_with_ntl_time)


    # Calculate mean and standard deviation
    ce_time = np.mean(ce_times)
    ce_time_std = np.std(ce_times)
    ntl_time = np.mean(ntl_times)
    ntl_time_std = np.std(ntl_times)
    ce_with_ntl_time = np.mean(ce_with_ntl_times)
    ce_with_ntl_time_std = np.std(ce_with_ntl_times)

    logger.info(f"CrossEntropyLoss: ({ce_time*1000:.2f} ± {ce_time_std*1000:.2f})ms")
    logger.info(f"NumberTokenLoss: ({ntl_time*1000:.2f} ± {ntl_time_std*1000:.2f})ms")
    logger.info(f"CEWithNTL: ({ce_with_ntl_time*1000:.2f} ± {ce_with_ntl_time_std*1000:.2f})ms")


if __name__ == "__main__":
    runtime_measurement(time_steps = 10, batch_size = 32, num_batches = 10)

# Output:
# [03/25/2025 22:43:40] - [INFO] - CrossEntropyLoss: (33.40 ± 0.13)ms
# [03/25/2025 22:43:40] - [INFO] - NumberTokenLoss: (0.45 ± 0.02)ms
# [03/25/2025 22:43:40] - [INFO] - CEWithNTL: (33.73 ± 0.23)ms