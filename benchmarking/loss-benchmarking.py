import time
import random
import string
import logging
import csv
import shutil
import copy

import torch
import numpy as np
from torch.optim import AdamW
import yaml

from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from ntl.transformer_backbone.t5.t5_vanilla_for_number_token_loss import T5VanillaForNumberTokenLoss
from ntl.loss_functions.number_token_loss import NumberTokenLoss
from ntl.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss
from ntl.loss_functions.abs_diff_number_token_loss import AbsDiffNumberTokenLoss

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

CUDA_DEVICE = "cuda"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Regular Cross Entropy (CE) Loss (Baseline)
# CE with NTL-MSE
# CE + NTL-Wasserstein

# 1. Measure standalone loss computation (mock tensor with fixed batch size)
# 2. Measure speed of forward pass
# 3. Measure speed of training step (including gradient update)
# Run each configuration in each setup 100 times (on GPU) and make a barplot showing the differences

class CustomLoss:
    def __init__(self, loss_functions: list):   
        self.loss_functions = loss_functions
    
    def forward(self, logits, labels):
        total_loss = 0
        for loss_function in self.loss_functions:
            total_loss += loss_function.forward(logits, labels)
        return total_loss

class CustomCrossEntropyLoss:
    def forward(self, logits, labels):
        batch_size, seq_len, num_classes = logits.shape

        assert labels.shape == (batch_size, seq_len), "The shapes of the labels and logits do not match."
        return torch.nn.functional.cross_entropy(logits.view(-1, num_classes), labels.view(-1))

def generate_inputs(batch_size, sequence_length, vocab_size, device):
    ntl_logits = torch.randn(batch_size, sequence_length, vocab_size, requires_grad=True, device=device)
    ntl_labels = torch.randint(vocab_size, (batch_size, sequence_length), device=device)

    return ntl_logits, ntl_labels

def get_random_letter():
    return random.choice(string.ascii_letters + " ")

def get_random_number():
    return random.choice(string.digits)

def find_random_input(tokenizer, n_tokens, number_share):
    text = ""

    token_count = 0
    loop_count = 0

    while(token_count < n_tokens and loop_count < 2 * n_tokens):
        if random.random() <= number_share:
            text += get_random_number()
        else:
            text += get_random_letter()
        token_count = len(tokenizer(text)['input_ids'])
        loop_count += 1
    

    assert token_count == n_tokens, f"Token count: {token_count}, n_tokens: {n_tokens}"
    
    return text

def generate_random_input(tokenizer, batch_size, n_tokens, number_share):
    texts = []

    for _ in range(batch_size):

        miss_count = 0
        found = False

        while(miss_count < 10 and found == False):

            try:
                texts.append(find_random_input(tokenizer, n_tokens, number_share))
                found = True
            except AssertionError as e:
                miss_count += 1
        
        if(found == False):
            raise RuntimeError(f"Failed to generate valid input after {miss_count} tries.")
    
    return texts

def generate_input_set(steps, batch_size, sequence_length, number_share, tokenizer):
    inputs = []

    for _ in range(steps):

        input_sentences = generate_random_input(tokenizer=tokenizer, 
                                        batch_size=batch_size, 
                                        n_tokens=sequence_length, 
                                        number_share=number_share
        )
        output_sentences = generate_random_input(tokenizer=tokenizer, 
                                        batch_size=batch_size, 
                                        n_tokens=sequence_length, 
                                        number_share=number_share
        )
        inputs.append((input_sentences, output_sentences))
    
    return inputs


def standalone(config, loss_func, loss_name, vocab_size, device):

    timer = Timer()

    for _ in range(config['steps']):
        logits, labels = generate_inputs(config['batch_size'],
                                         config['sequence_length'], 
                                         vocab_size, 
                                         device
        )

        timer.start()
        
        loss = loss_func.forward(logits, labels)

        timer.stop("complete_pass", device)
    
    execution_times, execution_time_stds = timer.get_results()

    logger.info(f"{loss_name} computation completed in ({execution_times['complete_pass']:.2f} ± {execution_time_stds['complete_pass']:.2f})s")
    logger.debug(f"Loss for {loss_name}: {loss.item()}")

    return execution_times, execution_time_stds

class Timer():  
    def __init__(self):
        self.start_time = -1
        self.times = {}

    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self, name, device):
        if(self.start_time == -1):
            raise RuntimeError("Timer not started.")
        
        if name not in self.times:
            self.times[name] = []

        if device == CUDA_DEVICE:
            torch.cuda.synchronize()

        self.times[name].append(time.perf_counter() - self.start_time)

    def get_results(self):
        results = {}
        results_err = {}
        
        for stop_point, times in self.times.items():
            results[stop_point] = np.mean(times)
            results_err[stop_point] = np.std(times)

        return results, results_err

def forward_pass(config, model, loss_func, loss_name, tokenizer, device, gradient_update = False):

    timer = Timer()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for _ in range(config['steps']):

        timer.start()

        input_sentences = generate_random_input(tokenizer=tokenizer, 
                                        batch_size=config['batch_size'], 
                                        n_tokens=config['sequence_length'], 
                                        number_share=config['number_share']
        )
        output_sentences = generate_random_input(tokenizer=tokenizer, 
                                        batch_size=config['batch_size'], 
                                        n_tokens=config['sequence_length'], 
                                        number_share=config['number_share']
        )

        timer.stop("input_generation", device)
        
        # Tokenize input
        input_encoding = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True)
        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        # Tokenize target
        target_encoding = tokenizer(output_sentences, return_tensors="pt", padding=True, truncation=True)
        labels = target_encoding["input_ids"].to(device)

        timer.stop("tokenization", device)

        # Replace padding token ID in labels with -100 (ignored in loss computation)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits  

        timer.stop("model_forward", device)            

        loss = loss_func.forward(logits, labels)
        loss.backward()

        timer.stop("loss_computation", device)

        # Perform one optimization step
        if(gradient_update):
            optimizer.step()
            optimizer.zero_grad()

        timer.stop("complete_pass", device)

    durations, duration_stds = timer.get_results()

    if gradient_update:
        logger.info(f"Training step with {loss_name} completed in ({durations['complete_pass']:.2f} ± {duration_stds['complete_pass']:.2f})s")
    else:
        logger.info(f"Forward pass with {loss_name} completed in ({durations['complete_pass']:.2f} ± {duration_stds['complete_pass']:.2f})s")
    
    logger.debug(f"Loss for {loss_name}: {loss.item()}")

    return durations, duration_stds

def standalone_benchmark(config, loss_functions, vocab_size, device):

    times = {}

    logger.info("Starting standalone benchmarking...")
    
    for name, loss in loss_functions.items():
        
        times[name] = standalone(
            config = config,
            loss_func=loss,
            loss_name=name,
            vocab_size=vocab_size,
            device=device
        )

    logger.info("Standalone benchmarking completed.")

    return times

def forward_pass_benchmark(config, loss_functions, tokenizer, model, device):
    
    times = {}
    logger.info("Starting forward pass benchmarking...")

    for name, loss in loss_functions.items():


        times[name] = forward_pass(
            config = config,
            model=model,
            loss_func=loss,
            loss_name=name,
            tokenizer=tokenizer,
            device = device,
            gradient_update = False
        )

    logger.info("Forward pass benchmarking completed.")

    return times

def training_step_benchmark(config, loss_functions, tokenizer, model, device):
    
    times = {}
    logger.info("Starting training step benchmarking...")
    
    for name,loss in loss_functions.items():

        times[name] = forward_pass(
            config = config,
            model=model,
            loss_func=loss,
            loss_name=name,
            tokenizer=tokenizer,
            device = device,
            gradient_update = True
        )

    logger.info("Training step benchmarking completed.")

    return times

# Prepare the data for writing rows
def prepare_rows(data):
    rows = []
    all_measurement_points = set()

    for benchmark, losses in data.items():
        for loss_function, values in losses.items():
            time_values, time_errors = values
            row = {
                "benchmark": benchmark,
                "loss": loss_function
            }
            for key in time_values:
                row[f"{key}"] = time_values[key]
                row[f"{key}_err"] = time_errors[key]
                all_measurement_points.add(key)
            rows.append(row)

    return rows, all_measurement_points

def save_results(filename, data):

    # Write to CSV
    with open(filename, mode='w', newline='') as file:
        rows, all_measurement_points = prepare_rows(data)

        # Dynamically generate fieldnames
        fieldnames = ["benchmark", "loss"]
        for point in sorted(all_measurement_points):
            fieldnames.append(f"{point}")
            fieldnames.append(f"{point}_err")

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    logger.info(f"Data has been written to {filename}")


def set_up():

    logger.info("Setup...")

    if torch.cuda.is_available():
        device = CUDA_DEVICE
    else:
        device = "cpu"

    logger.info(f"Used device: {device}")

    # Tokenizer
    tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")

    # Model 
    model = T5VanillaForNumberTokenLoss.from_pretrained("t5-small")
    model.to(device)

    # Vocab size
    pad_to_multiple_of = 64 # for model predictions 
    vocab_size = len(tokenizer) + pad_to_multiple_of - (len(tokenizer) % pad_to_multiple_of)

    # Loss
    ce_loss = CustomCrossEntropyLoss()
    mse_ntl_loss = NumberTokenLoss(tokenizer = tokenizer, 
                                   vocab_size = vocab_size,
                                   device = device
    )
    wasserstein_ntl_loss = WassersteinNumberTokenLoss(tokenizer = tokenizer, 
                                                      vocab_size = vocab_size,
                                                      order_numbers = False,
                                                      device=device
    )
    abs_diff_ntl_loss = AbsDiffNumberTokenLoss(tokenizer = tokenizer,
                                               vocab_size = vocab_size,
                                               device = device
    )

    ce_with_mse = CustomLoss([ce_loss, mse_ntl_loss])
    ce_with_wasserstein = CustomLoss([ce_loss, wasserstein_ntl_loss])
    ce_with_abs_diff = CustomLoss([ce_loss, abs_diff_ntl_loss])
    
    loss_functions = {"CE Loss": ce_loss, 
                      "CE and MSE Loss": ce_with_mse, 
                      "CE and Wassserstein Loss": ce_with_wasserstein, 
                      "CE and Abs Diff NTL Loss": ce_with_abs_diff}
    
    logger.info("Setup finished")
    
    return device, vocab_size, tokenizer, model, loss_functions

def load_config(file_path="config.yaml"):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {file_path}.")


def main(config = load_config("benchmarking/config.yaml")):
    
    times = {}
    device, vocab_size, tokenizer, model, loss_functions = set_up()

    # run up
    run_up_config = copy.deepcopy(config)
    run_up_config['standalone benchmark']['steps'] = 1
    run_up_config['forward pass benchmark']['steps'] = 1
    run_up_config['training step benchmark']['steps'] = 1

    standalone_benchmark(
        config = run_up_config['standalone benchmark'],
        loss_functions = loss_functions,
        vocab_size = vocab_size,
        device = device
    )

    # benchmark
    times["standalone"] = standalone_benchmark(
        config = config['standalone benchmark'],
        loss_functions = loss_functions,
        vocab_size = vocab_size,
        device = device
    )

    # run up
    forward_pass_benchmark(
        config = run_up_config['forward pass benchmark'],
        loss_functions = loss_functions,
        tokenizer = tokenizer,
        model = model,
        device = device
    )

    # benchmark
    times["forward pass"] = forward_pass_benchmark(
        config = config['forward pass benchmark'],
        loss_functions = loss_functions,
        tokenizer = tokenizer,
        model = model,
        device = device
    )

    # run up
    training_step_benchmark(
        config = run_up_config['training step benchmark'],
        loss_functions = loss_functions,
        tokenizer = tokenizer,
        model = model,
        device = device
    )

    # benchmark
    times["training step"] = training_step_benchmark(
        config = config['training step benchmark'],
        loss_functions = loss_functions,
        tokenizer = tokenizer,
        model = model,
        device = device
    )

    timestamp = int(time.time() * 100)

    # Save results as csv
    save_results(f'benchmarking/benchmark_results_{timestamp}.csv', times)

    # Save config object as yaml
    with open(f'benchmarking/config_{timestamp}.stored_yaml', 'w') as file:
        yaml.dump(config, file)

    
def number_share_benchmark():
    orig_config = load_config("benchmarking/config.yaml")

    for number_share in [0.1 * n for n in range(11)]:
        config = copy.deepcopy(orig_config)
        number_share = round(number_share, 1)
        config['standalone benchmark']['number_share'] = number_share
        config['forward pass benchmark']['number_share'] = number_share
        config['training step benchmark']['number_share'] = number_share
        main(config)
   
if __name__ == "__main__":
    number_share_benchmark()

# TODO: Custom loss 