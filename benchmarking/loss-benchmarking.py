import time
import random
import string
import logging
import csv
import gc
import shutil

import torch
import numpy as np
from torch.optim import AdamW
import yaml

from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from ntl.transformer_backbone.t5.t5_vanilla_for_number_token_loss import T5VanillaForNumberTokenLoss
from ntl.loss_functions.number_token_loss import NumberTokenLoss
from ntl.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss
from ntl.loss_functions.abs_diff_number_token_loss import AbsDiffNumberTokenLoss

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

class CustomCrossEntropyLoss:
    def forward(self, logits, labels):
        # Automatisch die Shapes bestimmen und dann umformen
        batch_size, seq_len, num_classes = logits.shape  # Die Form der Logits erfassen

        # Vergewissern, dass die Shapes übereinstimmen, bevor wir den Loss berechnen
        assert labels.shape == (batch_size, seq_len), "Die Shapes von Logits und Targets müssen übereinstimmen."

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
            raise Exception("Could not generate a valid input after 10 tries.")
    
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


def standalone(inputs, loss_func, loss_name):

    #gc.disable()

    times = []

    for logits, labels in inputs:
        start = time.perf_counter()
        loss = loss_func.forward(logits, labels)
        times.append(time.perf_counter() - start)
    
    execution_time = np.mean(times)
    execution_time_std = np.std(times)

    #gc.enable()

    logger.info(f"{loss_name} computation completed in ({execution_time:.2f} ± {execution_time_std:.2f})s")
    logger.debug(f"Loss for {loss_name}: {loss.item()}")

    return execution_time, execution_time_std

def forward_pass(inputs, model, loss_func, loss_name, tokenizer, device, gradient_update = False):

    times = []
    optimizer = AdamW(model.parameters(), lr=5e-5)
    #gc.disable()

    for input_sentences, output_sentences in inputs:

        start_time = time.perf_counter()
        
        # Tokenize input
        input_encoding = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True)
        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        # Tokenize target
        target_encoding = tokenizer(output_sentences, return_tensors="pt", padding=True, truncation=True)
        labels = target_encoding["input_ids"].to(device)

        # Replace padding token ID in labels with -100 (ignored in loss computation)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits              

        loss = loss_func.forward(logits, labels)
        loss.backward()

        # Perform one optimization step
        if(gradient_update):
            optimizer.step()
            optimizer.zero_grad()

        times.append(time.perf_counter() - start_time)

    duration = np.mean(times)
    duration_std = np.std(times)

    #gc.enable()

    if gradient_update:
        logger.info(f"Training step with {loss_name} completed in ({duration:.2f} ± {duration_std:.2f})s")
    else:
        logger.info(f"Forward pass with {loss_name} completed in ({duration:.2f} ± {duration_std:.2f})ss")
    
    logger.debug(f"Loss for {loss_name}: {loss.item()}")

    return duration, duration_std

def standalone_benchmark(config, loss_functions, vocab_size, device):

    times = {}

    logger.info("Starting standalone benchmarking...")
    
    for name, loss in loss_functions.items():
        
        # Generate Inputs
        single_inputs = [
            generate_inputs(config['batch_size'],
                            config['sequence_length'], 
                            vocab_size, 
                            device)
            for _ in range(config['steps'])
        ]

        times[name] = standalone(
            inputs = single_inputs,
            loss_func=loss,
            loss_name=name,
        )

    logger.info("Standalone benchmarking completed.")

    return times

def forward_pass_benchmark(config, loss_functions, tokenizer, model, device):
    
    times = {}
    logger.info("Starting forward pass benchmarking...")

    for name, loss in loss_functions.items():

        # Generate Inputs
        forward_inputs = generate_input_set(steps = config['steps'],
                                            batch_size = config['batch_size'],
                                            sequence_length = config['sequence_length'],
                                            number_share = config['number_share'],
                                            tokenizer = tokenizer
        )

        times[name] = forward_pass(
            inputs = forward_inputs,
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

        # Generate Inputs
        training_inputs = generate_input_set(steps = config['steps'],
                                            batch_size = config['batch_size'],
                                            sequence_length = config['sequence_length'],
                                            number_share = config['number_share'],
                                            tokenizer = tokenizer
        )

        times[name] = forward_pass(
            inputs = training_inputs,
            model=model,
            loss_func=loss,
            loss_name=name,
            tokenizer=tokenizer,
            device = device,
            gradient_update = True
        )

    logger.info("Training step benchmarking completed.")

    return times

class CustomLoss:
    def __init__(self, loss_functions: list):   
        self.loss_functions = loss_functions
    
    def forward(self, logits, labels):
        total_loss = 0
        for loss_function in self.loss_functions:
            total_loss += loss_function.forward(logits, labels)
        return total_loss
        

def set_up():

    logger.info("Setup...")

    if torch.cuda.is_available():
        device = "cuda"
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

def loaf_config(file_path = "config.yaml"):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = loaf_config("benchmarking/config.yaml")
    
    times = {}
    device, vocab_size, tokenizer, model, loss_functions = set_up()

    times["standalone"] = standalone_benchmark(
        config = config['standalone benchmark'],
        loss_functions = loss_functions,
        vocab_size = vocab_size,
        device = device
    )

    #gc.collect()
    time.sleep(1)

    times["forward pass"] = forward_pass_benchmark(
        config = config['forwad pass benchmark'],
        loss_functions = loss_functions,
        tokenizer = tokenizer,
        model = model,
        device = device
    )

    #gc.collect()
    time.sleep(1)

    times["training step"] = training_step_benchmark(
        config = config['training step benchmark'],
        loss_functions = loss_functions,
        tokenizer = tokenizer,
        model = model,
        device = device
    )

    timestamp = int(time.time() * 100)

    # Write to csv 
    with open(f'benchmarking/benchmark_results_{timestamp}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["loss"] + list(times.keys()))
        for loss_name in loss_functions.keys():
            writer.writerow([loss_name] + [times[key][loss_name] for key in times.keys()])

    shutil.copy("benchmarking/config.yaml", f"benchmarking/config_{timestamp}.stored_yaml")

if __name__ == "__main__":
    main()

# TODO: add order numbers?
# TODO: add realistic inputs from dataset
# TODO: Log to csv
# TODO: Check GPU Inputs
# TODO: Custom loss 
# TODO: plotting
# TODO: Save setting
# TODO: Every step new input
# TODO: unused variables in function signatures in tokenizer

# TODO: one plot for each benchmark
# TODO: add standart deviations
# TODO: add ce loss to the other losses