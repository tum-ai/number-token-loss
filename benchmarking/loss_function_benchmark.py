"""
Benchmarking suite for comparing different loss functions in T5 training.
Measures performance across three scenarios:
1. Standalone loss computation
2. Forward pass through the model
3. Complete training step including gradient updates

The suite supports various loss functions including:
- Standard Cross Entropy (CE)
- CE with Number Token Loss (NTL) using MSE
- CE with NTL using Wasserstein distance
- CE with NTL using Absolute Difference
"""


import argparse
import time
import random
import string
import logging
import csv
import copy
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from torch.optim import AdamW
import yaml
import transformers

from ntl.tokenizer.t5custom_tokenizer import T5Custom_Tokenizer
from ntl.transformer_backbone.t5.t5_vanilla_for_number_token_loss import T5VanillaForNumberTokenLoss
from ntl.loss_functions.number_token_loss import NumberTokenLoss
from ntl.loss_functions.wasserstein_distance_number_token_loss import WassersteinNumberTokenLoss
from ntl.loss_functions.abs_diff_number_token_loss import AbsDiffNumberTokenLoss

# Constants
CUDA_DEVICE = "cuda"
RANDOM_SEED = 42
PAD_TOKEN_IGNORE_INDEX = -100
PAD_TO_MULTIPLE_OF = 64  # for model predictions

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CompositeLoss:
    """Combines multiple loss functions into a single loss function."""
    
    def __init__(self, loss_functions: list):
        self.loss_functions = loss_functions
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of all loss functions.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            
        Returns:
            Combined loss value
        """
        return sum(loss_fn.forward(logits, labels) for loss_fn in self.loss_functions)

class CrossEntropyLoss:
    """Standard cross entropy loss implementation."""
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross entropy loss.
        
        Args:
            logits: Shape (batch_size, seq_len, num_classes)
            labels: Shape (batch_size, seq_len)
            
        Returns:
            Loss value
        """
        batch_size, seq_len, num_classes = logits.shape
        assert labels.shape == (batch_size, seq_len), "Label shape mismatch"
        return torch.nn.functional.cross_entropy(
            logits.view(-1, num_classes),
            labels.view(-1)
        )

class BenchmarkTimer:
    """Utility for measuring and recording execution times."""
    
    def __init__(self):
        self.start_time = -1
        self.times: Dict[str, List[float]] = {}

    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self, checkpoint_name: str, device: str):
        """
        Stop the timer and record the elapsed time.
        
        Args:
            checkpoint_name: Name of the timing checkpoint
            device: Device being used (for proper CUDA synchronization)
        """
        if self.start_time == -1:
            raise RuntimeError("Timer not started")
        
        if checkpoint_name not in self.times:
            self.times[checkpoint_name] = []

        if device == CUDA_DEVICE:
            torch.cuda.synchronize()
        
        self.times[checkpoint_name].append(time.perf_counter() - self.start_time)
        self.start_time = time.perf_counter()

    def get_statistics(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate mean and standard deviation for each checkpoint.
        
        Returns:
            Tuple of (means, standard_deviations)
        """
        means = {point: np.mean(times) for point, times in self.times.items()}
        stds = {point: np.std(times) for point, times in self.times.items()}
        return means, stds

    def get_overall_statistics(self) -> Tuple[float, float]:
        """
        Calculate overall sum and standard deviation across all checkpoints.
        
        Returns:
            Tuple of (mean, standard_deviation)
        """
        all_means = [np.mean(segment) for segment in self.times.values()]
        all_means_std = [np.std(segment) for segment in self.times.values()]
        return np.sum(all_means), np.sqrt(np.sum(np.square(all_means_std)))

def generate_synthetic_data(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic input data for benchmarking.
    
    Args:
        batch_size: Number of samples in batch
        sequence_length: Length of each sequence
        vocab_size: Size of vocabulary
        device: Device to place tensors on
        
    Returns:
        Tuple of (logits, labels)
    """
    logits = torch.randn(
        batch_size,
        sequence_length,
        vocab_size,
        requires_grad=True,
        device=device
    )
    labels = torch.randint(
        vocab_size,
        (batch_size, sequence_length),
        device=device
    )
    return logits, labels

def generate_random_text(
    tokenizer: T5Custom_Tokenizer,
    n_tokens: int,
    number_share: float
) -> str:
    """
    Generate random text with a specified ratio of numbers to letters.
    
    Args:
        tokenizer: Tokenizer to check token count
        n_tokens: Desired number of tokens
        number_share: Proportion of characters that should be numbers
        
    Returns:
        Generated text
    """
    text = ""
    token_count = 0
    max_attempts = 10 * n_tokens

    while token_count < n_tokens and len(text) < max_attempts:
        text += (random.choice(string.digits) if random.random() <= number_share 
                else random.choice(string.ascii_letters + " "))
        token_count = len(tokenizer(text)['input_ids'])

    if token_count != n_tokens:
        raise ValueError(f"Failed to generate text with exact token count: {token_count} != {n_tokens}")
    
    return text

def generate_batch_texts(
    tokenizer: T5Custom_Tokenizer,
    batch_size: int,
    n_tokens: int,
    number_share: float,
    max_attempts: int = 10
) -> List[str]:
    """
    Generate a batch of random texts.
    
    Args:
        tokenizer: Tokenizer to check token counts
        batch_size: Number of texts to generate
        n_tokens: Desired number of tokens per text
        number_share: Proportion of characters that should be numbers
        max_attempts: Maximum attempts per text
        
    Returns:
        List of generated texts
    """
    texts = []
    
    for _ in range(batch_size):
        for attempt in range(max_attempts):
            try:
                texts.append(generate_random_text(tokenizer, n_tokens, number_share))
                break
            except ValueError:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid text after {max_attempts} attempts")
    
    return texts

def run_standalone_benchmark(
    config: Dict[str, Any],
    loss_fn: Any,
    loss_name: str,
    vocab_size: int,
    device: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Benchmark standalone loss computation.
    
    Args:
        config: Benchmark configuration
        loss_fn: Loss function to benchmark
        loss_name: Name of the loss function
        vocab_size: Size of vocabulary
        device: Device to run on
        
    Returns:
        Tuple of (execution_times, standard_deviations)
    """
    timer = BenchmarkTimer()

    for _ in range(config['steps']):
        logits, labels = generate_synthetic_data(
            config['batch_size'],
            config['sequence_length'],
            vocab_size,
            device
        )

        timer.start()
        loss = loss_fn.forward(logits, labels)
        timer.stop("complete_pass", device)
    
    times, stds = timer.get_statistics()
    logger.info(
        f"{loss_name} computation: {times['complete_pass']:.2f} ± "
        f"{stds['complete_pass']:.2f}s"
    )
    logger.debug(f"{loss_name} loss value: {loss.item()}")

    return times, stds

def run_model_benchmark(
    config: Dict[str, Any],
    model: T5VanillaForNumberTokenLoss,
    loss_fn: Any,
    loss_name: str,
    tokenizer: T5Custom_Tokenizer,
    device: str,
    update_gradients: bool = False
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Benchmark model forward pass and optionally gradient updates.
    
    Args:
        config: Benchmark configuration
        model: Model to benchmark
        loss_fn: Loss function to use
        loss_name: Name of the loss function
        tokenizer: Tokenizer for input processing
        device: Device to run on
        update_gradients: Whether to perform gradient updates
        
    Returns:
        Tuple of (execution_times, standard_deviations)
    """
    timer = BenchmarkTimer()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for _ in range(config['steps']):
        timer.start()

        # Generate input data
        input_texts = generate_batch_texts(
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            n_tokens=config['sequence_length'],
            number_share=config['number_share']
        )
        output_texts = generate_batch_texts(
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            n_tokens=config['sequence_length'],
            number_share=config['number_share']
        )
        timer.stop("data_generation", device)
        
        # Process inputs
        input_encoding = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        # Process targets
        target_encoding = tokenizer(
            output_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        labels = target_encoding["input_ids"].to(device)
        labels[labels == tokenizer.pad_token_id] = PAD_TOKEN_IGNORE_INDEX
        
        timer.stop("preprocessing", device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        timer.stop("forward_pass", device)            

        # Loss computation and backward pass
        loss = loss_fn.forward(outputs.logits, labels)
        loss.backward()
        timer.stop("backward_pass", device)

        # Gradient update if requested
        if update_gradients:
            optimizer.step()
            optimizer.zero_grad()
            timer.stop("optimizer_step", device)

    times, stds = timer.get_statistics()
    overall_time, overall_std = timer.get_overall_statistics()

    step_type = "Training" if update_gradients else "Forward pass"
    logger.info(
        f"{step_type} with {loss_name}: {overall_time:.2f} ± {overall_std:.2f}s"
    )
    logger.debug(f"{loss_name} loss value: {loss.item()}")

    return times, stds

def save_benchmark_results(filename: str, results: Dict[str, Any]):
    """
    Save benchmark results to CSV file.
    
    Args:
        filename: Output filename
        results: Benchmark results dictionary
    """
    rows = []
    measurement_points = set()

    # Prepare rows and collect all measurement points
    for benchmark, losses in results.items():
        for loss_name, (times, errors) in losses.items():
            row = {
                "benchmark": benchmark,
                "loss": loss_name
            }
            for point, value in times.items():
                row[point] = value
                row[f"{point}_err"] = errors[point]
                measurement_points.add(point)
            rows.append(row)

    # Write to CSV
    fieldnames = ["benchmark", "loss"]
    fieldnames.extend(sorted(
        point + suffix for point in measurement_points 
        for suffix in ["", "_err"]
    ))

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Results written to {filename}")

def initialize_benchmarking_environment():
    """
    Initialize all components needed for benchmarking.
    
    Returns:
        Tuple of (device, vocab_size, tokenizer_dict, model_dict, loss_functions)
    """
    logger.info("Initializing benchmarking environment...")

    # Set device
    device = CUDA_DEVICE if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize components for CE Loss
    ce_tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
    ce_model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Initialize components for other losses
    custom_tokenizer = T5Custom_Tokenizer.from_pretrained("t5-small")
    custom_model = T5VanillaForNumberTokenLoss.from_pretrained("t5-small")
    
    # Calculate vocabulary size for custom components
    vocab_size = len(custom_tokenizer) + PAD_TO_MULTIPLE_OF - (len(custom_tokenizer) % PAD_TO_MULTIPLE_OF)

    # Create dictionaries to store models and tokenizers
    tokenizer_dict = {
        "CE": ce_tokenizer,
        "custom": custom_tokenizer
    }
    
    model_dict = {
        "CE": ce_model,
        "custom": custom_model
    }

    # Initialize loss functions
    ce_loss = CrossEntropyLoss()
    mse_ntl = NumberTokenLoss(tokenizer=custom_tokenizer, vocab_size=vocab_size, device=device)
    wasserstein_ntl = WassersteinNumberTokenLoss(
        tokenizer=custom_tokenizer,
        vocab_size=vocab_size,
        order_numbers=False,
        device=device
    )
    abs_diff_ntl = AbsDiffNumberTokenLoss(
        tokenizer=custom_tokenizer,
        vocab_size=vocab_size,
        device=device
    )

    # Create combined loss functions
    loss_functions = {
        "CE": ce_loss,
        "CE+MSE": CompositeLoss([ce_loss, mse_ntl]),
        "CE+Wasserstein": CompositeLoss([ce_loss, wasserstein_ntl]),
        "CE+AbsDiff": CompositeLoss([ce_loss, abs_diff_ntl])
    }
    
    logger.info("Initialization complete")
    return device, vocab_size, tokenizer_dict, model_dict, loss_functions

def run_benchmarks(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all benchmarking scenarios with different loss functions.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary containing all benchmark results
    """
    results = {}
    device, vocab_size, tokenizer_dict, model_dict, loss_functions = initialize_benchmarking_environment()

    # Perform warmup runs with minimal steps
    logger.info("Performing warmup runs for standalone benchmarks...")
    warmup_config = copy.deepcopy(config)
    for benchmark_type in warmup_config.values():
        benchmark_type['steps'] = 1

    # Warmup standalone benchmarks
    for name, loss_fn in loss_functions.items():
        run_standalone_benchmark(
            config=warmup_config['standalone benchmark'],
            loss_fn=loss_fn,  
            loss_name=name,
            vocab_size=vocab_size,
            device=device
        )

    # Run standalone benchmarks
    logger.info("Running standalone loss computation benchmarks...")
    results["standalone"] = {
        name: run_standalone_benchmark(
            config=config['standalone benchmark'],
            loss_fn=loss_fn,
            loss_name=name,
            vocab_size=vocab_size,
            device=device
        )
        for name, loss_fn in loss_functions.items()
    }

    # Warmup forward pass benchmarks
    logger.info("Performing warmup runs for forward pass benchmarks...")
    for name, loss_fn in loss_functions.items():
        run_standalone_benchmark(
            config=warmup_config['forward pass benchmark'],
            loss_fn=loss_fn,  
            loss_name=name,
            vocab_size=vocab_size,
            device=device
        )

    # Run forward pass benchmarks
    logger.info("Running forward pass benchmarks...")
    results["forward_pass"] = {
        name: run_model_benchmark(
            config=config['forward pass benchmark'],
            model=model_dict["CE"] if name == "CE" else model_dict["custom"],
            loss_fn=loss_fn,
            loss_name=name,
            tokenizer=tokenizer_dict["CE"] if name == "CE" else tokenizer_dict["custom"],
            device=device,
            update_gradients=False
        )
        for name, loss_fn in loss_functions.items()
    }

    # Warmup training step benchmarks
    logger.info("Performing warmup runs for training step benchmarks...")
    for name, loss_fn in loss_functions.items():
        run_standalone_benchmark(
            config=warmup_config['training step benchmark'],
            loss_fn=loss_fn,  
            loss_name=name,
            vocab_size=vocab_size,
            device=device
        )

    # Run training step benchmarks
    logger.info("Running training step benchmarks...")
    results["training_step"] = {
        name: run_model_benchmark(
            config=config['training step benchmark'],
            model=model_dict["CE"] if name == "CE" else model_dict["custom"],
            loss_fn=loss_fn,
            loss_name=name,
            tokenizer=tokenizer_dict["CE"] if name == "CE" else tokenizer_dict["custom"],
            device=device,
            update_gradients=True
        )
        for name, loss_fn in loss_functions.items()
    }

    return results

# The rest of the code remains the same as it doesn't interact with model/tokenizer initialization
def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load benchmark configuration from YAML file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")

def run_number_share_analysis(
    base_config_path: str = "benchmarking/config.yaml",
    number_shares: List[float] = None
):
    """
    Run benchmarks across different proportions of numerical tokens.
    
    Args:
        base_config_path: Path to base configuration file
        number_shares: List of number share values to test (default: 0.0 to 1.0 in 0.1 steps)
    """
    if number_shares is None:
        number_shares = [round(0.1 * n, 1) for n in range(11)]

    base_config = load_config(base_config_path)
    
    for share in number_shares:
        logger.info(f"Running benchmarks with {share:.1%} number share...")
        config = copy.deepcopy(base_config)
        
        # Update number share in all benchmark configurations
        for benchmark_type in config.values():
            benchmark_type['number_share'] = share
        
        # Run benchmarks and save results
        results = run_benchmarks(config)
        timestamp = int(time.time() * 100)
        
        # Save results
        save_benchmark_results(
            f'benchmarking/benchmark_results_{timestamp}.csv',
            results
        )
        
        # Save configuration
        with open(f'benchmarking/config_{timestamp}.stored_yaml', 'w') as file:
            yaml.dump(config, file)

def main():
    """Main entry point for the benchmarking suite."""
    parser = argparse.ArgumentParser(description='Neural network loss function benchmarking suite')
    parser.add_argument(
        '--config',
        default="benchmarking/config.yaml",
        help='Path to configuration file'
    )
    parser.add_argument(
        '--number-share-analysis',
        action='store_true',
        help='Run analysis across different number share values'
    )
    args = parser.parse_args()

    if args.number_share_analysis:
        run_number_share_analysis(args.config)
    else:
        config = load_config(args.config)
        results = run_benchmarks(config)
        
        timestamp = int(time.time() * 100)
        save_benchmark_results(
            f'benchmarking/benchmark_results_{timestamp}.csv',
            results
        )
        with open(f'benchmarking/config_{timestamp}.stored_yaml', 'w') as file:
            yaml.dump(config, file)

if __name__ == "__main__":
    main()