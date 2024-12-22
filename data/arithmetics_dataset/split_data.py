import argparse
import json
import random
from pathlib import Path


def split_jsonl(
    input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
):
    """
    Splits a JSONL file into train, validation, and test sets and saves them.

    Args:
        input_file (str): Path to the input JSONL file.
        output_dir (str): Directory to save the output files.
        train_ratio (float): Ratio for the training set.
        val_ratio (float): Ratio for the validation set.
        test_ratio (float): Ratio for the test set.
        seed (int): Random seed for reproducibility.
    """
    # Validate ratios
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    # Read the JSONL file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Shuffle the data
    random.seed(seed)
    random.shuffle(lines)

    # Split the data
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the splits
    for split_name, split_lines in zip(
        ["train", "val", "test"], [train_lines, val_lines, test_lines]
    ):
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            f.writelines(split_lines)
        print(f"Saved {split_name} set with {len(split_lines)} lines to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into train, val, and test sets."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Directory to save the output files."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train set ratio (default: 0.8)."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )

    args = parser.parse_args()
    split_jsonl(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
