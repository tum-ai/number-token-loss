from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Optional
import yaml

from expression_generator import ExpressionGenerator
from expression_solver import ExpressionSolver


@dataclass
class ExpressionGenerationConfig:
    """
    Configuration for generating arithmetic expressions and their evaluations.

    Attributes:
        num_samples: The number of expressions to generate.
        num_operators: The number of operators in each generated expression.
        max_digits: The maximum number of digits for each operand.
        operators: A dictionary mapping operator strings to their corresponding functions.
            Defaults to standard arithmetic operators: {"+", "-", "*", "/"}.
        precedence: A dictionary defining the precedence of operators.
            Defaults to {"+": 1, "-": 1, "*": 2, "/": 2}.
        negative_probability: The probability of generating a negative operand.
            Must be a value between 0 and 1. Defaults to 0.5.
        parentheses_probability: The probability of wrapping parts of the expression in parentheses. 
            Must be a value between 0 and 1. Defaults to 1.0.
    """

    num_samples: int
    num_operators: int
    max_digits: int
    operators: dict[str, Callable[[int, int], int]] = field(
        default_factory=lambda: {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
        }
    )
    precedence: dict[str, int] = field(
        default_factory=lambda: {"+": 1, "-": 1, "*": 2, "/": 2}
    )
    negative_probability: float = 0.5
    parentheses_probability: float = 1.0


def load_configs(config_path: Path) -> list[ExpressionGenerationConfig]:
    """
    Reads a YAML file and converts it into a list of ExpressionGenerationConfig objects.

    Parameters:
        config_path: The file path to the YAML configuration file.

    Returns:
        A list of ExpressionGenerationConfig objects.
    """
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    configs = []
    for entry in data:
        configs.append(
            ExpressionGenerationConfig(
                num_samples=entry["num_samples"],
                num_operators=entry["num_operators"],
                max_digits=entry["max_digits"],
                operators={
                    op: eval(func) for op, func in entry.get("operators", {}).items()
                } if "operators" in entry else {
                    "+": lambda a, b: a + b,
                    "-": lambda a, b: a - b,
                    "*": lambda a, b: a * b,
                },
                precedence=entry.get("precedence", {"+": 1, "-": 1, "*": 2, "/": 2}),
                negative_probability=entry.get("negative_probability", 0.5),
                parentheses_probability=entry.get("parentheses_probability", 1.0),
            )
        )

    return configs


def create_dataset(configs: list[ExpressionGenerationConfig], path: Path) -> None:
    """
    Creates a dataset of solved arithmetic expressions and writes it to a JSONL file.

    Parameters:
        configs: A list of configuration objects for generating samples.
        path: The file path where the dataset will be saved.
    """
    dataset = []

    for config in configs:
        dataset.extend(generate_samples(config))

    save_to_jsonl(dataset, path)


def generate_samples(config: ExpressionGenerationConfig) -> list[dict[str, Any]]:
    """
    Generates a list of solved arithmetic expression samples based on the provided configuration.

    Parameters:
        config: The configuration object specifying the generation rules.

    Returns:
        A list of dictionaries representing solved expressions and their details.
    """
    generator = ExpressionGenerator(
        operators=list(config.operators.keys()),
        negative_probability=config.negative_probability,
        parentheses_probability=config.parentheses_probability,
    )
    solver = ExpressionSolver(
        operators=config.operators,
    )

    samples = [
        solver.solve(
            generator.generate(
                num_operators=config.num_operators,
                max_digits=config.max_digits,
            )
        )
        for _ in range(config.num_samples)
    ]

    return samples


def save_to_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    """
    Saves a list of dictionaries to a JSONL file.

    Parameters:
        data: The data to save, where each dictionary is a JSON object.
        path: The file path where the JSONL file will be saved.
    """
    with path.open("w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def main(config_path: str, output_path: str) -> None:
    configs = load_configs(Path(config_path))
    create_dataset(
        configs=configs,
        path=Path(output_path),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python generate_dataset.py <config_path> <output_path>")
        sys.exit(1)

    config_path, output_path = sys.argv[1], sys.argv[2]
    main(config_path, output_path)
