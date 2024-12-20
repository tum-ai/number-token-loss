from typing import Any, Callable

from ast_node import ASTNode
from expression_parser import ExpressionParser


class ExpressionSolver:
    """
    Solves arithmetic expressions step-by-step.
    """

    def __init__(
        self, 
        operators: dict[str, Callable[[int, int], int]] = None, 
        precedence: dict[str, int] = None,
    ):
        """
        Initializes the ExpressionSolver with a parser instance and operator definitions.

        Parameters:
            operators (Dict[str, Callable[[int, int], int]]): A dictionary mapping operators to functions.
                Defaults to standard arithmetic operations: 
                    {"+": addition, "-": subtraction, "*": multiplication, "/": integer division}.
            precedence: A dictionary mapping operators to their precedence. Defaults to {"+": 1, "-": 1, "*": 2, "/": 2}.
        """
        self._operators = operators or {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a // b,
        }
        self._precedence = precedence or {"+": 1, "-": 1, "*": 2, "/": 2}
        self._parser = ExpressionParser(
            operators=set(self._operators.keys()),
            precedence=self._precedence,
        )

    def solve(self, expression: str) -> dict[str, Any]:
        """
        Converts an arithmetic expression into a representation of the evaluation process.

        Parameters:
            expression: The arithmetic expressixon to process.

        Returns:
            A dict the evaluation process and result in the specified format.
        """
        solution, steps = self._convert_to_steps_and_result(expression)
        question = f"What is {expression}?"
        answer = " ".join(f"<<{step}>>" for step in steps) + f" {solution}"
        solution = {
            "question": question,
            "steps": [f"<<{step}>>" for step in steps],
            "solution": str(solution),
            "answer": answer
        }
        return solution

    def _convert_to_steps_and_result(self, expression: str) -> tuple[int, list[str]]:
        """
        Converts an arithmetic expression into evaluation steps and a final result.

        Parameters:
            expression: The arithmetic expression to process.

        Returns:
            A tuple containing:
                - The final result as an integer.
                - A list of steps representing the evaluation process.
        """
        ast = self._parser.parse(expression)
        result, steps = self._evaluate_ast_with_steps(ast)
        return result, steps

    def _evaluate_ast_with_steps(self, node: ASTNode) -> tuple[int, list[str]]:
        """
        Recursively evaluates the AST and generates the evaluation steps.

        Parameters:
            node: The current AST node to evaluate.

        Returns:
            A tuple containing:
                - The evaluated result of the current node.
                - A list of steps representing the evaluation process.
        """
        if not node.left and not node.right:
            return int(node.value), []

        left_value, left_steps = self._evaluate_ast_with_steps(node.left)
        right_value, right_steps = self._evaluate_ast_with_steps(node.right)

        if node.value in self._operators:
            result = self._operators[node.value](left_value, right_value)
        else:
            raise ValueError(f"Unsupported operator: {node.value}")

        left_str = f"({left_value})" if left_value < 0 else str(left_value)
        right_str = f"({right_value})" if right_value < 0 else str(right_value)
        current_step = f"{left_str}{node.value}{right_str}={result}"
        steps = left_steps + right_steps + [current_step]

        return result, steps
