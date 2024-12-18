import random
from typing import Optional


class ExpressionGenerator:
    """
    Generates random arithmetic expressions with specified properties.
    """

    def __init__(
        self,
        operators: Optional[list[str]] = None,
        negative_probability: float = 0.5,
        parentheses_probability: float = 1.0,
    ):
        """
        Initializes the generator with operators, number digit limits, and probabilities.

        Parameters:
            operators: A list of operators to use in the expressions. Defaults to ["+", "-", "*"].
            negative_probability: The probability that a number is negative.
            parentheses_probability: The probability that parts of the expression are wrapped in parentheses.
        """
        self._operators = operators or ["+", "-", "*"]
        self._negative_probability = negative_probability
        self._parentheses_probability = parentheses_probability

    def generate(self, num_operators: int, max_digits: int) -> str:
        """
        Generates a random arithmetic expression.

        Parameters:
            num_operators: The number of operators to include in the expression.
            max_digits: The maximum number of digits for each number.

        Returns:
            A randomly generated arithmetic expression as a string.
        """
        number = self._generate_random_number(max_digits)
        expression = self._format_number(number)

        for _ in range(num_operators):
            operator = random.choice(self._operators)
            number = self._generate_random_number(max_digits)
            expression = self._add_operation(expression, operator, number)

        return self._add_parentheses(expression)

    def _generate_random_number(self, max_digits: int) -> int:
        """
        Generates a random integer with up to max_digits digits. The number may be negative.

        Parameters:
            max_digits: The maximum number of digits for the number.

        Returns:
            A random integer with up to max_digits digits, potentially negative.
        """
        number = random.randint(0, 10**max_digits - 1)
        if random.random() < self._negative_probability:
            number = -number
        return number

    def _add_operation(self, expression: str, operator: str, number: int) -> str:
        """
        Adds an operation to the existing expression.

        Parameters:
            expression: The current arithmetic expression.
            operator: The operator to use in the operation.
            number: The number to add to the expression.

        Returns:
            The updated arithmetic expression as a string.
        """
        formatted_number = self._format_number(number)
        return f"{expression} {operator} {formatted_number}"

    def _format_number(self, number: int) -> str:
        """
        Formats a number for inclusion in an expression, adding parentheses if the number is negative.

        Parameters:
            number: The number to format.

        Returns:
            The formatted number as a string.
        """
        return f"({number})" if number < 0 else str(number)

    def _add_parentheses(self, expression: str, wrap_full_expression: bool=False) -> str:
        """
        Adds parentheses to a random part of the expression based on parentheses_probability.

        Parameters:
            expression: The arithmetic expression to modify.
            wrap_full_expression: A flag indicating whether the entire expression can be enclosed in parentheses.

        Returns:
            The modified expression with parentheses as a string.
        """
        tokens = expression.split()

        min_num_tokens = 1 if wrap_full_expression else 3
        if len(tokens) <= min_num_tokens:
            return expression

        if random.random() > self._parentheses_probability:
            return expression

        num_operators = random.randint(1, len(tokens) // 2 - min_num_tokens // 2)
        start = 2 * random.randint(0, len(tokens) // 2 - num_operators)
        end = start + 2 * num_operators + 1

        left_expression = ' '.join(tokens[:start-1])
        center_expression = ' '.join(tokens[start:end])
        right_expression = ' '.join(tokens[end+1:])

        left_expression = f"{self._add_parentheses(left_expression, True)} {tokens[start-1]} " if start > 0 else ""
        center_expression = f"({self._add_parentheses(center_expression)})"
        right_expression = f" {tokens[end]} {self._add_parentheses(right_expression, True)}" if end < len(tokens) else ""

        return f"{left_expression}{center_expression}{right_expression}"
