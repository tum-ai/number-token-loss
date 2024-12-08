import random


def generate_expression(
    max_digits: int,
    num_operators: int,
    operators: list[str]=None,
    negative_probability: float=0.5,
    parentheses_probability: float=1.0,
) -> str:
    """
    Generates a random arithmetic expression with a specified number of operators.

    Parameters:
        max_digits: The maximum number of digits for each number in the expression.
        num_operators: The number of operators to include in the expression.
        operators: A list of operators to use in the expression.
        negative_probability: The probability that each number in the expression will be negative.
        parentheses_probability: The probability that each part of the expression will be put in parentheses.

    Returns:
        A randomly generated arithmetic expression.
    """
    if operators is None:
        operators = ['+', '-', '*']

    number = random_number(max_digits, negative_probability)
    expression = format_number(number)

    for _ in range(num_operators):
        operator = random.choice(operators)
        number = random_number(max_digits, negative_probability)
        expression = add_operation(expression, operator, number)

    expression = add_parentheses(expression, parentheses_probability)

    return expression

def random_number(max_digits: int, negative_probability: float=0.5) -> int:
    """
    Generates a random integer with up to max_digits digits. The number can be negative based on the negative_probability.

    Parameters:
        max_digits: The maximum number of digits for the generated number.
        negative_probability: The probability that the number will be negative.

    Returns:
        A random integer with up to max_digits digits, potentially negative.
    """
    number = random.randint(0, 10**max_digits - 1)
    if random.random() < negative_probability:
        number = -number
    return number

def add_operation(expression: str, operator: int, number: str) -> str:
    """
    Adds an operation to the existing expression.

    Parameters:
        expression: The current arithmetic expression.
        number: The number to add to the expression.
        operator: The operator to use in the operation.

    Returns:
        The updated arithmetic expression.
    """
    number_str = format_number(number)
    new_expr = f"{expression} {operator} {number_str}"
    return new_expr

def format_number(number: int) -> str:
    """
    Formats a number for inclusion in an expression, adding parentheses if the number is negative.

    Args:
        number: The number to format.

    Returns:
        The formatted number as a string.
    """
    return f"({number})" if number < 0 else str(number)

def add_parentheses(expression: str, parentheses_probability: float=0.5, wrap_full_expression: bool=False) -> str:
    """
    Adds parentheses to a random part of the expression based on parentheses_probability.

    Parameters:
        expression: The arithmetic expression to modify.
        parentheses_probability: The probability of adding parentheses.
        wrap_full_expression: A flag indicating whether the entire expression can be enclosed in parentheses.

    Returns:
        The modified expression with parentheses.
    """
    tokens = expression.split()

    min_num_tokens = 1 if wrap_full_expression else 3
    if len(tokens) <= min_num_tokens:
        return expression

    if random.random() > parentheses_probability:
        return expression

    num_operators = random.randint(1, len(tokens) // 2 - min_num_tokens // 2)
    start = 2 * random.randint(0, len(tokens) // 2 - num_operators)
    end = start + 2 * num_operators + 1

    left_expression = ' '.join(tokens[:start-1])
    center_expression = ' '.join(tokens[start:end])
    right_expression = ' '.join(tokens[end+1:])

    left_expression = f"{add_parentheses(left_expression, parentheses_probability, True)} {tokens[start-1]} " if start > 0 else ""
    center_expression = f"({add_parentheses(center_expression, parentheses_probability)})"
    right_expression = f" {tokens[end]} {add_parentheses(right_expression, parentheses_probability, True)}" if end < len(tokens) else ""

    expression = f"{left_expression}{center_expression}{right_expression}"

    return expression
