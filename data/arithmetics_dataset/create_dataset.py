from __future__ import annotations
from typing import Optional, List, Dict, Set
import random


class ExpressionGenerator:
    """
    Generates random arithmetic expressions with specified properties.
    """

    def __init__(
        self,
        operators: List[str]=None,
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


class ASTNode:
    """
    Represents a node in the Abstract Syntax Tree (AST).

    Attributes:
        value: The value of the node (operator or operand).
        left: The left child node, or None if no child exists.
        right: The right child node, or None if no child exists.
    """

    def __init__(self, value: str, left: Optional[ASTNode]=None, right: Optional[ASTNode]=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """
        Returns a string representation of the ASTNode.
        If the node has children, it displays the node as an operation.
        """
        if self.left or self.right:
            return f"({self.left} {self.value} {self.right})"
        return str(self.value)


class ExpressionParser:
    """
    Parses an arithmetic expression and constructs an Abstract Syntax Tree (AST).
    """

    def __init__(
        self,
        operators: Set[str]=None,
        precedence: Dict[str, int]=None,
    ):
        """
        Initializes the parser with a list of operators and their precedence.

        Parameters:
            operators: A list of valid operators. Defaults to ["+", "-", "*", "/"].
            precedence: A dictionary mapping operators to their precedence. Defaults to {"+" : 1, "-" : 1, "*" : 2, "/" : 2}.
        """
        self._operators = operators or {"+", "-", "*", "/"}
        self._precedence = precedence or {"+": 1, "-": 1, "*": 2, "/": 2}
        self._tokens: List[str] = []
        self._postfix: List[str] = []
        self._ast: Optional[ASTNode] = None

    def parse(self, expression: str) -> ASTNode:
        """
        Parses the input expression and constructs the AST.

        Parameters:
            expression: The arithmetic expression to parse.

        Returns:
            The root node of the generated AST.
        """
        self._tokenize(expression)
        self._convert_to_postfix()
        self._build_ast()
        return self._ast

    def _tokenize(self, expression: str) -> None:
        """
        Tokenizes the input expression into a list of numbers, operators, and parentheses.

        Parameters:
            expression: The arithmetic expression to tokenize.

        Updates:
            self._tokens: Populates the list of tokens derived from the input expression.
        """
        tokens = []
        current_number = ""

        for char in expression:
            if char.isdigit():
                current_number += char
            else:
                if current_number:
                    tokens.append(current_number)
                    current_number = ""
                if char in self._operators or char in {'(', ')'}:
                    tokens.append(char)

        if current_number:
            tokens.append(current_number)

        self._tokens = tokens

    def _convert_to_postfix(self) -> None:
        """
        Converts the tokenized expression into postfix notation using the Shunting Yard algorithm.

        Updates:
            self._postfix: Populates the list of tokens in postfix order.
        """
        output: List[str] = []
        operators: List[str] = []

        for token in self._tokens:
            if token.isdigit():
                output.append(token)
            elif token in self._operators:
                self._process_operator(token, operators, output)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                self._process_parenthesis(operators, output)

        while operators:
            output.append(operators.pop())

        self._postfix = output

    def _process_operator(self, operator: str, operators: List[str], output: List[str]) -> None:
        """
        Processes an operator token, handling precedence and stacking.

        Parameters:
            operator: The operator token to process.
            operators: The current operator stack.
            output: The current output list in postfix order.
        """
        while operators and self._precedence.get(operators[-1], 0) >= self._precedence.get(operator, 0):
            output.append(operators.pop())
        operators.append(operator)

    def _process_parenthesis(self, operators: List[str], output: List[str]) -> None:
        """
        Processes a closing parenthesis, popping operators to the output until an opening parenthesis is found.

        Parameters:
            operators: The current operator stack.
            output: The current output list in postfix order.
        """
        while operators and operators[-1] != '(':
            output.append(operators.pop())
        operators.pop()

    def _build_ast(self) -> None:
        """
        Builds the Abstract Syntax Tree (AST) from the postfix notation.

        Updates:
            self._ast: Sets the root node of the constructed AST.
        """
        stack: List[ASTNode] = []

        for token in self._postfix:
            if token.isdigit():
                stack.append(self._create_operand_node(token))
            elif token in self._operators:
                stack.append(self._create_operator_node(token, stack))

        self._ast = stack[0]

    def _create_operand_node(self, token: str) -> ASTNode:
        """
        Creates an AST node for a numeric operand.

        Parameters:
            token: The numeric token.

        Returns:
            An ASTNode representing the operand.
        """
        return ASTNode(token)

    def _create_operator_node(self, operator: str, stack: List[ASTNode]) -> ASTNode:
        """
        Creates an AST node for an operator with its children.

        Parameters:
            operator: The operator token.
            stack: The stack of AST nodes from which to retrieve the operator's children.

        Returns:
            An ASTNode representing the operator and its operands.
        """
        right_child = stack.pop()
        left_child = stack.pop()
        return ASTNode(operator, left_child, right_child)
