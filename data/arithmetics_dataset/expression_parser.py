from typing import Optional

from ast_node import ASTNode


class ExpressionParser:
    """
    Parses an arithmetic expression and constructs an Abstract Syntax Tree (AST).
    """

    def __init__(
        self,
        operators: set[str] = None,
        precedence: dict[str, int] = None,
    ):
        """
        Initializes the parser with a list of operators and their precedence.

        Parameters:
            operators: A set of valid operators. Defaults to {"+", "-", "*", "/"}.
            precedence: A dictionary mapping operators to their precedence. Defaults to {"+": 1, "-": 1, "*": 2, "/": 2}.
        """
        self._operators = operators or {"+", "-", "*", "/"}
        self._precedence = precedence or {"+": 1, "-": 1, "*": 2, "/": 2}
        self._tokens: list[str] = []
        self._postfix: list[str] = []
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
        i = 0

        while i < len(expression):
            char = expression[i]
            if char.isdigit():
                current_number += char
            elif char == '-' and (i == 0 or expression[i - 1] == '('):
                current_number += char
            else:
                if current_number:
                    tokens.append(current_number)
                    current_number = ""
                if char in self._operators or char in {'(', ')'}:
                    tokens.append(char)
            i += 1

        if current_number:
            tokens.append(current_number)

        self._tokens = tokens

    def _convert_to_postfix(self) -> None:
        """
        Converts the tokenized expression into postfix notation using the Shunting Yard algorithm.

        Updates:
            self._postfix: Populates the list of tokens in postfix order.
        """
        output: list[str] = []
        operators: list[str] = []

        for token in self._tokens:
            if token.lstrip('-').isdigit():
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

    def _process_operator(self, operator: str, operators: list[str], output: list[str]) -> None:
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

    def _process_parenthesis(self, operators: list[str], output: list[str]) -> None:
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
            self._ast: sets the root node of the constructed AST.
        """
        stack: list[ASTNode] = []

        for token in self._postfix:
            if token.lstrip('-').isdigit():
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

    def _create_operator_node(self, operator: str, stack: list[ASTNode]) -> ASTNode:
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
