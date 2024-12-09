from __future__ import annotations
from typing import Optional


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
