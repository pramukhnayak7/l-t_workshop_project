"""Small deterministic tools available to the assistant workflow."""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only basic arithmetic expressions are supported.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression without executing arbitrary code."""
    try:
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 40:
            return "Expression is too complex."
        result = _evaluate(tree.body)
        if abs(result) > 1e100:
            return "Result is outside the supported range."
        return str(int(result)) if result.is_integer() else f"{result:.8g}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return "I couldn't evaluate that expression. Try basic arithmetic, such as (18 * 4) / 3."