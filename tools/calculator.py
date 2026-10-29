"""Small numeric expression evaluator: no eval, imports, or object access."""
import ast
import math
import operator

OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
FUNCTIONS = {name: getattr(math, name) for name in (
    "sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil", "fabs"
)}
FUNCTIONS.update({"abs": abs, "round": round})
CONSTANTS = {"pi": math.pi, "e": math.e}


def calculator(expression: str) -> dict:
    if not isinstance(expression, str) or len(expression) > 1000:
        raise ValueError("expression must be a string of at most 1000 characters")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 150:
        raise ValueError("Expression is too complex")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.Name) and node.id in CONSTANTS:
            value = CONSTANTS[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent magnitude must not exceed 100")
            value = OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            if node.keywords or len(node.args) > 2:
                raise ValueError("Only up to two positional arguments are supported")
            value = FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args))
        else:
            raise ValueError("Only numeric arithmetic and approved math functions are allowed")
        if type(value) not in (int, float) or abs(value) > 1e100 or not math.isfinite(value):
            raise ValueError("Result must be finite with magnitude <= 1e100")
        return value

    return {"expression": expression, "result": evaluate(tree.body)}
