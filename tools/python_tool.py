"""Restricted scientific Python runner, with a separate process and timeout.

This reduces accidental/LLM-generated unsafe operations. It is NOT an OS sandbox
for hostile multi-tenant workloads; see README for deployment boundaries.
"""
import ast
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MAX_CODE_CHARS = 12000
MAX_OUTPUT_CHARS = 12000
ALLOWED_IMPORTS = {
    "numpy", "pandas", "scipy", "scipy.stats", "sklearn", "sklearn.metrics",
    "sklearn.linear_model", "math", "statistics",
}
ALLOWED_ATTRIBUTES = set("""
DataFrame Series Index array asarray arange linspace zeros ones full eye
mean std median var sum min max abs round sqrt log log10 exp sin cos tan
pi e nan inf isnan isfinite where clip percentile quantile concatenate stack
vstack hstack column_stack reshape ravel flatten transpose T shape size ndim
dtype astype to_numpy tolist item values index columns loc iloc at iat
head tail describe count value_counts corr cov dropna fillna isna notna
sort_values sort_index reset_index set_index rename drop select_dtypes
groupby agg aggregate rolling expanding diff pct_change cumsum cumprod
unique nunique duplicated rank sample copy get items keys
concat merge crosstab pivot_table cut qcut to_numeric to_datetime
stats pearsonr spearmanr linregress ttest_ind ttest_rel ttest_1samp
shapiro normaltest mannwhitneyu wilcoxon f_oneway chi2_contingency
norm t cdf ppf pdf statistic pvalue correlation slope intercept rvalue
stderr intercept_stderr confidence_interval low high
metrics r2_score mean_squared_error mean_absolute_error
linear_model LinearRegression fit predict score coef_ intercept_
random default_rng normal uniform integers choice seed
floor ceil fabs factorial comb perm degrees radians stdev pstdev variance
pvariance fmean multimode mode median_low median_high
""".split())
SAFE_BUILTINS = {
    name: getattr(builtins, name) for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
        "list", "max", "min", "print", "range", "round", "set", "sorted", "str",
        "sum", "tuple", "zip",
    )
}
ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult,
    ast.UAdd, ast.USub, ast.Not, ast.Invert, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Call, ast.keyword, ast.Attribute,
    ast.Import, ast.ImportFrom, ast.alias, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.If, ast.For,
    ast.Break, ast.Continue, ast.Pass,
)
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "input", "help", "type", "object",
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "builtins",
    "memoryview", "breakpoint", "exit", "quit",
}


def validate_code(code: str) -> ast.Module:
    if not isinstance(code, str) or not code.strip() or len(code) > MAX_CODE_CHARS:
        raise ValueError("code must contain 1–12000 characters")
    tree = ast.parse(code, mode="exec")
    nodes = list(ast.walk(tree))
    if len(nodes) > 2500:
        raise ValueError("Code is too complex")
    for node in nodes:
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr not in ALLOWED_ATTRIBUTES or isinstance(node.ctx, ast.Store):
                raise ValueError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, ast.keyword) and (node.arg is None or node.arg.startswith("_")):
            raise ValueError("Expanded/private keyword arguments are forbidden")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import is forbidden: {alias.name}")
                if alias.asname and (alias.asname.startswith("_") or alias.asname in FORBIDDEN_NAMES):
                    raise ValueError("Forbidden import alias")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise ValueError("Import source is forbidden")
            for alias in node.names:
                if alias.name not in ALLOWED_ATTRIBUTES:
                    raise ValueError(f"Imported member is forbidden: {alias.name}")
                if alias.asname and (alias.asname.startswith("_") or alias.asname in FORBIDDEN_NAMES):
                    raise ValueError("Forbidden import alias")
    return tree


class BoundedOutput(io.StringIO):
    def write(self, text: str) -> int:
        if self.tell() + len(text) > MAX_OUTPUT_CHARS:
            raise ValueError("Python output exceeds 12000 characters; print a smaller summary")
        return super().write(text)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError("Import is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _run_worker(payload: dict) -> dict:
    import math
    import statistics
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import stats
    import sklearn
    from sklearn import metrics, linear_model

    tree = validate_code(payload["code"])
    namespace = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import},
        "np": np, "numpy": np, "pd": pd, "pandas": pd, "scipy": scipy,
        "stats": stats, "sklearn": sklearn, "metrics": metrics,
        "linear_model": linear_model, "math": math, "statistics": statistics,
    }
    if payload.get("data_file"):
        # The parent validates and snapshots the file into this temporary directory.
        namespace["df"] = pd.read_csv(payload["data_file"])
    output = BoundedOutput()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        exec(compile(tree, "<research-analysis>", "exec"), namespace, namespace)
    return {"stdout": output.getvalue(),
            "message": "" if output.tell() else "执行成功。请使用 print(...) 输出计算结果。"}


def python_calculate(code: str, data_dir: Path, data_file: str = "", timeout: int = 30) -> dict:
    from tools.file_tool import resolve_data_file

    validate_code(code)
    source = resolve_data_file(data_dir, data_file, {".csv"}) if data_file else None
    # Do not inherit API keys, proxy credentials, or unrelated environment secrets.
    worker_env = {name: os.environ[name] for name in (
        "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL",
    ) if name in os.environ}
    worker_env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"})
    with tempfile.TemporaryDirectory(prefix="research-analysis-") as temporary:
        snapshot = ""
        if source:
            target = Path(temporary) / "input.csv"
            target.write_bytes(source.read_bytes())
            snapshot = str(target)
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=json.dumps({"code": code, "data_file": snapshot}),
                capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                cwd=temporary, env=worker_env, shell=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Python 执行超时（{timeout} 秒），已终止进程。"}
        if process.returncode != 0:
            return {"error": "Python 工作进程异常退出，请检查依赖或减少计算规模。"}
        try:
            result = json.loads(process.stdout)
        except ValueError:
            return {"error": "Python 工作进程返回了无效结果。"}
        if source:
            result["source"] = data_file
        return result


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    try:
        # CPU / output-file bounds on Unix; Windows uses the parent wall timeout.
        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        answer = _run_worker(json.load(sys.stdin))
    except Exception as error:
        answer = {"error": f"{type(error).__name__}: {str(error)[:1500]}"}
    print(json.dumps(answer, ensure_ascii=True, allow_nan=False))
