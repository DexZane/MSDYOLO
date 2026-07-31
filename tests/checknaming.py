"""MSDYOLO 自有代码和交付物的连续小写命名守卫。"""

import ast
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHONFILES = [
    ROOT / "trainmsd.py",
    *sorted((ROOT / "utils").glob("*.py")),
    *sorted((ROOT / "tests").glob("check*.py")),
]
OWNEDUTILS = {
    "clearbranch.py",
    "config.py",
    "decoder.py",
    "degradation.py",
    "distillation.py",
    "matching.py",
    "profiler.py",
    "rotatednms.py",
    "routing.py",
    "sparse.py",
    "trainer.py",
}
FORCEDNAMES = {"tmp_path"}
FORCEDATTRIBUTES = {"register_buffer"}
ALLOWEDDIRECTORIES = {"labelTxt"}


def validcontinuous(name):
    """判断名称是否连续小写或 Python 双下划线。"""
    return (
        name.startswith("__")
        and name.endswith("__")
        or name.islower()
        and "_" not in name
    )


def validconstant(name):
    """判断模块常量是否为连续大写。"""
    return name.isupper() and "_" not in name


def modulelevel(node, parents):
    """判断赋值是否位于模块作用域，包括模块级 try/except。"""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        current = parents.get(current)
    return True


def yamlkeys(value):
    """递归枚举 YAML 字典键。"""
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from yamlkeys(child)
    elif isinstance(value, list):
        for child in value:
            yield from yamlkeys(child)


def pythonblocks(document):
    """提取 Markdown 中标记为 Python 的代码示例。"""
    return re.findall(r"```python\n(.*?)```", document, flags=re.DOTALL)


class CheckNaming:

    def checkownedpythonfilenames(self):
        assert {path.name for path in (ROOT / "utils").glob("*.py") if path.name in OWNEDUTILS} == OWNEDUTILS
        assert all("_" not in path.stem for path in (ROOT / "tests").glob("check*.py"))

    def checkdefinitionsargumentsvariablesandselfattributes(self):
        violations = []
        for path in PYTHONFILES:
            if path.parent.name == "utils" and path.name not in OWNEDUTILS:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            moduleconstants = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and validconstant(node.id)
                and modulelevel(node, parents)
            }
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not validcontinuous(node.name):
                        violations.append((path.name, node.lineno, node.name))
                    arguments = [
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ]
                    for argument in arguments:
                        if argument.arg not in FORCEDNAMES and not validcontinuous(argument.arg):
                            violations.append((path.name, argument.lineno, argument.arg))
                elif isinstance(node, ast.ClassDef):
                    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", node.name):
                        violations.append((path.name, node.lineno, node.name))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if (
                        node.id not in FORCEDNAMES
                        and node.id not in moduleconstants
                        and not validcontinuous(node.id)
                    ):
                        violations.append((path.name, node.lineno, node.id))
                elif (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and node.attr not in FORCEDATTRIBUTES
                    and not validcontinuous(node.attr)
                ):
                    violations.append((path.name, node.lineno, f"self.{node.attr}"))
        assert violations == []

    def checkyamlkeysandablationvalues(self):
        violations = []
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            violations.extend((path.name, key) for key in yamlkeys(document) if "_" in key)
            assert document["ablationmode"] in {
                "baseline",
                "withdegradation",
                "withclearbranch",
                "full",
            }
        assert violations == []

    def checkownedpathscontainnounderscore(self):
        roots = [
            ROOT / "configs",
            ROOT / "tests",
            ROOT / "data" / "dota-test",
        ]
        violations = []
        for root in roots:
            for path in root.rglob("*"):
                relative = path.relative_to(ROOT)
                if "__pycache__" in path.parts:
                    continue
                if path.name in {"__init__.py", "__pycache__"}:
                    continue
                if path.name in ALLOWEDDIRECTORIES:
                    continue
                if "_" in path.name:
                    violations.append(str(relative))
        assert violations == []

    def checkcurrentdocumentpathsusehyphens(self):
        expected = {
            ROOT / "docs" / "cp4pre-techdef.md",
            ROOT / "docs" / "claude-review.md",
        }
        assert all(path.exists() for path in expected)
        deprecated = {
            ROOT / "docs" / "cp4pretechdef.md",
            ROOT / "docs" / "cp4pre_revisions.md",
            ROOT / "docs" / "cp4pre_techdef.md",
            ROOT / "docs" / "gpt12_submission.md",
            ROOT / "docs" / "p0a_completion_report.md",
        }
        assert not any(path.exists() for path in deprecated)

    def checkcurrenttechnicalexamplesfollowprojectnames(self):
        document = (ROOT / "docs" / "cp4pre-techdef.md").read_text(encoding="utf-8")
        violations = []
        for blockindex, block in enumerate(pythonblocks(document)):
            tree = ast.parse(block)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not validcontinuous(node.name):
                        violations.append((blockindex, node.name))
                    for argument in [
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ]:
                        if not validcontinuous(argument.arg):
                            violations.append((blockindex, argument.arg))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if not validcontinuous(node.id):
                        violations.append((blockindex, node.id))
        assert violations == []
