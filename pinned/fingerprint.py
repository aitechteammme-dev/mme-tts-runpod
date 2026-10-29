"""Version-independent fingerprints of upstream (Chatterbox) functions, computed WITHOUT importing the package.

Why not plain `ast.dump`: its output differs between Python versions (3.12 adds `type_params` to function nodes, ...), so a table computed on one interpreter would reject the same source on another. The canonical
form below walks the syntax tree and ignores positions, comments, formatting and the version-dependent fields (`type_params`, `type_comment`, `kind`). It does NOT ignore code: any change to the statements,
names, constants or operators of the function changes the fingerprint.

`legacy_ast_dump_fingerprint` reproduces the older `sha256(ast.dump(node))` form (the one the streaming adapter used) so the two tables can be cross-checked on one interpreter.
"""
from __future__ import annotations

import ast
import hashlib
import textwrap
from typing import Iterable

_IGNORED_FIELDS = {"type_params", "type_comment", "kind"}


def _canonical(node) -> str:
    if isinstance(node, ast.AST):
        parts = [type(node).__name__]
        for name in node._fields:
            if name in _IGNORED_FIELDS:
                continue
            parts.append(f"{name}={_canonical(getattr(node, name, None))}")
        return "(" + ",".join(parts) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_canonical(x) for x in node) + "]"
    return repr(node)


def canonical_fingerprint(node: ast.AST) -> str:
    return hashlib.sha256(_canonical(node).encode("utf-8")).hexdigest()


def legacy_ast_dump_fingerprint(node: ast.AST) -> str:
    """sha256 of `ast.dump` of the function node taken DIRECTLY from the parsed module (decorators are part of the node): the form of the streaming adapter's TABLE.
    (The adapter's own `ast_fingerprint()` dedents and re-parses each method, which changes indented multi-line docstrings: it disagrees with its table for T3.inference and
    S3Token2Mel.forward on the exact intended revision. Never hash a dedented re-parse.)"""
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def find_function(source: str, qualname: str) -> tuple[ast.AST, str]:
    """Locate `Class.method` or `function` at the top level of the module `source`. Returns (the node itself, decorators included, taken from the module tree; its source text)."""
    tree = ast.parse(source)
    parts = qualname.split(".")
    scope: Iterable = tree.body
    target = None
    for i, name in enumerate(parts):
        target = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name), None)
        if target is None:
            raise KeyError(qualname)
        scope = target.body if isinstance(target, ast.ClassDef) else []
    start = min([target.lineno] + [d.lineno for d in getattr(target, "decorator_list", [])])
    text = textwrap.dedent("\n".join(source.splitlines()[start - 1:target.end_lineno]))      # for messages only: never fingerprinted
    return target, text


def fingerprints_from_source(source: str, qualnames: Iterable[str]) -> dict[str, tuple[str, str]]:
    """{qualname: (canonical, legacy_ast_dump)} for each function found in `source`."""
    out = {}
    for q in qualnames:
        node, _ = find_function(source, q)
        out[q] = (canonical_fingerprint(node), legacy_ast_dump_fingerprint(node))
    return out
