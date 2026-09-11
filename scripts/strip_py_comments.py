#!/usr/bin/env python3
"""Emit a comment- and docstring-free copy of a Python source file.

Final-phase Docker images should carry runnable code, not the development prose (design notes,
metric deltas, dataset provenance) that lives in the readable repo sources. This strips every
comment and every module/class/function docstring, then re-emits equivalent code via ast.unparse.

Safety: comments never reach the AST, so they cannot survive. Docstrings are removed as explicit
Expr(Constant str) leading statements (an emptied body gets a `pass`). The result self-checks —
the emitted code is re-parsed and its AST must equal the docstring-stripped input AST, which proves
only comments and docstrings were removed and no executable code was altered.

Usage: strip_py_comments.py SRC.py DST.py
"""
import ast
import sys


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())
    return tree


def strip_source(src):
    stripped_ast = _strip_docstrings(ast.parse(src))
    out = ast.unparse(stripped_ast)
    assert ast.dump(ast.parse(out)) == ast.dump(stripped_ast), \
        "strip altered executable code — refusing to emit"
    return out + "\n"


def main(src_path, dst_path):
    with open(src_path) as f:
        out = strip_source(f.read())
    with open(dst_path, "w") as f:
        f.write(out)
    print(f"[strip] {src_path} -> {dst_path} ({len(out)} bytes, comment/docstring-free)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
