"""`polanon` is a standalone library.

Two promises are made about it in the README and enforced here, because both are the kind that
decay quietly: it never reaches into a web framework or the hub built on it, and importing it
pulls in nothing but the standard
library. spaCy and httpx are optional extras, which only stays true while every import of them
happens inside a function.
"""

import ast
import importlib.util
import sys
from pathlib import Path

#: The package as Python imports it: the source tree in the main job, the installed wheel in the job
#: that moves the source aside. Reading the source path there found no files, and two of these
#: tests passed without checking anything.
LIBRARY = Path(importlib.util.find_spec("polanon").origin).parent


def test_the_library_is_where_these_tests_look():
    assert (LIBRARY / "__init__.py").is_file()
    assert len(list(LIBRARY.rglob("*.py"))) > 20


#: Packages the library must never import, at any depth: the hub itself and the frameworks it uses.
FORBIDDEN = ("piigate", "fastapi", "sqlmodel", "sqlalchemy", "starlette", "jinja2")

#: Modules allowed to import a third-party package at module level, and what they may import.
#: The judges that talk to a server need an HTTP client; nothing else does, and `polanon/__init__`
#: never imports them, so `import polanon` stays dependency-free.
ALLOWED_EXTERNAL = {
    ("judge/ollama.py", "httpx"),
    ("judge/remote.py", "httpx"),
}


def _module_level_imports(tree: ast.Module) -> set[str]:
    """Return the root package of every import that runs when the module is imported.

    A class body runs on import, so it counts. A function body does not, and that is exactly how the
    heavy optional pieces are loaded, so anything inside `def` is skipped however deeply it is
    nested.
    """
    roots: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if isinstance(child, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                roots.add(child.module.split(".")[0])
            visit(child)

    visit(tree)
    return roots


def test_the_library_never_imports_the_hub_or_a_framework():
    offenders = []
    for path in sorted(LIBRARY.rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            statement = line.strip()
            if statement.startswith(("import ", "from ")) and any(
                name in statement for name in FORBIDDEN
            ):
                offenders.append(f"{path.relative_to(LIBRARY)}: {statement}")
    assert not offenders, "\n".join(offenders)


def test_importing_the_library_pulls_in_nothing_but_the_standard_library():
    """`pip install --no-deps polanon` has to be enough to mask a document."""
    standard = set(sys.stdlib_module_names)
    offenders = []
    for path in sorted(LIBRARY.rglob("*.py")):
        relative = path.relative_to(LIBRARY).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for root in sorted(_module_level_imports(tree)):
            if root in standard or root == "polanon":
                continue
            if (relative, root) in ALLOWED_EXTERNAL:
                continue
            offenders.append(f"{relative} imports {root} at module level")
    assert not offenders, "\n".join(offenders)


def test_the_public_facade_does_not_reach_the_optional_judges():
    """The allowance above is only safe while `polanon/__init__` leaves those modules alone."""
    tree = ast.parse((LIBRARY / "__init__.py").read_text(encoding="utf-8"))
    # Module level only: the same file imports NerGuard inside default_guards, which is the point.
    imported = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "polanon.judge.ollama" not in imported
    assert "polanon.judge.remote" not in imported
    assert "polanon.guards.ner" not in imported


def test_the_heavy_guard_is_loaded_only_when_it_is_asked_for():
    """default_guards(with_ner=False) must not touch spaCy, so a bare install still works."""
    source = (LIBRARY / "__init__.py").read_text(encoding="utf-8")
    assert "from polanon.guards.ner import NerGuard" in source
    lazy = source.split("def default_guards")[1]
    assert "from polanon.guards.ner import NerGuard" in lazy
