from __future__ import annotations

import ast
from pathlib import Path


def _iter_runtime_import_offenders(source_root: Path):
    for file_path in sorted(source_root.glob("**/*.py")):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError as exc:
            raise AssertionError(f"Failed to parse {file_path}: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src.scripts"):
                        yield file_path, node.lineno, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.scripts"):
                yield file_path, node.lineno, node.module


def test_runtime_code_does_not_import_legacy_scripts():
    source_root = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        f"{file_path}:{line_number} -> {import_name}"
        for file_path, line_number, import_name in _iter_runtime_import_offenders(source_root)
    ]

    assert not offenders, "Legacy runtime imports were found:\n" + "\n".join(offenders)