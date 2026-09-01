"""Tests proving Rho's import closure has no rhoalpha imports.

Three strategies:
1. Verify loaded modules contain no rhoalpha cross-references.
2. AST-scan rho source files for module-level rhoalpha imports.
3. Runtime import with rhoalpha blocked at the import level.
"""

import ast
import importlib
import sys
from pathlib import Path


def _is_rho_module(module_name: str) -> bool:
    return module_name == "rho.policies.rho" or module_name.startswith("rho.policies.rho.")


def test_rho_static_import_closure_has_no_rhoalpha():
    """No rho/rho_common module should hold a reference to an rhoalpha module object."""
    rho_deps = {module_name for module_name in sys.modules if _is_rho_module(module_name)}
    rhoalpha_modules = set()
    for mod_name in list(sys.modules):
        if not mod_name.startswith("rho.policies.rhoalpha"):
            continue
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for dep_name in rho_deps:
            dep = sys.modules.get(dep_name)
            if dep is None:
                continue
            for attr in vars(dep).values():
                if attr is mod:
                    rhoalpha_modules.add(mod_name)

    assert not rhoalpha_modules, (
        f"rho.policies.rho transitively depends on rhoalpha modules: {sorted(rhoalpha_modules)}"
    )


def test_rho_source_files_have_no_rhoalpha_imports():
    """AST scan: no module-level ``from rho.policies.rhoalpha`` imports in rho/."""
    root = Path(__file__).resolve().parents[2] / "rho" / "policies" / "rho"

    violations = []
    for py_file in root.rglob("*.py"):
        source = py_file.read_text()
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "rho.policies.rhoalpha" in node.module:
                violations.append(
                    f"{py_file.relative_to(root.parent)}:{node.lineno}: from {node.module} import ..."
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "rho.policies.rhoalpha" in alias.name:
                        violations.append(
                            f"{py_file.relative_to(root.parent)}:{node.lineno}: import {alias.name}"
                        )

    assert not violations, "Module-level rhoalpha imports found in rho source:\n" + "\n".join(violations)


def test_rho_imports_with_rhoalpha_unavailable():
    """Runtime import with rhoalpha blocked at the import level."""
    saved = {}
    for module_name in list(sys.modules):
        if _is_rho_module(module_name):
            saved[module_name] = sys.modules.pop(module_name)

    from rho.policies.base import PolicyConfig

    registry = getattr(PolicyConfig, "_choice_registry", {})
    saved_reg = {}
    if "rho" in registry:
        saved_reg["rho"] = registry.pop("rho")

    class _BlockRhoAlpha:
        def find_module(self, fullname, path=None):
            if fullname.startswith("rho.policies.rhoalpha"):
                return self
            return None

        def load_module(self, fullname):
            raise ImportError(f"Blocked by test: {fullname}")

    blocker = _BlockRhoAlpha()
    sys.meta_path.insert(0, blocker)
    try:
        mod = importlib.import_module("rho.policies.rho")
        assert hasattr(mod, "RhoPolicy")
        assert hasattr(mod, "RhoConfig")
    finally:
        sys.meta_path.remove(blocker)
        for module_name in list(sys.modules):
            if _is_rho_module(module_name):
                del sys.modules[module_name]
        sys.modules.update(saved)
        registry.update(saved_reg)
