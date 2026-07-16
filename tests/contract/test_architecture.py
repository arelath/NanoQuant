import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_dependency_directions() -> None:
    root = Path("src/nanoquant")
    rules = {
        "domain": ("nanoquant.application", "nanoquant.infrastructure", "nanoquant.cli"),
        "ports": ("nanoquant.infrastructure", "nanoquant.application", "nanoquant.cli"),
        "application": ("nanoquant.infrastructure", "nanoquant.cli"),
        "runtime": ("nanoquant.application", "nanoquant.infrastructure", "nanoquant.config", "nanoquant.domain"),
    }
    violations = []
    for package, forbidden in rules.items():
        for path in (root / package).rglob("*.py"):
            for imported in _imports(path):
                if imported.startswith(forbidden):
                    violations.append(f"{path}: {imported}")
    assert violations == []


def test_numbered_runfiles_are_thin() -> None:
    forbidden_imports = {"argparse", "nanoquant.infrastructure"}
    forbidden_calls = {"open", "print"}
    violations = []
    for path in Path("experiments").glob("[0-9][0-9][0-9][-_]*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imports(path)
        if any(name.startswith(tuple(forbidden_imports)) for name in imports):
            violations.append(f"{path}: forbidden import")
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls
            for node in ast.walk(tree)
        ):
            violations.append(f"{path}: local orchestration call")
    assert violations == []


def test_src_contains_no_experiment_definitions() -> None:
    assert not Path("src/nanoquant/recipes").exists()
    violations = []
    for path in Path("src/nanoquant").rglob("*.py"):
        for imported in _imports(path):
            if imported == "recipes" or imported.startswith("recipes."):
                violations.append(f"{path}: {imported}")
    assert violations == []


def test_recipes_package_has_one_import_spelling() -> None:
    assert not Path("experiments/__init__.py").exists()


def test_runtime_distribution_contains_only_the_deployment_packages() -> None:
    root = Path("packaging/runtime")
    configuration = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'packages = ["nanoquant", "nanoquant.runtime"]' in configuration
    assert '"nanoquant" = "src/nanoquant"' in configuration
    assert '"nanoquant.runtime" = "../../src/nanoquant/runtime"' in configuration
    assert {path.name for path in (root / "src" / "nanoquant").iterdir()} == {
        "__init__.py"
    }
