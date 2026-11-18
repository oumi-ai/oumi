import ast
import sys
from pathlib import Path

import oumi_verl_ext

ALLOWED = set(sys.stdlib_module_names) | {
    "oumi_verl_ext",
    "verl",
    "transfer_queue",
    "numpy",
    "torch",
}


def test_oumi_verl_ext_imports_only_stdlib_and_verl_env_packages():
    package_dir = Path(oumi_verl_ext.__file__).parent
    for path in package_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, path
                assert node.module is not None, path
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert roots <= ALLOWED, f"{path.name} imports {roots - ALLOWED}"
