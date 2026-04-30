from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pkgutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, MutableMapping, Sequence, Set, Tuple


DEFAULT_ROOT_MODULES: Tuple[str, ...] = (
    "pyquda",
    "pyquda_utils",
    "pyquda_io",
    "pyquda_plugins",
    "pyquda_comm",
)
DEFAULT_OUTPUT_FILENAME = "all_api.md"

# Latest PyQUDA repo is split across multiple sibling packages.  These probes
# complement pkgutil.walk_packages() for cases where public modules are exposed
# by compiled extensions, lazy imports, or re-exported package attributes.
PUBLIC_MODULE_HINTS: Dict[str, Tuple[str, ...]] = {
    "pyquda": (
        "__main__",
        "action",
        "dirac",
        "enum_quda",
        "field",
        "gamma",
        "hmc",
        "quda",
        "quda_define",
    ),
    "pyquda_utils": (
        "core",
        "source",
        "phase",
        "gamma",
        "io",
        "gpt",
        "gauge_nd_sun",
        "hmc_param",
        "milc_rhmc_param",
        "quasi_axial_gauge_fixing",
        "wilson_loop",
    ),
    "pyquda_io": (),
    "pyquda_plugins": (),
    "pyquda_comm": (),
}


@dataclass
class ImportIssue:
    module_name: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract public API docs from latest PyQUDA-style multi-package layouts."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional output markdown file path or output directory. Defaults to all_api.md beside this script.",
    )
    parser.add_argument(
        "--modules",
        default=",".join(DEFAULT_ROOT_MODULES),
        help="Comma-separated root modules to inspect. Default covers latest PyQUDA repo layout.",
    )
    return parser.parse_args()


def normalize_roots(raw_modules: str) -> List[str]:
    roots: List[str] = []
    for item in str(raw_modules).split(","):
        name = item.strip()
        if name and name not in roots:
            roots.append(name)
    if not roots:
        raise ValueError("No root modules provided.")
    return roots


def get_signature_or_fallback(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        text_sig = getattr(obj, "__text_signature__", None)
        if isinstance(text_sig, str) and text_sig.strip():
            return text_sig.strip()
        return "(...) [Signature hidden by Cython/C++]"


def ensure_output_path(output_path: str | None) -> str:
    if not output_path:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_OUTPUT_FILENAME)
    if os.path.isdir(output_path):
        return os.path.join(output_path, DEFAULT_OUTPUT_FILENAME)
    return output_path


def try_import(
    module_name: str,
    issues: MutableMapping[str, ImportIssue],
    import_cache: MutableMapping[str, ModuleType | None],
) -> ModuleType | None:
    if module_name in import_cache:
        return import_cache[module_name]
    try:
        module = importlib.import_module(module_name)
        import_cache[module_name] = module
        return module
    except Exception as exc:  # pragma: no cover - environment dependent
        import_cache[module_name] = None
        issues.setdefault(module_name, ImportIssue(module_name=module_name, error=repr(exc)))
        return None


def add_module(modules: MutableMapping[str, ModuleType], module: ModuleType | None) -> None:
    if module is None:
        return
    modules.setdefault(module.__name__, module)


def discover_modules(
    root_name: str,
    issues: MutableMapping[str, ImportIssue],
    import_cache: MutableMapping[str, ModuleType | None],
) -> "OrderedDict[str, ModuleType]":
    modules: "OrderedDict[str, ModuleType]" = OrderedDict()
    root_module = try_import(root_name, issues, import_cache)
    if root_module is None:
        return modules
    add_module(modules, root_module)

    root_all = getattr(root_module, "__all__", ())
    for attr_name in root_all:
        try:
            attr_obj = getattr(root_module, attr_name)
        except Exception:
            continue
        if inspect.ismodule(attr_obj):
            add_module(modules, attr_obj)

    for _, name, _ispkg in pkgutil.walk_packages(
        getattr(root_module, "__path__", []), root_module.__name__ + "."
    ):
        add_module(modules, try_import(name, issues, import_cache))

    for suffix in PUBLIC_MODULE_HINTS.get(root_name, ()):
        full_name = f"{root_name}.{suffix}" if suffix else root_name
        add_module(modules, try_import(full_name, issues, import_cache))

    for _name, obj in inspect.getmembers(root_module, inspect.ismodule):
        if obj.__name__ == root_name or obj.__name__.startswith(root_name + "."):
            add_module(modules, obj)

    return OrderedDict(sorted(modules.items()))


def is_public_name(name: str) -> bool:
    return not (name.startswith("_") and name != "__init__")


def should_keep_object(
    *,
    module_name: str,
    export_name: str,
    obj,
    module_exports: Set[str],
) -> bool:
    if not is_public_name(export_name):
        return False

    obj_module = getattr(obj, "__module__", None)
    if obj_module == module_name:
        return True

    if export_name in module_exports:
        return True

    return obj_module is None and export_name in module_exports


def get_origin_name(obj) -> str | None:
    obj_module = getattr(obj, "__module__", None)
    obj_name = getattr(obj, "__name__", None)
    if isinstance(obj_module, str) and obj_module and isinstance(obj_name, str) and obj_name:
        return f"{obj_module}.{obj_name}"
    return None


def iter_declared_methods(cls):
    for method_name, method_obj in cls.__dict__.items():
        if not is_public_name(method_name):
            continue
        if inspect.isroutine(method_obj):
            yield method_name, method_obj


def render_class_block(full_name: str, cls, origin_name: str | None = None) -> str:
    lines: List[str] = [f"### Class: {full_name}", ""]
    if origin_name and origin_name != full_name:
        lines.append(f"Origin: `{origin_name}`")
        lines.append("")
    methods_written = False
    for method_name, method_obj in iter_declared_methods(cls):
        signature = get_signature_or_fallback(method_obj)
        lines.append(f"    def {method_name}{signature}")
        lines.append("")
        methods_written = True
    if not methods_written:
        lines.append("    # No public methods discovered.")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_function_block(full_name: str, func, origin_name: str | None = None) -> str:
    signature = get_signature_or_fallback(func)
    lines = [f"### Function: {full_name}{signature}"]
    if origin_name and origin_name != full_name:
        lines.extend(["", f"Origin: `{origin_name}`"])
    return "\n".join(lines)


def collect_public_blocks(
    modules: MutableMapping[str, ModuleType]
) -> "OrderedDict[str, List[str]]":
    rendered: "OrderedDict[str, List[str]]" = OrderedDict()

    for module_name, module in modules.items():
        module_exports = set(getattr(module, "__all__", ()) or ())
        blocks: List[str] = []
        seen_names: Set[str] = set()

        for export_name, obj in inspect.getmembers(module):
            if export_name in seen_names:
                continue
            if not should_keep_object(
                module_name=module_name,
                export_name=export_name,
                obj=obj,
                module_exports=module_exports,
            ):
                continue

            full_name = f"{module_name}.{export_name}"
            origin_name = get_origin_name(obj)
            if inspect.isclass(obj):
                blocks.append(render_class_block(full_name, obj, origin_name))
                seen_names.add(export_name)
            elif inspect.isroutine(obj):
                blocks.append(render_function_block(full_name, obj, origin_name))
                seen_names.add(export_name)

        rendered[module_name] = blocks

    return rendered


def render_markdown(
    *,
    root_modules: Sequence[str],
    module_blocks: MutableMapping[str, List[str]],
    issues: Sequence[ImportIssue],
) -> str:
    lines: List[str] = [
        "# PyQUDA API 接口提取文档",
        "",
        f"> Roots: {', '.join(root_modules)}",
        "",
    ]

    for module_name, blocks in module_blocks.items():
        lines.append(f"## --- 模块: {module_name} ---")
        lines.append("")
        if not blocks:
            lines.append("_No public classes/functions discovered in this module._")
            lines.append("")
            continue
        for block in blocks:
            lines.append(block)
            lines.append("")
            lines.append("")

    if issues:
        lines.append("## --- 导入警告 ---")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.module_name}`: {issue.error}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def extract_api_info(root_modules: Sequence[str], output_path: str) -> None:
    issues: "OrderedDict[str, ImportIssue]" = OrderedDict()
    import_cache: Dict[str, ModuleType | None] = {}
    discovered: "OrderedDict[str, ModuleType]" = OrderedDict()

    for root_name in root_modules:
        for module_name, module in discover_modules(root_name, issues, import_cache).items():
            discovered.setdefault(module_name, module)

    if not discovered:
        print("致命错误: 未能导入任何目标模块。", file=sys.stderr)
        sys.exit(2)

    module_blocks = collect_public_blocks(discovered)
    output_filepath = ensure_output_path(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(
                render_markdown(
                    root_modules=root_modules,
                    module_blocks=module_blocks,
                    issues=list(issues.values()),
                )
            )
    print(f"成功: API 提取完成，保存至 {output_filepath}")
    if issues:
        print(f"提示: 共记录 {len(issues)} 条唯一模块导入警告。")


def main() -> None:
    args = parse_args()
    root_modules = normalize_roots(args.modules)
    extract_api_info(root_modules, args.output)


if __name__ == "__main__":
    main()
