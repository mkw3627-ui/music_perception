#!/usr/bin/env python3
"""Generate plugin.toml's fallback entries from @plugin_entry declarations."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# >>> STATIC ENTRIES (generated from @plugin_entry) >>>"


def _constant(node: ast.AST, constants: dict[str, object]) -> object:
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return ast.literal_eval(node)


def _signature_schema(node: ast.FunctionDef | ast.AsyncFunctionDef,
                     constants: dict[str, object], existing: dict) -> dict:
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults: dict[str, ast.AST] = {}
    if node.args.defaults:
        defaults.update(zip((arg.arg for arg in positional[-len(node.args.defaults):]),
                            node.args.defaults))
    defaults.update((arg.arg, value) for arg, value in zip(node.args.kwonlyargs,
                                                           node.args.kw_defaults)
                    if value is not None)
    arguments = [arg for arg in positional + list(node.args.kwonlyargs)
                 if arg.arg not in {"self", "cls"}]
    valid_names = {arg.arg for arg in arguments}

    prior = existing if isinstance(existing, dict) else {}
    prior_properties = prior.get("properties", {})
    if not isinstance(prior_properties, dict) or not set(prior_properties) <= valid_names:
        prior_properties = {}
    properties = dict(prior_properties)
    required = []
    for arg in arguments:
        if arg.arg not in properties:
            schema_type = None
            if isinstance(arg.annotation, ast.Name):
                schema_type = {"str": "string", "int": "integer", "float": "number",
                               "bool": "boolean", "dict": "object", "list": "array"}.get(
                                   arg.annotation.id)
            prop = {"type": schema_type} if schema_type else {}
            if arg.arg in defaults:
                try:
                    prop["default"] = _constant(defaults[arg.arg], constants)
                except (ValueError, TypeError):
                    pass
            properties[arg.arg] = prop
        if arg.arg not in defaults:
            required.append(arg.arg)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def entries_from_source(source: Path = ROOT, packaged_metadata: Path | None = None) -> list[dict]:
    tree = ast.parse((source / "__init__.py").read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and value is not None:
                    try:
                        constants[target.id] = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        pass

    old_entries: dict[str, dict] = {}
    try:
        manifest = tomllib.loads((source / "plugin.toml").read_text(encoding="utf-8"))
        old_entries.update({entry["id"]: entry for entry in
                            manifest.get("plugin", {}).get("entries", [])})
    except (OSError, ValueError, TypeError, KeyError):
        pass
    meta_path = packaged_metadata or source / "plugin.meta.json"
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            for entry in metadata.get("entries", []):
                old_entries.setdefault(entry["id"], entry)
        except (OSError, ValueError, TypeError, KeyError):
            old_entries = {}

    found: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "plugin_entry"):
                continue
            kwargs = {item.arg: item.value for item in decorator.keywords}
            try:
                entry_id = _constant(kwargs["id"], constants) if "id" in kwargs else node.name
                entry = {"id": entry_id}
                for field in ("name", "description", "input_schema"):
                    if field in kwargs:
                        entry[field] = _constant(kwargs[field], constants)
                if "name" not in entry:
                    entry["name"] = node.name
                if "description" not in entry:
                    entry["description"] = ""
                if "input_schema" not in entry:
                    entry["input_schema"] = _signature_schema(
                        node, constants,
                        old_entries.get(entry_id, {}).get("input_schema", {}))
                if not isinstance(entry_id, str) or not entry_id:
                    raise ValueError("plugin entry id must be a non-empty string")
            except (KeyError, ValueError, TypeError, SyntaxError) as exc:
                raise ValueError(f"cannot read plugin entry on {node.name}: {exc}") from exc
            if entry_id in found:
                raise ValueError(f"duplicate @plugin_entry id: {entry_id}")
            found[entry_id] = entry
    return [found[key] for key in sorted(found)]


def _toml_key(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value)


def _toml_value(value: object) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = ((key, item) for key, item in value.items() if item is not None)
        return "{ " + ", ".join(
            f"{_toml_key(key)} = {_toml_value(item)}" for key, item in fields
        ) + " }"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def render_entries(entries: list[dict]) -> str:
    lines = [MARKER, "# Generated by tools/sync_static_entries.py; do not edit by hand."]
    for entry in entries:
        lines.extend(("", "[[plugin.entries]]", f"id = {_toml_value(entry['id'])}"))
        for field in ("name", "description", "input_schema"):
            if entry.get(field):
                lines.append(f"{field} = {_toml_value(entry[field])}")
    return "\n".join(lines) + "\n"


def sync(source: Path = ROOT, *, write: bool = False) -> bool:
    manifest = source / "plugin.toml"
    current = manifest.read_text(encoding="utf-8")
    if MARKER in current:
        prefix = current.split(MARKER, 1)[0].rstrip()
    else:
        prefix = current.rstrip()
    updated = prefix + "\n\n" + render_entries(entries_from_source(source))
    # Parse the generated document before replacing the existing manifest.
    parsed = tomllib.loads(updated)
    if len(parsed.get("plugin", {}).get("entries", [])) != len(entries_from_source(
            source, packaged_metadata=source / "plugin.meta.json")):
        raise ValueError("generated static entries failed TOML validation")
    if write:
        manifest.write_text(updated, encoding="utf-8", newline="\n")
    return current == updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT,
                        help="plugin directory whose @plugin_entry declarations are used")
    parser.add_argument("--write", action="store_true", help="update plugin.toml")
    options = parser.parse_args()
    unchanged = sync(options.source, write=options.write)
    if options.write:
        print("Static plugin entries synchronized.")
        return 0
    if unchanged:
        print("Static plugin entries are current.")
        return 0
    print("Static plugin entries are stale; run tools/sync_static_entries.py --write.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
