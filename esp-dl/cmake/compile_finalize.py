#!/usr/bin/env python3
"""Generate compile-time files from spec YAML + compile_req.yml.

Writes dl_compile_config.h, dl_compile_srcs.cmake, dl_kernel.inc, and
dl_conv_select.inc. No req file means compile everything in ops.yml (DL_COMPILE_ALL).

Source lists come from ops.yml (`srcs` + `target` for the current chip).
Kernels in the union become DL_KERNEL_* macros; Conv functions are stripped
that way. Names not in kernels.yml are ignored.

kernel_abi in a req file must not exceed kernels.yml; older abi is allowed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_SELECT_TARGET = {
    "esp32p4": "pie_v2",
    "esp32s31": "pie_v2",
    "esp32s3": "pie_v1",
}


def select_target(idf_target: str) -> str:
    return _SELECT_TARGET.get(idf_target, "c")


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_flow_list(text: str) -> list[Any]:
    inner = text[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(part.strip()) for part in inner.split(",")]


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return _parse_flow_list(text)
    if text in ("{}",):
        return {}
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.lower() in ("null", "~", ""):
        return None
    try:
        if text.startswith("0") and text != "0":
            raise ValueError
        return int(text)
    except ValueError:
        pass
    return text


def parse_simple_yaml(text: str) -> Any:
    """Indent-based YAML subset: maps, lists, scalars. No anchors/tags/multiline."""
    raw_lines = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = _strip_comment(line)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if stripped.lstrip().startswith("- "):
            raw_lines.append((lineno, indent, "list", stripped.lstrip()[2:].strip()))
        else:
            raw_lines.append((lineno, indent, "key", stripped.lstrip()))

    def parse_block(idx: int, parent_indent: int) -> tuple[Any, int]:
        mapping: dict[str, Any] = {}
        sequence: list[Any] = []
        mode = None
        while idx < len(raw_lines):
            lineno, indent, kind, payload = raw_lines[idx]
            if indent < parent_indent:
                break
            if indent != parent_indent and mode is None:
                raise ValueError("invalid indent at line %d" % lineno)
            if indent > parent_indent:
                raise ValueError("unexpected indent at line %d" % lineno)

            if kind == "list":
                if mode == "map":
                    raise ValueError("mixed map/list at line %d" % lineno)
                mode = "list"
                if payload == "":
                    child, idx = parse_block(idx + 1, indent + 2)
                    sequence.append(child)
                elif ":" in payload and not payload.startswith("{"):
                    key, rest = payload.split(":", 1)
                    item: dict[str, Any] = {}
                    rest = rest.strip()
                    nxt_indent = (
                        raw_lines[idx + 1][1] if idx + 1 < len(raw_lines) else -1
                    )
                    if rest != "":
                        item[key.strip()] = _parse_scalar(rest)
                        idx += 1
                    elif idx + 1 < len(raw_lines) and nxt_indent > indent:
                        child, idx = parse_block(idx + 1, nxt_indent)
                        item[key.strip()] = child
                    else:
                        item[key.strip()] = {}
                        idx += 1
                    child_indent = indent + 2
                    if (
                        idx < len(raw_lines)
                        and raw_lines[idx][1] == child_indent
                        and raw_lines[idx][2] == "key"
                    ):
                        rest_map, idx = parse_block(idx, child_indent)
                        if not isinstance(rest_map, dict):
                            raise ValueError(
                                "expected map continuation at line %d" % lineno
                            )
                        item.update(rest_map)
                    sequence.append(item)
                else:
                    sequence.append(_parse_scalar(payload))
                    idx += 1
                continue

            if mode == "list":
                raise ValueError("mixed map/list at line %d" % lineno)
            mode = "map"
            if ":" not in payload:
                raise ValueError("expected key: at line %d" % lineno)
            key, rest = payload.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            nxt_indent = raw_lines[idx + 1][1] if idx + 1 < len(raw_lines) else -1
            if rest != "":
                mapping[key] = _parse_scalar(rest)
                idx += 1
            elif idx + 1 < len(raw_lines) and nxt_indent > indent:
                child, idx = parse_block(idx + 1, nxt_indent)
                mapping[key] = child
            else:
                mapping[key] = {}
                idx += 1
        if mode == "list":
            return sequence, idx
        return mapping, idx

    if not raw_lines:
        return {}
    doc, idx = parse_block(0, raw_lines[0][1])
    if idx != len(raw_lines):
        raise ValueError("unparsed YAML starting at line %d" % raw_lines[idx][0])
    return doc


def load_yaml(path: Path) -> Any:
    return parse_simple_yaml(path.read_text(encoding="utf-8"))


def target_files(target_map: Any, idf_target: str) -> list[str]:
    if not target_map:
        return []
    key = select_target(idf_target)
    files = target_map.get(key) or []
    if not isinstance(files, list):
        raise ValueError("target.%s must be a list" % key)
    return list(files)


def collect_spec_files(spec: dict, ops_doc: dict, idf_target: str) -> list[str]:
    files: list[str] = []
    groups = ops_doc.get("groups") or {}
    for name in spec.get("use") or []:
        if name not in groups:
            raise SystemExit("ops.yml group %r not found" % name)
        files.extend(collect_spec_files(groups[name], ops_doc, idf_target))
    files.extend(spec.get("srcs") or [])
    files.extend(target_files(spec.get("target"), idf_target))
    return files


def unique(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _req_kernel_abi(data: dict) -> int | None:
    ver = data.get("kernel_abi")
    if ver is None:
        return None
    return int(ver)


def union_reqs(req_paths: list[Path]) -> tuple[set[str], set[str], int | None]:
    ops: set[str] = set()
    kernels: set[str] = set()
    abi = None
    for path in req_paths:
        data = load_yaml(path)
        if not isinstance(data, dict):
            raise SystemExit("%s: expected a mapping" % path)
        ver = _req_kernel_abi(data)
        if ver is not None:
            if abi is None:
                abi = ver
            elif abi != ver:
                raise SystemExit("kernel_abi mismatch between compile_req files")
        for op in data.get("ops") or []:
            ops.add(str(op))
        for kernel in data.get("kernels") or []:
            kernels.add(str(kernel))
    return ops, kernels, abi


def op_canonical_map(ops_doc: dict) -> dict[str, str]:
    """op_type or alias -> canonical ops.yml key."""
    out: dict[str, str] = {}
    for name, spec in (ops_doc.get("ops") or {}).items():
        out[str(name)] = str(name)
        if not isinstance(spec, dict):
            continue
        for alias in spec.get("aliases") or []:
            out[str(alias)] = str(name)
    return out


def kernel_symbol_list(kernels_doc: dict) -> list[str]:
    raw = kernels_doc.get("symbols") or []
    if isinstance(raw, list):
        return [str(n) for n in raw]
    if isinstance(raw, dict):
        return [str(n) for n in raw]
    raise SystemExit("kernels.yml: symbols must be a list")


def kernel_name_set(kernels_doc: dict) -> set[str]:
    names = set(kernel_symbol_list(kernels_doc))
    aliases = kernels_doc.get("aliases") or {}
    if isinstance(aliases, dict):
        names.update(str(a) for a in aliases)
    return names


def validate_select(conv_yml: Path, ops_doc: dict, legal_kernels: set[str]) -> None:
    """Conv.yml: stem must be an ops.yml key; slot kernels must exist."""
    if not conv_yml.is_file():
        raise SystemExit("Conv.yml not found: %s" % conv_yml)
    alias_map = op_canonical_map(ops_doc)
    op = conv_yml.stem
    if op not in alias_map:
        raise SystemExit("%s: op %r is not in ops.yml" % (conv_yml, op))
    doc = load_yaml(conv_yml)
    if not isinstance(doc, dict):
        raise SystemExit("%s: expected a mapping" % conv_yml)
    slots = doc.get("slots") or {}
    if not isinstance(slots, dict) or not slots:
        raise SystemExit("%s: missing slots:<target>: [..]" % conv_yml)
    rules = doc.get("rules") or []
    if not isinstance(rules, list):
        raise SystemExit("%s: rules must be a list" % conv_yml)
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise SystemExit("%s: rule %d is not a mapping" % (conv_yml, i))
        target = rule.get("target")
        if target not in slots:
            raise SystemExit(
                "%s: rule %d target %r has no slots entry" % (conv_yml, i, target)
            )
        slot_names = slots[target]
        if not isinstance(slot_names, list) or not slot_names:
            raise SystemExit(
                "%s: slots.%s must be a non-empty list" % (conv_yml, target)
            )
        for slot in slot_names:
            name = conv_slot_name(rule, str(slot))
            if name is None:
                if str(slot) == "border" and str(rule.get("kshape")) == "11":
                    continue
                raise SystemExit("%s: rule %d missing slot %r" % (conv_yml, i, slot))
            if name not in legal_kernels:
                raise SystemExit(
                    "%s: rule %d %s=%s is not in kernels.yml"
                    % (conv_yml, i, slot, name)
                )


def write_compile_req(
    path: Path, kernel_abi: int, ops: list[str], kernels: list[str]
) -> None:
    lines = ["kernel_abi: %d" % int(kernel_abi), "ops:"]
    for op in ops:
        lines.append("  - %s" % op)
    lines.append("kernels:")
    for kernel in kernels:
        lines.append("  - %s" % kernel)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def macro_op(op: str) -> str:
    return "DL_OP_" + op


def macro_kernel(name: str) -> str:
    ident = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    return "DL_KERNEL_" + ident.upper()


def kernel_if(name: str) -> str:
    return "#if DL_COMPILE_ALL || %s" % macro_kernel(name)


def write_header(
    path: Path, compile_all: bool, ops: set[str], kernels: set[str]
) -> None:
    lines = [
        "/* Generated by compile_finalize.py. Do not edit. */",
        "#pragma once",
        "",
        "#define DL_COMPILE_ALL %d" % (1 if compile_all else 0),
        "",
    ]
    if not compile_all:
        for op in sorted(ops):
            lines.append("#define %s 1" % macro_op(op))
        if ops:
            lines.append("")
        for kernel in sorted(kernels):
            lines.append("#define %s 1" % macro_kernel(kernel))
        if kernels:
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srcs_cmake(path: Path, srcs: list[str]) -> None:
    lines = [
        "# Generated by compile_finalize.py. Do not edit.",
        "set(ESPDL_COMPILE_SRCS",
    ]
    for src in srcs:
        lines.append("    %s" % src.replace("\\", "/"))
    lines.append(")")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def module_specs(ops_doc: dict) -> list[tuple[str, str, str]]:
    """ops.yml ops carrying a `module:` spec -> (op, header, class), sorted."""
    out: list[tuple[str, str, str]] = []
    for name, spec in (ops_doc.get("ops") or {}).items():
        if not isinstance(spec, dict):
            continue
        mod = spec.get("module")
        if mod is None:
            continue
        if not isinstance(mod, dict) or not mod.get("header") or not mod.get("class"):
            raise SystemExit("ops.yml: op %r has an incomplete module spec" % name)
        out.append((str(name), str(mod["header"]), str(mod["class"])))
    return sorted(out)


def write_module_includes(path: Path, mods: list[tuple[str, str, str]]) -> None:
    """Gated module-header includes, consumed at file scope by dl_module_creator.cpp."""
    lines = ["/* Generated by compile_finalize.py. Do not edit. */"]
    for op, header, _cls in mods:
        lines.append("#if DL_COMPILE_ALL || defined(%s)" % macro_op(op))
        lines.append('#include "%s"' % header)
        lines.append("#endif")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_module_register(path: Path, mods: list[tuple[str, str, str]]) -> None:
    """Gated register_module() rows, consumed inside register_dl_modules()."""
    lines = ["/* Generated by compile_finalize.py. Do not edit. */"]
    for op, _header, cls in mods:
        lines.append("#if DL_COMPILE_ALL || defined(%s)" % macro_op(op))
        lines.append('register_module("%s", %s::deserialize);' % (op, cls))
        lines.append("#endif")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def canonicalize_ops(ops: set[str], alias_map: dict[str, str]) -> list[str]:
    selected = []
    unknown = []
    for op in sorted(ops):
        canon = alias_map.get(op)
        if canon is None:
            unknown.append(op)
        else:
            selected.append(canon)
    if unknown:
        raise SystemExit("op(s) not in ops.yml: %s" % ", ".join(unknown))
    return unique(selected)


def resolve_srcs(
    ops_doc: dict,
    target: str,
    ops: set[str] | None,
    alias_map: dict[str, str],
) -> list[str]:
    files: list[str] = []
    catalog_ops = ops_doc.get("ops") or {}
    if ops is None:
        selected = sorted(catalog_ops)
    else:
        selected = canonicalize_ops(ops, alias_map)
    for op in selected:
        spec = catalog_ops.get(op) or {}
        if not isinstance(spec, dict):
            spec = {}
        files.extend(collect_spec_files(spec, ops_doc, target))
    return unique(files)


_CONV_DTYPE = {"s8": 0, "s16": 1, "w8a16": 2}
_CONV_KSHAPE = {"11": 0, "33": 1, "hw": 2}
_CONV_ACT = {"linear": 0, "relu": 1, "leakyrelu": 2, "prelu": 3}


def pack_conv_key(
    dtype: str,
    group: str,
    kshape: str,
    bias: bool,
    act: str,
    per_ch: bool,
    c_align: bool = False,
) -> int:
    """Must match dl_conv_pack_key() in dl_base_conv_select.cpp."""
    try:
        d = _CONV_DTYPE[str(dtype)]
        k = _CONV_KSHAPE[str(kshape)]
        a = _CONV_ACT[str(act)]
    except KeyError as exc:
        raise SystemExit("Conv pack: unknown field %s" % (exc,))
    g = 1 if str(group) == "dw" else 0
    return (
        (d & 3)
        | ((g & 1) << 2)
        | ((k & 3) << 3)
        | ((1 if bias else 0) << 5)
        | ((a & 3) << 6)
        | ((1 if per_ch else 0) << 8)
        | ((1 if c_align else 0) << 9)
    )


def conv_slot_name(rule: dict, slot: str) -> str | None:
    if slot not in rule:
        return None
    val = rule[slot]
    if val is None:
        return None
    name = str(val).strip()
    return name or None


def conv_rule_slots(
    rule: dict, slot_names: list
) -> tuple[str | None, str | None, str | None]:
    vals = [conv_slot_name(rule, str(s)) for s in slot_names]
    while len(vals) < 3:
        vals.append(None)
    return (vals[0], vals[1], vals[2])


def _c_intern(name: str | None) -> str:
    return "nullptr" if not name else '"%s"' % name


def pack_conv_rule(rule: dict) -> int:
    return pack_conv_key(
        str(rule.get("dtype")),
        str(rule.get("group")),
        str(rule.get("kshape")),
        bool(rule.get("bias")),
        str(rule.get("act")),
        bool(rule.get("per_ch")),
        bool(rule.get("c_align")),
    )


_INTERN_PREFIX = {
    "pie_v2": "dl_esp32p4_",
    "pie_v1": "dl_tie728_",
    "c": "dl_c_",
}


def intern_names(kernels_doc: dict, idf_target: str) -> list[str]:
    prefix = _INTERN_PREFIX[select_target(idf_target)]
    return sorted(n for n in kernel_symbol_list(kernels_doc) if n.startswith(prefix))


def write_kernel_inc(path: Path, names: list[str]) -> None:
    lines = ["/* Generated by compile_finalize.py. Do not edit. */"]
    for name in names:
        lines.append(kernel_if(name))
        lines.append('    {"%s", (dl_kernel_erased_t)(%s)},' % (name, name))
        lines.append("#endif")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def conv_select_rows(
    conv_doc: dict, idf_target: str
) -> list[tuple[int, str | None, str | None, str | None]]:
    st = select_target(idf_target)
    slots = (conv_doc.get("slots") or {}).get(st)
    if not isinstance(slots, list) or len(slots) < 2:
        raise SystemExit("Conv.yml: slots.%s must list at least two slots" % st)
    seen: dict[int, tuple[str | None, str | None, str | None]] = {}
    rows: list[tuple[int, str | None, str | None, str | None]] = []
    for i, rule in enumerate(conv_doc.get("rules") or []):
        if not isinstance(rule, dict):
            raise SystemExit("Conv.yml: rule %d is not a mapping" % i)
        if str(rule.get("target")) != st:
            continue
        key = pack_conv_rule(rule)
        pair = conv_rule_slots(rule, slots)
        prev = seen.get(key)
        if prev is None:
            seen[key] = pair
            rows.append((key, pair[0], pair[1], pair[2]))
        elif prev != pair:
            raise SystemExit(
                "Conv.yml: packed key 0x%04x conflict (%s vs %s)" % (key, prev, pair)
            )
    rows.sort(key=lambda r: r[0])
    return rows


def write_conv_select(path: Path, conv_doc: dict, idf_target: str) -> None:
    st = select_target(idf_target)
    rows = conv_select_rows(conv_doc, idf_target)
    lines = [
        "/* Generated packed Conv select for %s (%s). Do not edit. */"
        % (idf_target, st)
    ]
    for key, slot0, slot1, slot2 in rows:
        lines.append(
            "    {0x%04x, %s, %s, %s},"
            % (key, _c_intern(slot0), _c_intern(slot1), _c_intern(slot2))
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate esp-dl compile files from YAML")
    ap.add_argument("--ops", type=Path, help="spec/ops.yml")
    ap.add_argument("--kernels", type=Path, help="spec/kernels.yml")
    ap.add_argument(
        "--conv-yml",
        type=Path,
        default=None,
        help="spec/select/Conv.yml (default: spec/select/Conv.yml next to ops.yml)",
    )
    ap.add_argument("--target", help="IDF_TARGET, e.g. esp32p4")
    ap.add_argument("--component-dir", type=Path)
    ap.add_argument("--header", type=Path)
    ap.add_argument("--srcs", type=Path)
    ap.add_argument("--kernel-inc", type=Path, help="generated dl_kernel.inc")
    ap.add_argument("--conv-select", type=Path, help="generated dl_conv_select.inc")
    ap.add_argument(
        "--module-includes-inc",
        type=Path,
        help="generated dl_module_includes.inc (gated module-header includes)",
    )
    ap.add_argument(
        "--module-register-inc",
        type=Path,
        help="generated dl_module_register.inc (gated register_module rows)",
    )
    ap.add_argument(
        "--req", type=Path, nargs="*", default=[], help="compile_req.yml files"
    )
    args = ap.parse_args()

    missing_gen = [
        n
        for n in ("ops", "kernels", "target", "component_dir", "header", "srcs")
        if getattr(args, n) is None
    ]
    if missing_gen:
        raise SystemExit(
            "compile_finalize.py requires --%s"
            % ", --".join(p.replace("_", "-") for p in missing_gen)
        )

    ops_doc = load_yaml(args.ops)
    if not isinstance(ops_doc, dict):
        raise SystemExit("ops.yml: expected a mapping")
    kernels_doc = load_yaml(args.kernels)
    if not isinstance(kernels_doc, dict):
        raise SystemExit("kernels.yml: expected a mapping")

    alias_map = op_canonical_map(ops_doc)
    legal_kernels = kernel_name_set(kernels_doc)
    conv_yml = (
        args.conv_yml
        if args.conv_yml is not None
        else args.ops.parent / "select" / "Conv.yml"
    )
    validate_select(conv_yml, ops_doc, legal_kernels)
    spec_abi = kernels_doc.get("kernel_abi")
    spec_abi_i = int(spec_abi) if spec_abi is not None else None

    compile_all = not args.req
    if compile_all:
        ops: set[str] | None = None
        kernels: set[str] = set()
    else:
        ops_set, kernels, req_abi = union_reqs(args.req)
        if req_abi is not None and spec_abi_i is not None and req_abi > spec_abi_i:
            raise SystemExit(
                "compile_req kernel_abi %s is newer than kernels.yml kernel_abi %s"
                % (req_abi, spec_abi_i)
            )
        unknown_k = sorted(k for k in kernels if k not in legal_kernels)
        if unknown_k:
            print(
                "compile_req: ignoring kernel(s) not in kernels.yml: %s"
                % ", ".join(unknown_k),
                file=sys.stderr,
            )
            kernels = {k for k in kernels if k in legal_kernels}
        ops = ops_set

    srcs = resolve_srcs(ops_doc, args.target, ops, alias_map)
    missing = [s for s in srcs if not (args.component_dir / s).is_file()]
    if missing:
        raise SystemExit("spec paths do not exist:\n  " + "\n  ".join(missing))

    write_header(args.header, compile_all, ops or set(), kernels)
    write_srcs_cmake(args.srcs, srcs)
    mods = module_specs(ops_doc)
    bad_headers = sorted(
        {
            h
            for _op, h, _cls in mods
            if not (args.component_dir / "dl/module/include" / h).is_file()
        }
    )
    if bad_headers:
        raise SystemExit(
            "ops.yml module headers not found:\n  " + "\n  ".join(bad_headers)
        )
    if args.kernel_inc is not None:
        write_kernel_inc(args.kernel_inc, intern_names(kernels_doc, args.target))
    if args.conv_select is not None:
        write_conv_select(args.conv_select, load_yaml(conv_yml), args.target)
    if args.module_includes_inc is not None:
        write_module_includes(args.module_includes_inc, mods)
    if args.module_register_inc is not None:
        write_module_register(args.module_register_inc, mods)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — CMake needs a clear stderr
        print(exc, file=sys.stderr)
        sys.exit(1)
