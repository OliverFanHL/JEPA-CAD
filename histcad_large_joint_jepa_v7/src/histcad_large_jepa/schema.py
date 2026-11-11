from __future__ import annotations

import ast
import json
import math
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PRIMITIVE_TYPES = ("line", "circle", "arc", "ellipse", "elliptical_arc", "nurbs")
CONSTRAINT_TYPES = (
    "Perpendicular",
    "Parallel",
    "Horizontal",
    "Vertical",
    "Equal",
    "Tangent",
    "Normal",
    "Coincident",
    "Concentric",
    "Fix",
    "Midpoint",
    "Mirror",
    "Angle",
    "Diameter",
    "Radius",
    "MajorRadius",
    "MinorRadius",
    "Distance",
    "Length",
)
FEATURE_TYPES = ("Extrude", "Revolve", "HelixSweep", "Fillet", "Chamfer")
_UNITS = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
}
_NUMBER_UNIT = re.compile(r"(?<=\d)\s+(?=[A-Za-z])|(?<=\.)\s+(?=[A-Za-z])")


def load_history(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a list of feature dictionaries in {path}")
    return value


def primitive_type(name: str) -> str:
    return next(
        (
            kind
            for kind in sorted(PRIMITIVE_TYPES, key=len, reverse=True)
            if name == kind or name.startswith(f"{kind}_")
        ),
        "unknown",
    )


def infer_feature_type(feature: dict[str, Any]) -> str:
    operation = str(feature.get("operation", ""))
    if operation in {"Fillet", "Chamfer"}:
        return operation
    if "pitch" in feature or "turns" in feature:
        return "HelixSweep"
    if "axis" in feature and "start" in feature and "end" in feature:
        return "Revolve"
    if "sketch" in feature:
        return "Extrude"
    return operation or "Unknown"


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id.lower() in _UNITS:
        return _UNITS[node.id.lower()]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("unsupported dimension expression")


def parse_dimension(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace("^", "**")
    if not text or text[0] not in "+-.0123456789(":
        return None
    text = re.sub(r"\bin\b", "inch", text, flags=re.IGNORECASE)
    text = _NUMBER_UNIT.sub("*", text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            result = _safe_eval(ast.parse(text, mode="eval"))
        return result if math.isfinite(result) else None
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_strings(child)


def iter_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        yield float(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_numbers(child)


def constraint_instances(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def uid_from_json_path(path: str | Path) -> str:
    item = Path(path)
    return f"{item.parent.name}/{item.stem}"


def step_path_from_json(path: str | Path) -> Path:
    source = Path(path)
    parts = list(source.parts)
    json_index = next(index for index, value in enumerate(parts) if value == "JSON")
    step_folder = parts[json_index + 1].replace("histcad_sequences_", "histcad_step_")
    return Path(*parts[:json_index], "STEP", step_folder, *parts[json_index + 2 :]).with_suffix(
        ".step"
    )
