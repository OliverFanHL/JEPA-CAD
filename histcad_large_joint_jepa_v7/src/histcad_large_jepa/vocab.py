from __future__ import annotations

from .schema import CONSTRAINT_TYPES, FEATURE_TYPES, PRIMITIVE_TYPES

NODE_TYPES = (
    "PAD",
    "GLOBAL",
    "FEATURE",
    "PRIMITIVE",
    "CONSTRAINT",
    "BOUNDARY_REF",
    "CONTROL_POINT",
    "KNOT",
)
SUBTYPES = (
    "PAD",
    "UNKNOWN",
    "GLOBAL",
    *FEATURE_TYPES,
    *PRIMITIVE_TYPES,
    *CONSTRAINT_TYPES,
    "near_point",
    "control_point",
    "knot",
)
AUX_TYPES = (
    "NONE",
    "NewBody",
    "Join",
    "Cut",
    "Intersect",
    "Fillet",
    "Chamfer",
    "MINIMUM",
    "HORIZONTAL",
    "VERTICAL",
    "LEFT",
    "RIGHT",
    "Right",
    "Left",
)
PARAMETRIC_RELATIONS = (
    "SELF",
    "GLOBAL_TO_NODE",
    "NODE_TO_GLOBAL",
    "FEATURE_TO_MEMBER",
    "MEMBER_TO_FEATURE",
    "CONSTRAINT_TO_ENTITY",
    "ENTITY_TO_CONSTRAINT",
    "CONSTRAINT_TO_START",
    "START_TO_CONSTRAINT",
    "CONSTRAINT_TO_END",
    "END_TO_CONSTRAINT",
    "CONSTRAINT_TO_MIDDLE",
    "MIDDLE_TO_CONSTRAINT",
    "CONSTRAINT_TO_CENTER",
    "CENTER_TO_CONSTRAINT",
    "PRIMITIVE_TO_PARAMETER",
    "PARAMETER_TO_PRIMITIVE",
    "PREVIOUS_FEATURE",
    "NEXT_FEATURE",
)
STEP_RELATIONS = (
    "SELF",
    "GLOBAL_TO_TYPE",
    "TYPE_TO_GLOBAL",
    "TYPE_REFERENCE",
    "REVERSE_TYPE_REFERENCE",
    "TYPE_HAS_INSTANCE",
    "INSTANCE_OF_TYPE",
    "INSTANCE_REFERENCE",
    "REVERSE_INSTANCE_REFERENCE",
)


def mapping(values: tuple[str, ...]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


NODE_TO_ID = mapping(NODE_TYPES)
SUBTYPE_TO_ID = mapping(SUBTYPES)
AUX_TO_ID = mapping(AUX_TYPES)
PARAMETRIC_RELATION_TO_ID = mapping(PARAMETRIC_RELATIONS)


def lookup(values: dict[str, int], value: str, fallback: str) -> int:
    return values.get(value, values[fallback])
