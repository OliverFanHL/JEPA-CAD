"""Linear-time STEP entity scanner for downstream TMCAD preprocessing.

The original HistCAD-v7 regex requires `);` and can scan catastrophically on
valid TMCAD statements ending `) ;`. This local matcher preserves v7's
`ENTITY_RE.findall(text)` interface without changing the pretrained source.
"""

from __future__ import annotations

import re

PARSER_VERSION = "tmcad_step_statements_v2"
_DATA_START = re.compile(r"(?m)^[ \t]*DATA[ \t]*;", re.IGNORECASE)
_DATA_END = re.compile(r"(?m)^[ \t]*ENDSEC[ \t]*;", re.IGNORECASE)
_STRING_OR_SEMICOLON = re.compile(r"'(?:[^']|'')*'|;", re.DOTALL)
_ENTITY = re.compile(r"^[ \t\r\n]*#(\d+)[ \t\r\n]*=[ \t\r\n]*([A-Z][A-Z0-9_]*)[ \t\r\n]*\((.*)\)[ \t\r\n]*;$", re.DOTALL)
_ASSIGNMENT = re.compile(r"^[ \t\r\n]*#\d+[ \t\r\n]*=")


class StepStatementMatcher:
    """Read semicolon-terminated DATA statements, respecting STEP quotes."""

    def __init__(self) -> None:
        self.last_assignment_count = 0
        self.last_parsed_count = 0

    def findall(self, text: str) -> list[tuple[str, str, str]]:
        start = _DATA_START.search(text)
        if start is None:
            self.last_assignment_count = self.last_parsed_count = 0
            return []
        end = _DATA_END.search(text, start.end())
        stop = end.start() if end is not None else len(text)
        records: list[tuple[str, str, str]] = []
        assignments = 0
        cursor = start.end()
        for token in _STRING_OR_SEMICOLON.finditer(text, cursor, stop):
            if token.group() != ";":
                continue
            statement = text[cursor:token.end()]
            cursor = token.end()
            if not _ASSIGNMENT.match(statement):
                continue
            assignments += 1
            match = _ENTITY.match(statement)
            if match is not None:
                records.append(match.groups())
        self.last_assignment_count = assignments
        self.last_parsed_count = len(records)
        return records


STEP_MATCHER = StepStatementMatcher()


def install_step_matcher(step_graph_module) -> None:
    """Replace only this Python process's matcher, not the v7 source file."""
    step_graph_module.ENTITY_RE = STEP_MATCHER



