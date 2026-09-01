"""Structural checks on the generated markdown.

The report/metric writers build markdown by appending strings to a list, and
long sentences are wrapped as adjacent string literals inside those list
literals. That wrapping is deliberate, but it means a *missing comma* and an
*intended concatenation* look identical to a linter -- so the tables are checked
here instead, where a merged or short-changed row actually shows up.
"""

from __future__ import annotations

import pytest

from driftless_train.paths import EVIDENCE_PATH, METRIC_DIR

DOCS = [EVIDENCE_PATH,
        METRIC_DIR / "crossval.md",
        METRIC_DIR / "allan_imu_noise.md",
        METRIC_DIR / "measurement_noise.md",
        METRIC_DIR / "dataset_audit.md"]


def _tables(text: str) -> list[list[str]]:
    """Split into contiguous runs of pipe-prefixed lines."""
    tables, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_tables_are_rectangular(doc):
    if not doc.exists():
        pytest.skip(f"{doc.name} not generated yet")
    for t in _tables(doc.read_text()):
        widths = {line.count("|") for line in t}
        assert len(widths) == 1, (
            f"{doc.name}: ragged table, column counts {sorted(widths)}\n"
            + "\n".join(t[:6]))
        assert len(t) >= 3, f"{doc.name}: table with no body rows:\n{t}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_unrendered_placeholders(doc):
    if not doc.exists():
        pytest.skip(f"{doc.name} not generated yet")
    text = doc.read_text()
    # A stray brace pair means an f-string prefix was lost during line wrapping,
    # so the literal `{expr}` was written out instead of its value.
    for bad in ("{res[", "{cv[", "{ev[", "{af[", "{p30[", "{sp[", "None m"):
        assert bad not in text, f"{doc.name}: unrendered {bad!r}"
