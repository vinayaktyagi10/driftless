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


def test_readme_export_table_matches_the_export_report():
    """The README's export table is hand-written; the report is generated.

    A reviewer found the two had drifted (the README still advertised a
    `.onnx.data` sidecar that no longer exists). Stale published numbers are the
    defect this repo has hit most often, so pin the table to its source rather
    than re-checking it by eye.
    """
    import json
    import re

    from driftless_train.paths import MODEL_DIR, TRAIN_ROOT

    report_path = MODEL_DIR / "export_report.json"
    if not report_path.exists():
        pytest.skip("no export report committed")
    rep = json.loads(report_path.read_text())
    readme = (TRAIN_ROOT / "README.md").read_text()

    for runtime, key in (("ONNX", "onnx"), ("TFLite", "tflite")):
        sub = rep.get(key, {})
        if sub.get("skipped"):
            continue
        row = next((ln for ln in readme.splitlines()
                    if ln.startswith(f"| **{runtime}**")), None)
        assert row, f"no {runtime} row in the README export table"

        size = re.search(r"\|\s*([\d.]+) KB", row)
        assert size and float(size.group(1)) == sub["size_kb"], (
            f"{runtime} size in README ({size.group(1) if size else '?'} KB) != "
            f"export_report.json ({sub['size_kb']} KB) -- regenerate the table")

        lat = re.search(r"([\d.]+) ms/window", row)
        assert lat and abs(float(lat.group(1))
                           - sub["latency_ms_per_window"]) < 5e-4, (
            f"{runtime} latency in README != export_report.json")

        # The README renders parity to two significant figures, so compare the
        # rendering rather than the raw float.
        rel = re.search(r"([\d.]+e-[\d]+) rel", row)
        assert rel and rel.group(1) == f"{sub['max_rel_diff']:.1e}", (
            f"{runtime} parity in README ({rel.group(1) if rel else '?'}) != "
            f"export_report.json ({sub['max_rel_diff']:.1e})")


def test_no_tracked_artifact_embeds_an_absolute_path():
    """A committed artifact must not carry the producing machine's directories.

    Found in review: `export_report.json` had one, and the exported `.onnx`
    carried 283 of them in per-node stack traces. Both are fixed at the
    generator; this stops them coming back.
    """
    import subprocess

    from driftless_train.paths import REPO_ROOT

    tracked = subprocess.run(["git", "ls-files", "-z", "training/artifacts"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    offenders = []
    for name in tracked.stdout.split("\0"):
        if not name:
            continue
        blob = (REPO_ROOT / name).read_bytes()
        for needle in (b"/Users/", b"/home/", b"C:\\Users\\"):
            if needle in blob:
                offenders.append(f"{name} contains {needle.decode('latin-1')!r}")
                break
    assert not offenders, ("absolute paths in tracked artifacts:\n"
                           + "\n".join(offenders))
