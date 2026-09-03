"""Fetch IO-VNBD smartphone sequences from GitHub.

Every CSV in onyekpeu/IO-VNBD is tracked by Git LFS, so raw.githubusercontent.com
serves a ~130-byte pointer file instead of data. The real bytes come from the
media endpoint:

    https://media.githubusercontent.com/media/<owner>/<repo>/<ref>/<path>

We build a manifest from the git tree API, then download only the sequences we
ask for -- the full set is 564 CSVs / multiple GB and role 03 needs the
smartphone (S-*) synchronised subset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import MANIFEST_PATH, RAW_DIR

REPO = "onyekpeu/IO-VNBD"
REF = "master"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{REPO}/{REF}"

LFS_POINTER_MAGIC = b"version https://git-lfs"

# Parent-folder name -> (driver, terrain/route family). Used for route-wise
# splitting: slices from one family must never straddle train and test.
_GROUP_RE = re.compile(r"^(?P<family>[A-Za-z]+)\s*\((?P<driver>Driver [A-Z])\)$")


@dataclass(frozen=True)
class Sequence:
    seq_id: str
    side: str           # "S" = smartphone sensors, "V" = vehicle CAN + survey GNSS
    route: str          # folder name; the S/V pair for one drive shares this
    family: str
    driver: str
    repo_path: str
    size_bytes: int

    @property
    def url(self) -> str:
        return f"{MEDIA_BASE}/{urllib.parse.quote(self.repo_path, safe='/')}"

    @property
    def local_path(self) -> Path:
        return RAW_DIR / f"{self.seq_id}.csv"


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "driftless-ml"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def build_manifest() -> list[Sequence]:
    """Enumerate every synchronised sequence, both the S and V side of each drive.

    Both sides matter: S carries the phone IMU we must run on, V carries dense
    10 Hz ground truth (survey GNSS at 7 decimals, CAN velocity, wheel speeds,
    yaw rate). In the Synchronised set the two share a clock, so V labels the S
    input sample-for-sample.
    """
    tree = _http_json(TREE_API)
    if "tree" not in tree:
        raise RuntimeError(f"unexpected tree API response: {tree}")

    seqs: list[Sequence] = []
    for node in tree["tree"]:
        path = node["path"]
        if node["type"] != "blob" or not path.endswith(".csv"):
            continue
        # Synchronised set only: V and S rows share one clock there.
        if not path.startswith("Synchronised"):
            continue
        # The repo ships the same 72 sequences twice -- once under "Categorised
        # IOVNB Dataset" (nested in <Family> (Driver X)/<route>/ folders) and once
        # flat under "Uncategorised IOVNB Dataset". Keep the categorised copy: it
        # is the only one that tells us the driver and route family, which is what
        # route-wise train/test splitting depends on.
        if "Uncategorised" in path:
            continue
        name = path.rsplit("/", 1)[-1]
        if name[:2] not in ("S-", "V-"):
            continue

        parts = path.split("/")
        route = parts[-2]
        family, driver = "unknown", "unknown"
        for part in reversed(parts[:-1]):
            m = _GROUP_RE.match(part)
            if m:
                family = m.group("family")
                driver = m.group("driver")
                break

        seqs.append(Sequence(
            seq_id=name[:-4],
            side=name[0],
            route=route,
            family=family,
            driver=driver,
            repo_path=path,
            size_bytes=int(node.get("size", 0)),   # pointer size, not payload size
        ))

    seqs.sort(key=lambda s: (s.family, s.route, s.side))

    dupes = {(s.side, s.seq_id) for s in seqs}
    if len(dupes) != len(seqs):
        raise RuntimeError(f"duplicate seq_ids in manifest: {len(seqs)} rows, "
                           f"{len(dupes)} unique")
    unknown = [s.seq_id for s in seqs if s.family == "unknown"]
    if unknown:
        raise RuntimeError(f"could not infer route family for: {unknown}")
    return seqs


def save_manifest(seqs: list[Sequence], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo": REPO,
        "ref": REF,
        "generated_unix": int(time.time()),
        "count": len(seqs),
        "sequences": [asdict(s) for s in seqs],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_manifest(path: Path = MANIFEST_PATH) -> list[Sequence]:
    data = json.loads(path.read_text())
    return [Sequence(**s) for s in data["sequences"]]


def download(seq: Sequence, force: bool = False, retries: int = 3) -> tuple[bool, str]:
    """Download one sequence. Returns (downloaded_now, message)."""
    dest = seq.local_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        head = dest.open("rb").read(len(LFS_POINTER_MAGIC))
        if head != LFS_POINTER_MAGIC and dest.stat().st_size > 1024:
            return False, f"skip (have {dest.stat().st_size / 1e6:.1f} MB)"

    tmp = dest.with_suffix(".csv.part")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                seq.url, headers={"User-Agent": "driftless-ml"})
            with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as out:
                total = int(r.headers.get("content-length", 0))
                got = 0
                while chunk := r.read(1 << 20):
                    out.write(chunk)
                    got += len(chunk)
            if tmp.open("rb").read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC:
                tmp.unlink(missing_ok=True)
                return False, "ERROR: got an LFS pointer, not data"
            if total and got != total:
                last_err = f"truncated {got}/{total}"
                continue
            tmp.replace(dest)
            return True, f"ok ({got / 1e6:.1f} MB)"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 * attempt)

    tmp.unlink(missing_ok=True)
    return False, f"FAILED after {retries} tries: {last_err}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download IO-VNBD smartphone sequences.")
    ap.add_argument("--refresh-manifest", action="store_true",
                    help="re-query the GitHub tree API")
    ap.add_argument("--side", choices=["S", "V", "both"], default="both",
                    help="S = phone sensors, V = vehicle ground truth")
    ap.add_argument("--families", nargs="*", default=None,
                    help="restrict to these route families, e.g. S Vta Vtb Vw")
    ap.add_argument("--seq", nargs="*", default=None, help="explicit seq_ids")
    ap.add_argument("--limit", type=int, default=None, help="cap number of sequences")
    ap.add_argument("--list", action="store_true", help="list and exit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.refresh_manifest or not MANIFEST_PATH.exists():
        print("building manifest from GitHub tree API...", file=sys.stderr)
        seqs = build_manifest()
        save_manifest(seqs)
        n_s = sum(1 for x in seqs if x.side == "S")
        print(f"manifest: {len(seqs)} files ({n_s} S + {len(seqs)-n_s} V) "
              f"-> {MANIFEST_PATH}", file=sys.stderr)
    seqs = load_manifest()

    if args.side != "both":
        seqs = [s for s in seqs if s.side == args.side]
    if args.families:
        want = {f.lower() for f in args.families}
        seqs = [s for s in seqs if s.family.lower() in want]
    if args.seq:
        want_ids = set(args.seq)
        seqs = [s for s in seqs if s.seq_id in want_ids]
    if args.limit:
        seqs = seqs[: args.limit]

    if args.list:
        by_family: dict[str, list[str]] = {}
        for s in seqs:
            by_family.setdefault(f"{s.family} ({s.driver})", []).append(
                f"{s.side}:{s.route}")
        for k in sorted(by_family):
            ids = by_family[k]
            print(f"{k:22} {len(ids):3d}  {', '.join(sorted(set(ids)))}")
        n_s = sum(1 for x in seqs if x.side == "S")
        print(f"\ntotal {len(seqs)} files: {n_s} S + {len(seqs)-n_s} V")
        return 0

    ok = fail = 0
    for i, s in enumerate(seqs, 1):
        did, msg = download(s, force=args.force)
        status = "FAIL" if msg.startswith(("ERROR", "FAILED")) else "OK  "
        if status == "FAIL":
            fail += 1
        else:
            ok += 1
        print(f"[{i:3d}/{len(seqs)}] {status} {s.seq_id:12} {msg}", flush=True)

    print(f"\n{ok} ok, {fail} failed -> {RAW_DIR}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
