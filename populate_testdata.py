#!/usr/bin/env python3
"""Populate a project's TESTDATADIR from the WUSMAC Behavior source tree.

Copies a curated, renamed subset of the SSPsyGene mouse-assay data so pipeline
outputs can be diffed against the already-preprocessed L3 data (kept as l3_og).

Layout produced (per model):
    <model>/<key>.xlsx
    <model>/<assay>/l1/...       # raw data source
    <model>/<assay>/l3_og/...    # existing preprocessed, for comparison

Dry-run by default; pass --go to actually copy.

Examples:
    python populate_testdata.py --models 001_MYT1L        # dry-run preview, one model
    python populate_testdata.py --models 001_MYT1L --go   # actually copy that model
    python populate_testdata.py --go                      # copy everything
    python populate_testdata.py --go --verify                    # copy all + validate
    python populate_testdata.py --go --verify --models 001_MYT1L # one model
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

# --- Configuration (rarely changes; runtime behavior is via CLI flags) -------

# Box-synced source root.
SRC_ROOT = Path(
    r"C:\Users\william.b\Box\SSPsyGene - WashU Mouse Assay Center\Behavior"
)

# Where the test fixture tree gets built. Edit this, or override with --dst.
DST_ROOT = Path(r"C:\Users\william.b\Documents\Github\fedverse\TESTDATADIR")

# Source assay folder -> lowercase test name. Only these assays are copied.
ASSAY_MAP = {
    "1_Bandit100_0": "bandit100",
    "4_Bandit80_20": "bandit80",
    "2_FR1":         "fr1",
    "5_PR":          "pr",
}

# Source level folder -> test name.
# L1 = raw data source; L3 = preprocessed, renamed l3_og so the pipeline's own
# L3 output can be validated against the original.
LEVEL_MAP = {
    "L1": "L1",
    "L3": "L3_og",
}

# --- Verify settings (used only with --verify) -------------------------------

# A bad copied archive is hydrated + recopied up to this many times before
# it's reported as a hard failure.
MAX_VERIFY_ATTEMPTS = 3

# Pause between attempts, giving Box time to finish the download.
VERIFY_RETRY_WAIT_S = 2.0


def find_child_dir(parent: Path, name: str) -> Path | None:
    # case-insensitive lookup so any odd Box/OS casing still resolves
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def force_hydrate(path: Path, chunk: int = 1 << 20) -> None:
    # reading end-to-end blocks until Box serves real bytes -> synchronous hydrate
    with open(path, "rb") as f:
        while f.read(chunk):
            pass


def zip_ok(path: Path) -> tuple[bool, str]:
    # a valid archive proves real, fully-hydrated bytes (xlsx are zips too)
    if not zipfile.is_zipfile(path):
        return False, "not a valid zip (placeholder/truncated?)"
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()  # None means every entry's CRC checks out
        return (bad is None), ("" if bad is None else f"corrupt entry: {bad}")
    except zipfile.BadZipFile as e:
        return False, str(e)


def verify_archive(src: Path, dst: Path, rel: Path, failures: list) -> None:
    # validate the copied archive; on a stub, hydrate the source and recopy
    for attempt in range(1, MAX_VERIFY_ATTEMPTS + 1):
        ok, why = zip_ok(dst)
        if ok:
            if attempt > 1:
                print(f"  [verify] {rel} ok (attempt {attempt})")
            return
        if attempt == MAX_VERIFY_ATTEMPTS:
            print(f"  [FAIL] {rel} — {why}")
            failures.append((str(rel), why))
            return
        # force Box to pull the real file, then recopy over the stub
        print(f"  [verify] {rel} bad ({why}); hydrating + recopy…")
        force_hydrate(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        time.sleep(VERIFY_RETRY_WAIT_S)


def copy_tree(src: Path, dst: Path, root: Path, dry_run: bool,
              verify: bool, failures: list) -> None:
    rel = dst.relative_to(root)  # short, readable log line
    if dry_run:
        print(f"  [plan] {rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # dirs_exist_ok merges on re-run instead of erroring on an existing tree
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"  copied {rel}")
    if verify:
        # check every copied archive; map each dst zip back to its source
        for dz in sorted(dst.rglob("*.zip")):
            verify_archive(src / dz.relative_to(dst), dz,
                           dz.relative_to(root), failures)


def copy_file(src: Path, dst: Path, root: Path, dry_run: bool,
              verify: bool, failures: list) -> None:
    rel = dst.relative_to(root)
    if dry_run:
        print(f"  [plan] {rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)  # copy2 preserves mtime
    print(f"  copied {rel}")
    # .xlsx is a zip container so it's checkable; .xls (OLE) is skipped
    if verify and dst.suffix.lower() in {".zip", ".xlsx"}:
        verify_archive(src, dst, rel, failures)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # copying is off by default so a mistaken run can't pull GBs from Box
    p.add_argument("--go", action="store_true",
                   help="actually copy (default: dry-run preview only)")
    p.add_argument("--models", nargs="*", default=None,
                   help="restrict to these model folder names (default: all)")
    p.add_argument("--src", type=Path, default=SRC_ROOT, help="source root")
    p.add_argument("--dst", type=Path, default=DST_ROOT, help="destination root")
    p.add_argument("--verify", action="store_true",
                   help="after copying, integrity-check zips/xlsx and "
                        "hydrate+recopy any Box placeholder stubs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.go
    src_root, dst_root = args.src, args.dst

    if not src_root.is_dir():
        sys.exit(f"Source root not found: {src_root}")
    if "CHANGE" in str(dst_root):
        sys.exit("Set DST_ROOT (or pass --dst) to your TESTDATADIR first.")

    # optional whitelist of model folder names, else every folder under root
    wanted = set(args.models) if args.models else None
    model_dirs = sorted(
        p for p in src_root.iterdir()
        if p.is_dir() and (wanted is None or p.name in wanted)
    )
    if not model_dirs:
        sys.exit("No matching model folders found.")

    print(f"{'DRY-RUN — ' if dry_run else ''}{len(model_dirs)} model(s) "
          f"-> {dst_root}")
    if args.verify and dry_run:
        print("note: --verify has no effect in a dry-run (nothing copied)")

    verify = args.verify and not dry_run
    failures: list = []  # (relpath, reason) for archives still bad after retries

    for model_dir in model_dirs:
        # 001_MYT1L -> 001_myt1l (prefix kept for ordering; drop below if unwanted)
        model_out = dst_root / model_dir.name.lower()
        print(f"\n{model_dir.name}")

        # per-model key spreadsheet, matched at the model root by name + ext
        for key in model_dir.iterdir():
            if (key.is_file() and "key" in key.name.lower()
                    and key.suffix.lower() in {".xlsx", ".xls"}):
                copy_file(key, model_out / key.name.lower(), dst_root,
                          dry_run, verify, failures)

        for src_assay, out_assay in ASSAY_MAP.items():
            assay_dir = find_child_dir(model_dir, src_assay)
            if assay_dir is None:
                print(f"  (skip) {src_assay} — not found")
                continue
            for src_level, out_level in LEVEL_MAP.items():
                level_dir = find_child_dir(assay_dir, src_level)
                if level_dir is None:
                    print(f"  (skip) {src_assay}/{src_level} — not found")
                    continue
                copy_tree(level_dir, model_out / out_assay / out_level,
                          dst_root, dry_run, verify, failures)

    # a clean exit under --verify means the local copy is genuinely runnable
    if failures:
        print(f"\n[verify] {len(failures)} file(s) still bad after "
              f"{MAX_VERIFY_ATTEMPTS} attempts:")
        for rel, why in failures:
            print(f"  - {rel}: {why}")
        sys.exit(1)
    if verify:
        print("\n[verify] all copied archives valid.")


if __name__ == "__main__":
    main()