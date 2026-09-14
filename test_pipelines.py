#!/usr/bin/env python3
"""Run the FED3 pipeline over every copied model/assay, then diff L3 vs l3_og.

Name-agnostic: models, L1 zips, and keys are discovered on disk, so new data
dropped into DATA_ROOT is picked up with no edits here. Adding a *new assay*
(vs a new model) is the only thing that needs a change — one entry in
ASSAY_RUNNERS.

Flow:
    plan  (default)  list the jobs + resolved inputs, run nothing
    --go             run the pipeline per job, then compare its L3 to l3_og
    --compare-only   skip running, just compare existing pipeline L3 to l3_og

Examples:
    python test_pipelines.py                           # plan: resolved l1/key per job
    python test_pipelines.py --go                       # run + compare, all jobs
    python test_pipelines.py --go --models 001_myt1l    # just one model
    python test_pipelines.py --compare-only             # re-diff without re-running
    python test_pipelines.py --go --detail cells.csv    # + per-cell agreement log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

# --- Configuration -----------------------------------------------------------

# TESTDATADIR populated by the copy script (its DST_ROOT). Edit or pass --data.
DATA_ROOT = Path(r"C:\Users\william.b\Documents\Github\fedverse\TESTDATADIR")

# Subfolder (under each assay root) that the pipeline WRITES its L3 into.
PIPELINE_L3_SUBDIR = "L3"

# Reference preprocessed data copied over as l3_og (matches the copy script).
REFERENCE_L3_SUBDIR = "l3_og"

# Absolute tolerance for numeric CSV comparison.
FLOAT_TOL = 1e-5

# Cap on per-cell disagreement rows written per file, so a wildly-off file
# can't blow up the detail CSV. Summary still counts every disagreement.
MAX_CELL_ROWS_PER_FILE = 1000


# --- Per-assay pipeline dispatch --------------------------------------------
# Each runner takes (root, l1_zip, key) and invokes the right pipeline call.
# root is the assay dir; the pipeline writes its L3/L4 output under it.

def _run_bandit(root: Path, l1_zip: Path, key: Path, bandittype: str):
    # imported here so plan / compare-only work without the pipeline installed
    from fedverse.fedassays import fedbandit
    return fedbandit.run_bandit_l1_l4(l1_zip, key, root, bandittype=bandittype)


def _run_fr(root: Path, l1_zip: Path, key: Path):
    from fedverse.fedassays import fedfr  # fr module for the fr runner
    return fedfr.run_fr_l1_l4(l1_zip, key, root)


def _run_pr(root: Path, l1_zip: Path, key: Path):
    from fedverse.fedassays import fedpr1  # pr module for the pr runner
    return fedpr1.run_pr_l1_l4(l1_zip, key, root)


def _run_beam(root: Path, l1_zip: Path, key: Path):
    from fedverse.fedassays import beam
    return beam.run_beam_l1_l4(l1_zip, key, root)


# assay folder name -> callable(root, l1_zip, key). Register new assays here.
# All lambdas forward (root, l1, key) unchanged — no reordering.
ASSAY_RUNNERS = {
    "bandit100": lambda root, l1, key: _run_bandit(root, l1, key, "100"),
    "bandit80":  lambda root, l1, key: _run_bandit(root, l1, key, "80"),
    "fr1":       lambda root, l1, key: _run_fr(root, l1, key),
    "pr":        lambda root, l1, key: _run_pr(root, l1, key),
    "beam":      lambda root, l1, key: _run_beam(root, l1, key),
}


# --- Discovery helpers -------------------------------------------------------

def find_child_dir(parent: Path, name: str) -> Path | None:
    # case-insensitive so any odd casing still resolves
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def find_key(model_root: Path) -> Path | None:
    # key lives once per model at the model root (copied as <gene>_<n>_key.xlsx)
    for f in model_root.iterdir():
        if (f.is_file() and "key" in f.name.lower()
                and f.suffix.lower() in {".xlsx", ".xls"}):
            return f
    return None


def find_l1_zip(l1_dir: Path, assay_name: str) -> Path | None:
    # discover the L1 zip without hardcoding its name
    zips = [z for z in l1_dir.glob("*.zip")]
    if not zips:
        return None
    if len(zips) == 1:
        return zips[0]
    # multiple: prefer the one whose name references this assay, else first
    for z in zips:
        if assay_name in z.name.lower():
            return z
    print(f"    warn: {len(zips)} zips in {l1_dir}, using {zips[0].name}")
    return zips[0]


def discover_jobs(data_root: Path, want_models: set[str] | None,
                  want_assays: set[str] | None):
    # yield (model_dir, assay_dir, assay_name) for every registered assay found
    for model_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if want_models and model_dir.name not in want_models:
            continue
        for assay_name in ASSAY_RUNNERS:
            if want_assays and assay_name not in want_assays:
                continue
            assay_dir = find_child_dir(model_dir, assay_name)
            if assay_dir:
                yield model_dir, assay_dir, assay_name


# --- Comparison (file level) -------------------------------------------------

def _hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rel_files(base: Path) -> dict[Path, Path]:
    # relative path -> absolute path, for every file under base
    return {p.relative_to(base): p for p in base.rglob("*") if p.is_file()}


def _frames_match(a: Path, b: Path, tol: float) -> bool:
    # tolerant CSV/TSV compare: shape + columns exact, numerics within tol
    import numpy as np
    import pandas as pd
    sep = "\t" if a.suffix.lower() == ".tsv" else ","
    da, db = pd.read_csv(a, sep=sep), pd.read_csv(b, sep=sep)
    if da.shape != db.shape or list(da.columns) != list(db.columns):
        return False
    for col in da.columns:
        sa, sb = da[col], db[col]
        numeric = (pd.api.types.is_numeric_dtype(sa)
                   and pd.api.types.is_numeric_dtype(sb))
        if numeric:
            # assumes matching row order (deterministic L3 output)
            if not np.allclose(sa.to_numpy(), sb.to_numpy(),
                               rtol=0, atol=tol, equal_nan=True):
                return False
        elif not sa.astype(str).equals(sb.astype(str)):
            return False
    return True


def _files_match(a: Path, b: Path, tol: float) -> bool:
    # CSV/TSV -> tolerant frame compare; anything else -> exact byte hash
    if a.suffix.lower() in {".csv", ".tsv"}:
        try:
            return _frames_match(a, b, tol)
        except Exception:
            pass  # unparseable -> fall back to hashing
    return _hash(a) == _hash(b)


def compare_dirs(pipeline_dir: Path, og_dir: Path, tol: float) -> dict:
    new_map, og_map = _rel_files(pipeline_dir), _rel_files(og_dir)
    new_keys, og_keys = set(new_map), set(og_map)
    common = new_keys & og_keys

    differing = [rel for rel in sorted(common)
                 if not _files_match(new_map[rel], og_map[rel], tol)]
    return {
        "identical": len(common) - len(differing),
        "differing": differing,
        "only_in_pipeline": sorted(new_keys - og_keys),
        "only_in_og": sorted(og_keys - new_keys),
    }


# --- Comparison (cell level, for --detail) -----------------------------------

def compare_cells(pipeline_dir: Path, og_dir: Path, tol: float, job: str):
    """Per-cell agreement check over common CSV/TSV files.

    Returns (cell_rows, file_rows):
      cell_rows  one row per DISAGREEING cell (capped per file)
      file_rows  one summary row per file: agreement fraction, max |Δ|, etc.
    """
    import numpy as np
    import pandas as pd

    new_map, og_map = _rel_files(pipeline_dir), _rel_files(og_dir)
    cell_rows: list[dict] = []
    file_rows: list[dict] = []

    for rel in sorted(set(new_map) & set(og_map)):
        a, b = new_map[rel], og_map[rel]
        if a.suffix.lower() not in {".csv", ".tsv"}:
            continue  # cell check only applies to tabular files
        sep = "\t" if a.suffix.lower() == ".tsv" else ","
        try:
            da = pd.read_csv(a, sep=sep)
            db = pd.read_csv(b, sep=sep)
        except Exception as e:
            file_rows.append({"job": job, "file": str(rel),
                              "result": f"unreadable: {e}"})
            continue

        base = {"job": job, "file": str(rel),
                "pipe_shape": str(da.shape), "og_shape": str(db.shape)}
        # can't align cells if the grids don't match; report structurally
        if list(da.columns) != list(db.columns):
            file_rows.append({**base, "result": "columns_mismatch"})
            continue
        if da.shape != db.shape:
            file_rows.append({**base, "result": "shape_mismatch"})
            continue

        emitted = 0          # cell rows written for this file (capped)
        truncated = False    # hit the per-file cap?
        n_disagree = 0       # counted regardless of cap
        max_abs = 0.0

        for col in da.columns:
            sa, sb = da[col], db[col]
            # is_numeric_dtype handles extension dtypes (e.g. StringDtype),
            # which np.issubdtype chokes on
            numeric = (pd.api.types.is_numeric_dtype(sa)
                       and pd.api.types.is_numeric_dtype(sb))
            if numeric:
                # na_value=nan so nullable Int64/Float64 NA -> plain np.nan
                va = sa.to_numpy(dtype=float, na_value=np.nan)
                vb = sb.to_numpy(dtype=float, na_value=np.nan)
                # positional compare; NaN vs NaN counts as agreement
                bad = np.where(~np.isclose(va, vb, rtol=0, atol=tol,
                                           equal_nan=True))[0]
                n_disagree += len(bad)
                for i in bad:
                    d = abs(va[i] - vb[i])
                    if not np.isnan(d):
                        max_abs = max(max_abs, d)
                    if emitted < MAX_CELL_ROWS_PER_FILE:
                        cell_rows.append({
                            "job": job, "file": str(rel), "row": int(i),
                            "column": col, "pipeline": va[i], "og": vb[i],
                            "abs_diff": d, "kind": "numeric"})
                        emitted += 1
                    else:
                        truncated = True
            else:
                # exact text compare. fill NA with a sentinel BEFORE comparing so
                # nullable-string eq never yields pd.NA (which breaks np.where):
                # both-NA -> sentinel==sentinel -> agree; NA-vs-value -> disagree
                _NA = "\x00__na__\x00"
                aa = sa.astype("string").fillna(_NA).to_numpy()
                bb = sb.astype("string").fillna(_NA).to_numpy()
                idx = np.where(aa != bb)[0]  # plain object/str array, no NA
                n_disagree += len(idx)
                for i in idx:
                    if emitted < MAX_CELL_ROWS_PER_FILE:
                        cell_rows.append({
                            "job": job, "file": str(rel), "row": int(i),
                            "column": col, "pipeline": sa.iloc[int(i)],
                            "og": sb.iloc[int(i)], "abs_diff": "",
                            "kind": "text"})
                        emitted += 1
                    else:
                        truncated = True

        n_cells = int(da.size)
        file_rows.append({
            **base,
            "result": "ok" if n_disagree == 0 else "cells_differ",
            "cells": n_cells, "disagree": int(n_disagree),
            "agree_frac": round(1 - n_disagree / n_cells, 6) if n_cells else "",
            "max_abs_diff": max_abs, "truncated": truncated})

    # files on only one side can't be cell-compared; record for completeness
    for rel in sorted(set(new_map) - set(og_map)):
        file_rows.append({"job": job, "file": str(rel),
                          "result": "only_in_pipeline"})
    for rel in sorted(set(og_map) - set(new_map)):
        file_rows.append({"job": job, "file": str(rel), "result": "only_in_og"})

    return cell_rows, file_rows


def write_detail(cell_path: Path, cell_rows: list, file_rows: list) -> Path:
    # per-cell disagreements
    cell_fields = ["job", "file", "row", "column", "pipeline", "og",
                   "abs_diff", "kind"]
    with open(cell_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cell_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cell_rows)

    # per-file agreement summary written alongside (stem + _summary.csv)
    summary_path = cell_path.with_name(cell_path.stem + "_summary.csv")
    sum_fields = ["job", "file", "result", "pipe_shape", "og_shape",
                  "cells", "disagree", "agree_frac", "max_abs_diff", "truncated"]
    with open(summary_path, "w", newline="") as f:
        # missing keys fill blank; extras dropped, so partial rows are fine
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(file_rows)
    return summary_path


# --- Driver ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--go", action="store_true",
                      help="run the pipeline then compare (default: plan only)")
    mode.add_argument("--compare-only", action="store_true",
                      help="compare existing pipeline L3 vs l3_og, don't run")
    p.add_argument("--models", nargs="*", default=None,
                   help="restrict to these model folder names")
    p.add_argument("--assays", nargs="*", default=None,
                   help="restrict to these assay names (e.g. bandit100 pr)")
    p.add_argument("--data", type=Path, default=DATA_ROOT, help="TESTDATADIR")
    p.add_argument("--tol", type=float, default=FLOAT_TOL,
                   help="absolute tolerance for numeric CSV compare")
    p.add_argument("--report", type=Path, default=None,
                   help="optional CSV path for the per-job counts summary")
    p.add_argument("--detail", type=Path, default=None,
                   help="optional CSV path for per-cell disagreements "
                        "(also writes <stem>_summary.csv with agreement stats)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data
    if "CHANGE" in str(data_root):
        sys.exit("Set DATA_ROOT (or pass --data) to your TESTDATADIR first.")
    if not data_root.is_dir():
        sys.exit(f"Data root not found: {data_root}")

    want_models = set(args.models) if args.models else None
    want_assays = set(args.assays) if args.assays else None
    jobs = list(discover_jobs(data_root, want_models, want_assays))
    if not jobs:
        sys.exit("No matching model/assay jobs found.")

    plan = not (args.go or args.compare_only)
    rows = []                 # per-job counts for --report
    all_cell_rows: list = []  # per-cell disagreements for --detail
    all_file_rows: list = []  # per-file agreement summary for --detail

    for model_dir, assay_dir, assay_name in jobs:
        tag = f"{model_dir.name}/{assay_name}"
        l1_dir = find_child_dir(assay_dir, "l1")
        l1_zip = find_l1_zip(l1_dir, assay_name) if l1_dir else None
        key = find_key(model_dir)

        # every job needs both inputs; report and skip if either is missing
        if l1_zip is None or key is None:
            miss = "L1 zip" if l1_zip is None else "key"
            print(f"[skip] {tag} — missing {miss}")
            rows.append({"job": tag, "status": f"skip (no {miss})"})
            continue

        if plan:
            print(f"[plan] {tag}\n       l1  = {l1_zip}\n       key = {key}")
            rows.append({"job": tag, "status": "planned"})
            continue

        # run unless we're only comparing pre-existing output
        if not args.compare_only:
            print(f"[run]  {tag}")
            try:
                ASSAY_RUNNERS[assay_name](assay_dir, l1_zip, key)
            except Exception as e:
                print(f"       pipeline error: {e}")
                rows.append({"job": tag, "status": f"run error: {e}"})
                continue

        # compare pipeline L3 against the reference l3_og
        pipe_l3 = find_child_dir(assay_dir, PIPELINE_L3_SUBDIR)
        og_l3 = find_child_dir(assay_dir, REFERENCE_L3_SUBDIR)
        if pipe_l3 is None or og_l3 is None:
            miss = PIPELINE_L3_SUBDIR if pipe_l3 is None else REFERENCE_L3_SUBDIR
            print(f"       cannot compare — no {miss} dir")
            rows.append({"job": tag, "status": f"no {miss} dir"})
            continue

        r = compare_dirs(pipe_l3, og_l3, args.tol)
        ok = not (r["differing"] or r["only_in_pipeline"] or r["only_in_og"])
        status = "MATCH" if ok else "DIFF"
        print(f"       {status}  identical={r['identical']} "
              f"differ={len(r['differing'])} "
              f"only_pipeline={len(r['only_in_pipeline'])} "
              f"only_og={len(r['only_in_og'])}")
        # show the offending files so a DIFF is actionable
        for rel in r["differing"]:
            print(f"         ~ {rel}")
        for rel in r["only_in_pipeline"]:
            print(f"         + {rel} (pipeline only)")
        for rel in r["only_in_og"]:
            print(f"         - {rel} (og only)")

        rows.append({
            "job": tag, "status": status,
            "identical": r["identical"], "differing": len(r["differing"]),
            "only_in_pipeline": len(r["only_in_pipeline"]),
            "only_in_og": len(r["only_in_og"]),
        })

        # per-cell agreement pass (only when requested — it re-reads the CSVs)
        if args.detail:
            c_rows, f_rows = compare_cells(pipe_l3, og_l3, args.tol, tag)
            all_cell_rows += c_rows
            all_file_rows += f_rows
            # concise readout: only the files that actually disagree
            for fr in f_rows:
                if fr.get("result") == "cells_differ":
                    note = " (truncated)" if fr.get("truncated") else ""
                    print(f"         · {fr['file']}: "
                          f"{fr['disagree']}/{fr['cells']} cells, "
                          f"max|Δ|={fr['max_abs_diff']:.3g}{note}")

    # optional machine-readable summaries
    if args.report and rows:
        fields = sorted({k for row in rows for k in row})
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nreport -> {args.report}")

    if args.detail and (all_cell_rows or all_file_rows):
        summary_path = write_detail(args.detail, all_cell_rows, all_file_rows)
        print(f"\ncell detail  -> {args.detail}")
        print(f"file summary -> {summary_path}")


if __name__ == "__main__":
    main()