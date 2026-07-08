#!/usr/bin/env python3
"""
extract_upstream.py — install upstream packages into an isolated venv and rip the
source of the functions you need (plus their in-module helpers) into your project's
`extracted/` folder, with provenance + license files.

Edit SPECS below, then:  python extract_upstream.py --out src/fed3pipe/extracted
Re-run any time you need more functions or upstream changes.

What it does NOT do by default: follow dependencies across module boundaries
(e.g. fed3.load -> FEDFrame in another file). It DETECTS and REPORTS those so you
can decide to extract vs. reimplement. Use --follow-cross-module to pull them too.
"""

from __future__ import annotations
import argparse, ast, builtins, json, subprocess, sys, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG: declare what you need. One entry per output module.
# ─────────────────────────────────────────────────────────────────────────────
SPECS = [
    {
        # Renamed so the file makes it obvious where the code came from.
        "out_module": "fed3bandit_extracted.py",
        "package": "fed3bandit",
        "pip": "fed3bandit==0.0.5",
        "functions": ["binned_paction", "true_probs", "count_pellets",
                      "count_pokes", "pokes_per_pellet", "reversal_peh"],
    },
    {
        "out_module": "fed3_extract.py",
        "package": "fed3",
        "pip": "git+https://github.com/earnestt1234/fed3.git",
        "functions": ["load", "as_aligned"],
        # FEDFrame is extracted into its own sibling module (fed3_fedframe.py).
        # Tell the closure to STOP at these names and import them at the top of
        # the emitted module instead — prevents duplicating the class definition.
        # The mapping is: {upstream_name: "where to import it from"} where the
        # right-hand side is a Python import string emitted as `from <X> import <name>`.
        "skip_symbols": {"FEDFrame": ".fed3_fedframe"},
    },
    {
        # The FEDFrame class lives in fed3/core/fedframe.py and is what `load()`
        # returns. Extracting it as its own module keeps the dependency explicit
        # rather than hidden inside fed3_loading.py.
        "out_module": "fed3_fedframe.py",
        "package": "fed3",
        "pip": "git+https://github.com/earnestt1234/fed3.git",
        # 'symbols' (not 'functions'): the extractor accepts classes too. Naming the
        # private helper `_filterout` explicitly is belt-and-braces — the closure
        # would catch it as a free name anyway.
        "symbols": ["FEDFrame", "_filterout"],
    },
]

BUILTINS = set(dir(builtins))


# ─────────────────────────────────────────────────────────────────────────────
# venv + install
# ─────────────────────────────────────────────────────────────────────────────
def venv_python(venv: Path) -> Path:
    """Path to the venv's interpreter (Windows: Scripts\\python.exe, POSIX: bin/python)."""
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def ensure_venv(venv: Path, skip_install: bool, specs) -> Path:
    py = venv_python(venv)
    if not py.exists():
        print(f"[venv] creating {venv}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade",
                        "pip", "setuptools"], check=True)
    if not skip_install:
        for s in specs:
            print(f"[pip] installing {s['pip']}  (--no-deps: source only, no dependency tree)")
            subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-deps", s["pip"]], check=True)
    return py


def site_packages(py: Path) -> Path:
    out = subprocess.run(
        [str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def dist_info(sp: Path, package: str):
    """Return (version, commit_or_None, license_text_or_None)."""
    cand = sorted(sp.glob(f"{package}-*.dist-info")) + sorted(sp.glob(f"{package.replace('-','_')}-*.dist-info"))
    if not cand:
        return ("?", None, None)
    di = cand[0]
    version = di.name.split("-", 1)[1].rsplit(".dist-info", 1)[0]
    commit = None
    durl = di / "direct_url.json"
    if durl.exists():
        try:
            commit = json.loads(durl.read_text()).get("vcs_info", {}).get("commit_id")
        except Exception:
            pass
    lic = None
    for p in list(di.glob("licenses/*LICENSE*")) + list(di.glob("*LICENSE*")):
        lic = p.read_text(encoding="utf-8", errors="replace"); break
    return (version, commit, lic)


# ─────────────────────────────────────────────────────────────────────────────
# AST analysis
# ─────────────────────────────────────────────────────────────────────────────
def index_package(pkg_dir: Path):
    """Map every top-level symbol (function OR class) across the package's .py files."""
    # `symbols` unifies functions and classes — both are nameable, extractable units.
    # Keeping them in one dict simplifies the closure logic in resolve().
    symbols, consts, file_imports, file_src = {}, {}, {}, {}
    for f in pkg_dir.rglob("*.py"):
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        file_src[f] = src
        file_imports[f] = {}
        for n in tree.body:
            # Treat classes and functions identically: both are top-level definitions
            # we can locate by name and slice out of the source verbatim.
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                symbols[n.name] = (f, n)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = (f, n)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    file_imports[f][a.asname or a.name.split(".")[0]] = ("import", a.name)
            elif isinstance(n, ast.ImportFrom):
                mod = ("." * (n.level or 0)) + (n.module or "")
                for a in n.names:
                    file_imports[f][a.asname or a.name] = ("from", mod, a.name)
    return symbols, consts, file_imports, file_src


def free_names(node):
    """Names a function OR class references that it does NOT bind locally."""
    bound = set()
    # Functions have parameters; classes don't. Only collect params if present.
    if isinstance(node, ast.FunctionDef):
        for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            bound.add(a.arg)
        if node.args.vararg: bound.add(node.args.vararg.arg)
        if node.args.kwarg:  bound.add(node.args.kwarg.arg)
    used = set()
    for s in ast.walk(node):
        if isinstance(s, ast.Name):
            if isinstance(s.ctx, ast.Store):
                bound.add(s.id)
            else:
                used.add(s.id)
        elif isinstance(s, (ast.FunctionDef, ast.Lambda, ast.ClassDef)):
            # Don't descend into nested defs/classes — their internal refs aren't ours.
            # (ast.walk would otherwise see methods' params/locals as our free names.)
            pass
    return used - bound - BUILTINS


def resolve(spec, symbols, consts, file_imports, follow_cross):
    """
    Closure over local symbols (functions + classes). Classify everything else.
    Returns (ordered symbols, ordered constants, cross-module refs, external imports, pip deps, sibling imports).
    """
    targets = spec.get("symbols") or spec.get("functions") or []
    # Names we should NOT extract; we'll emit `from <X> import <name>` for each at the
    # top of the output module. Use this when the symbol lives in a sibling extracted
    # module (e.g. FEDFrame -> fed3_fedframe), to avoid duplicating definitions.
    skip = spec.get("skip_symbols") or {}
    for t in targets:
        if t not in symbols:
            raise SystemExit(f"[error] {spec['package']}: symbol '{t}' not found in package")

    needed_syms, needed_consts, stack = set(), set(), list(targets)
    cross_module, external, pip_deps = {}, {}, set()
    sibling_imports = {}                  # name -> import-from path (from skip_symbols)

    while stack:
        name = stack.pop()
        # If the user told us to source this name from a sibling module, do that
        # instead of expanding into it. (Done here, before adding to needed_syms.)
        if name in skip:
            sibling_imports[name] = skip[name]
            continue
        if name in needed_syms or name not in symbols:
            continue
        needed_syms.add(name)
        f, node = symbols[name]
        for nm in free_names(node):
            if nm in skip:
                sibling_imports[nm] = skip[nm]
            elif nm in symbols:
                if nm not in needed_syms:
                    stack.append(nm)
            elif nm in consts:
                needed_consts.add(nm)
            elif nm in file_imports.get(f, {}):
                imp = file_imports[f][nm]
                module = imp[1]
                if module.startswith(".") or module.split(".")[0] == spec["package"]:
                    cross_module[nm] = module
                    if follow_cross and nm in symbols:
                        stack.append(nm)
                else:
                    external[nm] = module
                    pip_deps.add(module.split(".")[0])

    ordered_syms = sorted(needed_syms,
                          key=lambda n: (str(symbols[n][0]), symbols[n][1].lineno))
    ordered_consts = sorted(needed_consts,
                            key=lambda n: (str(consts[n][0]), consts[n][1].lineno))
    return (ordered_syms, ordered_consts, cross_module, external,
            sorted(pip_deps), sibling_imports)


def emit_module(out_path: Path, ordered_syms, ordered_consts,
                symbols, consts, file_src, header, sibling_imports=None):
    chunks = [header]
    sibling_imports = sibling_imports or {}
    files = ({symbols[n][0] for n in ordered_syms}
             | {consts[n][0]  for n in ordered_consts})
    imports = []
    # Pull every import line from the source files we're slicing from, but DROP any
    # that bind a name we're going to re-source from a sibling module. Without this,
    # the upstream `from fed3.core import FEDFrame` would coexist with our intended
    # `from .fed3_fedframe import FEDFrame` and the upstream one would win (or fail).
    siblings_set = set(sibling_imports)
    for f in files:
        tree = ast.parse(file_src[f])
        for n in tree.body:
            if isinstance(n, ast.ImportFrom):
                # Skip the import line entirely if all of its bound names are siblings.
                bound = {a.asname or a.name for a in n.names}
                if bound and bound <= siblings_set:
                    continue
            elif isinstance(n, ast.Import):
                bound = {a.asname or a.name.split(".")[0] for a in n.names}
                if bound and bound <= siblings_set:
                    continue
            else:
                continue
            imports.append(ast.get_source_segment(file_src[f], n))
    # Now add the sibling re-imports as the canonical source of those names.
    for nm, where in sorted(sibling_imports.items()):
        imports.append(f"from {where} import {nm}")
    chunks.append("\n".join(dict.fromkeys(imports)))

    for n in ordered_consts:
        f, node = consts[n]
        chunks.append(ast.get_source_segment(file_src[f], node))
    for n in ordered_syms:
        f, node = symbols[n]
        chunks.append(ast.get_source_segment(file_src[f], node))
    out_path.write_text("\n\n\n".join(chunks) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/fedlib/extracted", help="target extracted/ folder")
    ap.add_argument("--venv", default=".extract_venv", help="isolated venv path")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--follow-cross-module", action="store_true",
                    help="also pull same-package deps from other files (e.g. FEDFrame)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    py = ensure_venv(Path(args.venv), args.skip_install, SPECS)
    sp = site_packages(py)
    today = datetime.date.today().isoformat()
    prov_lines = [f"# Extraction provenance\n\nGenerated: {today}\n"]

    for spec in SPECS:
        pkg_dir = sp / spec["package"]
        if not pkg_dir.exists():
            raise SystemExit(f"[error] package dir not found: {pkg_dir}")
        version, commit, lic = dist_info(sp, spec["package"])
        symbols, consts, file_imports, file_src = index_package(pkg_dir)
        ordered_syms, ordered_consts, cross, external, pip_deps, siblings = resolve(
            spec, symbols, consts, file_imports, args.follow_cross_module)

        requested = spec.get("symbols") or spec.get("functions") or []
        helpers   = [n for n in ordered_syms if n not in requested]
        header = (f'"""\nEXTRACTED FROM UPSTREAM — re-validate against the original before editing logic.\n\n'
                  f'Package : {spec["package"]} {version}'
                  + (f' @ {commit[:12]}' if commit else '') + '\n'
                  f'Install : {spec["pip"]}\n'
                  f'Requested: {requested}\n'
                  f'Helpers pulled in: {helpers}\n'
                  f'Constants pulled in: {ordered_consts}\n'
                  f'Sibling imports: {siblings}\n'
                  f'Extracted: {today}\n"""')
        emit_module(out / spec["out_module"],
                    ordered_syms, ordered_consts, symbols, consts, file_src, header,
                    sibling_imports=siblings)

        if lic:
            (out / f"LICENSE.{spec['package']}").write_text(lic, encoding="utf-8")

        print(f"\n=== {spec['out_module']}  ({spec['package']} {version}) ===")
        print(f"  symbols written   : {ordered_syms}")
        if ordered_consts:
            print(f"  constants written : {ordered_consts}")
        if siblings:
            print(f"  sibling imports   : {siblings}")
        print(f"  external deps     : {external or '(none)'}")
        if cross:
            print(f"  !! CROSS-MODULE (defined elsewhere in {spec['package']}): {cross}")
            print(f"     -> decide: extract these too (--follow-cross-module) OR reimplement.")
        prov_lines.append(
            f"## {spec['out_module']}\n"
            f"- package: `{spec['package']}` {version}" + (f" @ `{commit}`" if commit else "") + "\n"
            f"- install: `{spec['pip']}`\n"
            f"- symbols: {ordered_syms}\n"
            f"- constants: {ordered_consts or 'none'}\n"
            f"- runtime deps: {pip_deps}\n"
            f"- cross-module refs (unresolved): {cross or 'none'}\n")

    (out / "PROVENANCE.md").write_text("\n".join(prov_lines), encoding="utf-8")
    print(f"\n[done] wrote modules + LICENSE.* + PROVENANCE.md to {out}/")


if __name__ == "__main__":
    main()