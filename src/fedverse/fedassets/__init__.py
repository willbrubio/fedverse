"""Bundled image assets for fedlib figures (schematics, logos).

Resolve assets by filename instead of hardcoding absolute paths:

    from fedlib import assets
    assemble_l4(..., schematic_path=assets.get("fr1_schematic.jpg"))
"""
from importlib.resources import files

# Traversable handle to THIS subpackage's folder (fedlib/assets).
# __name__ is "fedlib.assets" on import, so this resolves correctly whether
# fedlib is installed editable, installed normally, or run from source.
_ASSET_DIR = files(__name__)

# Extensions we treat as image assets — used only to build the error list below.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".svg")


def get(name: str):
    """Resolve a bundled asset filename to a path matplotlib can read.

    The return value works with both `Path(...).exists()` and
    `mpimg.imread(str(...))` — exactly what assemble_l4's schematic_path
    argument expects, so it drops straight in.

    Raises FileNotFoundError (listing what IS available) on a bad name, so a
    typo fails loudly right here instead of silently blanking the schematic
    panel — recall the function does `if ... Path(schematic_path).exists()`,
    which would quietly skip a misspelled file.
    """
    path = _ASSET_DIR / name
    if not path.is_file():
        # Friendly "did you mean" list built from whatever actually got bundled.
        available = [
            p.name for p in _ASSET_DIR.iterdir()
            if p.name.lower().endswith(_IMAGE_EXTS)
        ]
        raise FileNotFoundError(
            f"asset {name!r} not found in fedlib.assets. Available: {available}"
        )
    return path