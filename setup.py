"""Packaging config for fedverse.

Classic setuptools setup() — this file is the single source of truth for
package metadata. pyproject.toml is kept for the build-system declaration
only; if you move a metadata field back into a [project] table there, that
table wins and the value here is silently ignored.

Build and upload:
    python -m pip install --upgrade build twine
    python -m build                 # -> dist/fedverse-<ver>.tar.gz + .whl
    python -m twine check dist/*
    python -m twine upload dist/*
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

# Runtime deps — every third-party package imported under src/fedverse.
# pandas is pinned <3 because the extracted fed3/fed3bandit code relies on
# pandas 2.x behaviour that the 3.0 release changes.
INSTALL_REQUIRES = [
    "pandas<3",
    "numpy",
    "matplotlib",
    "scipy",
    "seaborn",
    "statsmodels",
    "pingouin",
    "tqdm",
    "ipywidgets",
    "ipython",       # IPython.display, used by the assay plotting helpers
    "openpyxl",      # engine for pd.ExcelFile / read_excel on .xlsx L1 files
]

setup(
    name="fedverse",
    version="0.0.1",
    description="FED3 multi-assay analysis pipeline (SSPsyGene)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="William Rubio",
    author_email="bernard.william@gmail.com",
    url="https://github.com/willbrubio/fedverse",
    project_urls={
        "Source": "https://github.com/willbrubio/fedverse",
        "Issues": "https://github.com/willbrubio/fedverse/issues",
    },
    license="MIT",
    license_files=["LICENSE.md", "src/fedverse/extracted/LICENSE.*"],
    # src-layout: importable packages live under src/, not at the repo root.
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # Schematic images resolved at runtime by fedverse.fedassets.get().
    package_data={"fedverse.fedassets": ["*.png", "*.svg", "*.jpg", "*.jpeg"]},
    include_package_data=True,
    zip_safe=False,  # fedassets.get() walks the package directory on disk
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        # No "License ::" classifier on purpose: setuptools>=77 treats the
        # license= value above as an SPDX expression and rejects packages that
        # also carry a legacy license classifier.
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=["FED3", "neuroscience", "behavior", "operant", "SSPsyGene"],
)