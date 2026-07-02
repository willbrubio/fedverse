"""
EXTRACTED FROM UPSTREAM — re-validate against the original before editing logic.

Package : fed3 0.0.1 @ 52faa3837b43
Install : git+https://github.com/earnestt1234/fed3.git
Requested: ['load', 'as_aligned']
Helpers pulled in: []
Constants pulled in: []
Sibling imports: {'FEDFrame': '.fed3_fedframe'}
Extracted: 2026-06-30
"""


from collections.abc import Iterable
import os
import warnings
import pandas as pd
from .fed3_fedframe import FEDFrame


def as_aligned(feds, alignment, inplace=False):
    '''
    Helper function for setting the alignment of one or more FEDFrames.
    See `fed3.core.fedframe.FEDFrame.set_alignment()` for more information.

    Parameters
    ----------
    feds : FEDFrame or collection of FEDFrames
        FEDFrames to set alignment for
    alignment: 'str':
        Alignment string.
    inplace : bool
        When True, the FEDFrames are modified in place; otherwise,
        new copies are created.

    Returns
    -------
    aligned or None
        Either one FEDFrame or a list of FEDFrames with new alignment..

    '''
    if isinstance(feds, FEDFrame):
        aligned = feds.set_alignment(alignment, inplace=inplace)
    else:
        aligned = [f.set_alignment(alignment) for f in feds]

    return aligned


def load(path, index_col='MM:DD:YYYY hh:mm:ss', dropna=True,
         deduplicate_index=None, offset='1S', reset_counts=False,
         reset_columns=('Pellet_Count', 'Left_Poke_Count', 'Right_Poke_Count')):
    '''
    Load FED3 data from a CSV/Excel file.  This is the typical
    recommended way for importing FED3 data.  Relies mostly
    on `pandas.read_csv()` and `pandas.read_excel()` for the parsing.

    Parameters
    ----------
    path : str
        System path to FED3 data file.
    index_col : str, optional
        Timestamp column to use as index. The default is 'MM:DD:YYYY hh:mm:ss'.
    dropna : bool, optional
        Remove all empty rows. The default is True.
    deduplicate_index, offset, reset_counts, reset_columns: optional
        Arguments passed to `fed3.FEDFrame.deduplicate_index()`, used
        to remove duplicate timestamps as the data are loaded.

    Returns
    -------
    f : fed3.FEDFrame
        New FEDFrame object.

    '''
    # read the path
    name, ext = os.path.splitext(path)
    ext = ext.lower()

    read_opts = {'.csv':pd.read_csv, '.xlsx':pd.read_excel}
    func = read_opts[ext]
    feddata = func(path,
                   parse_dates=True,
                   index_col=index_col)
    if dropna:
        feddata = feddata.dropna(how='all')

    name = os.path.basename(name)
    f = FEDFrame(feddata)
    f._load_init(name=name,
                 path=path,
                 deduplicate_index=deduplicate_index,
                 offset=offset,
                 reset_counts=reset_counts,
                 reset_columns=reset_columns)

    return f
