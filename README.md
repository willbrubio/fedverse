# fedlib

The purpose of fedlib is to create a unifying package of libraries that concern themselves with the analysis of FED devices in conjunction with the established workflow for the SSPsyGene project. Currently, fedlib inherits functions from fed3bandit written by Alex Legaria (https://fed3bandit.readthedocs.io/en/latest/analysis/fed3live_api.html) and the fed3 library written by Tom Earnest (https://earnestt1234.github.io/fed3/fed3/index.html). Additionally, this library includes code written for FED3Analyses by Chantelle Murrell (https://github.com/KravitzLab/FED3Analyses). 

This library was created for the intention of being used in conjunction with the analysis pipeline for the WUSMAC SSPsyGene analysis pipeline of FED devices. 

## Sub Library Architecture
This library consists of sub libraries in this specific heiarchy:
extracted
    fed3_fedframe.py
    fed3_loading.py
    fed3bandit_extracted.py
fedassays
    fedbandit.py
    fedpr1.py
    fedfr1.py
fedcore
    core.py
fedutils
    fedlog

extracted:
sublibrary extracted contains code from the fed3 and fed3bandit library needed for the analysis workflows.

fedassays:
sublibrary contains code from Murrell that conducts the analysis of different assays to replicate the openly available colab journals.

fedcore:
sublibrary contains code from Murrell shared across all assay types used to read in processed data.

fedutils:
sublibrary contains utility functions to help deliver structured messages throughout the library.



