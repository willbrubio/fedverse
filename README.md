# fedverse

The purpose of fedverse is to create a unifying package of libraries that concern themselves with the analysis of FED devices in conjunction with the established workflow for the SSPsyGene project. Currently, fedverse inherits functions from fed3bandit written by Alex Legaria (https://fed3bandit.readthedocs.io/en/latest/analysis/fed3live_api.html) and the fed3 library written by Tom Earnest (https://earnestt1234.github.io/fed3/fed3/index.html). Additionally, this library includes code written for FED3Analyses by Chantelle Murrell (https://github.com/KravitzLab/FED3Analyses). 

This library was created for the intention of of automating a analysis pipeline for the WUSMAC SSPsyGene analysis of FED devices. 

The pipeline created by this library is meant to ingest L1 files alongside their key file and automate the process of creating an L4. Anything besides that functionality is outside the scope of the intended pipeline. 


## Install Instructions
To install this library you can run:
```
pip install fedverse
```

## import fedverse
You can import different sub modules by calling them out:
```
import fedverse
from fedverse.fedcore import core
from fedverse.fedassays import fedbandit, fedfr, fedpr1
```

## Run Pipeline
This package is built for to analyze SSPsyGene behavioral data. To that end the core function of this library is to support the pipeline functions that will create an "L4" deliverable for bandit, PR, and FR assays. The pipeine requires the path to fed files within a zipped folder and the path to SSPsyGene style key as well as the path you want outputs to be written in to. Additionally, the bandit pipeline allows for a "bandittype" argument to be passed in to differentiate bandit 80 & 100 schematics. Run the pipelines as follows:  
```
# Bandit Pipeline
root = Path(r"your/path/here")
res = fedbandit.run_bandit_l1_l4(root / "Bandit100_L1.zip",
                                 root / "Bandit_Key.xlsx", root, bandittype = "100")

# FR Pipeline
res = fedfr.run_fr_l1_l4(root / "FR_L1.zip",
                          root / "FR_Key.xlsx", root)

# PR1 Pipeline
res = fedpr1.run_pr_l1_l4(root / "PR1_L1.zip",
                          root / "PR1_Key.xlsx", root)
```


## Sub Library Architecture
This library consists of sub libraries in this specific heiarchy:
```
├──extracted
|   ├──fed3_fedframe.py
|   ├──fed3_loading.py
|   └──fed3bandit_extracted.py
├──fedassays
|   ├──fedbandit.py
|   ├──fedpr1.py
|   └──fedfr1.py
├──fedassets
├──fedcore
|   └──core.py
├──fedutils
|   └──fedlog.py
```
extracted:
sublibrary extracted contains code from the fed3 and fed3bandit library needed for the analysis workflows.

fedassays:
sublibrary contains code from Murrell that conducts the analysis of different assays to replicate the openly available colab journals.

fedcore:
sublibrary contains code from Murrell shared across all assay types used to read in processed data.

fedutils:
sublibrary contains utility functions to help deliver structured messages throughout the library.



## AI Attribution
Code was written with Assistance using generative AI. 
ver: Claude Opus 4.8