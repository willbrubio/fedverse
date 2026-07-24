# run all pipelines for testing purposes

#!pip install pingouin
import sys
sys.path.insert(0, "src")           # relative to the project root where the notebook runs
from fedverse.extracted import fed3bandit_extracted, fed3_loading, fed3_fedframe
from fedverse.fedcore import core
from fedverse.fedassays import fedbandit, fedpr1, fedfr
from fedverse.fedutils.fedlog import status

# Import other libraries (will have to kick out to top level)
import pandas as pd
from pathlib import Path

# for reloading packages
import importlib

# reload packages
importlib.reload(fedbandit)
importlib.reload(fedpr1)
importlib.reload(fedfr)
importlib.reload(core)

# Run Bandit pipelines for Myt1l and FMR1
# Myt1l
root = Path(r"C:\Users\william.b\Box\Kravitz Lab Box Drive\William\5_fed3_pipeline\TESTDIRDATA\MYT1L\bandit100")
res = fedbandit.run_bandit_l1_l4(root / "MYT1L_001_Bandit100_L1.zip",
                                 root / "MYT1L_001_Key.xlsx", root, bandittype = "100")

# FMR1
root = Path(r"C:\Users\william.b\Box\Kravitz Lab Box Drive\William\5_fed3_pipeline\TESTDIRDATA\FMR1\bandit100")
res = fedbandit.run_bandit_l1_l4(root / "FMR1_007_Bandit100_L1.zip",
                                 root / "FMR1_007_Key.xlsx", root, bandittype = "100")



# Run PR1 pipelines for Myt1l and FMR1
# Myt1l
root = Path(r"C:\Users\william.b\Box\Kravitz Lab Box Drive\William\5_fed3_pipeline\TESTDIRDATA\MYT1L\pr1")
res = fedpr1.run_pr_l1_l4(root / "MYT1L_001_PR1_L1.zip",
                          root / "MYT1L_001_Key.xlsx", root)

# FMR1
root = Path(r"C:\Users\william.b\Box\Kravitz Lab Box Drive\William\5_fed3_pipeline\TESTDIRDATA\FMR1\pr1")
res = fedpr1.run_pr_l1_l4(root / "FMR1_007_PR1_L1.zip",
                          root / "FMR1_007_Key.xlsx", root)


# Run FR1 pipelines for Myt1l and FMR1
# Myt1l
root = Path(r"C:\Users\william.b\Box\Kravitz Lab Box Drive\William\5_fed3_pipeline\TESTDIRDATA\MYT1L\fr1")
res = fedfr.run_fr_l1_l4(root / "MYT1L_001_FR1_L1.zip",
                          root / "MYT1L_001_Key.xlsx", root)

# FMRI
root = Path(r"C:\Users\william.b\Box\Kravitz Lab Box Drive\William\5_fed3_pipeline\TESTDIRDATA\FMR1\fr1")
res = fedfr.run_fr_l1_l4(root / "FMR1_007_FR1_L1.zip",
                          root / "FMR1_007_Key.xlsx", root)

