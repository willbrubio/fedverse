from pathlib import Path
import re

# ============================================================
# DESTINATION ROOT
# ============================================================

ROOT = Path(
    r"C:\Users\Murrell\Box\SSPsyGene - WashU Mouse Assay Center\Behavior\DRACC_Share_Folder_WU-SMAC_Behavior_Data"
)


# ============================================================
# FOLDER NAMES TO CHANGE
# ============================================================

RENAME_MAP = {
    "1_Bandit 100-0": "1_Bandit100_0",
    "4_Bandit 80-20": "4_Bandit80_20",
}


# ============================================================
# RENAME BANDIT FOLDERS FOR GENES 001-020
# ============================================================

renamed = 0

for gene_folder in ROOT.iterdir():

    if not gene_folder.is_dir():
        continue

    # Must begin with 3 digits
    match = re.match(r"^(\d{3})", gene_folder.name)

    if not match:
        continue

    gene_number = int(match.group(1))

    # Only genes 001 through 020
    if not 1 <= gene_number <= 20:
        continue

    print(f"\nChecking: {gene_folder.name}")

    for old_name, new_name in RENAME_MAP.items():

        old_folder = gene_folder / old_name
        new_folder = gene_folder / new_name

        if not old_folder.is_dir():
            continue

        # Prevent accidentally overwriting an existing folder
        if new_folder.exists():
            print(f"  SKIPPED - destination already exists:")
            print(f"    {new_folder}")
            continue

        print(f"  RENAMING:")
        print(f"    {old_name}")
        print(f"    -> {new_name}")

        old_folder.rename(new_folder)

        renamed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Folders renamed: {renamed}")