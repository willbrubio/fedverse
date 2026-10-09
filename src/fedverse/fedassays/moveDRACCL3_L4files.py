from pathlib import Path
import shutil

# ============================================================
# SETTINGS
# ============================================================

# Folder containing:
# 001_MYT1L
# 002_DNMT3A
# 003_NF1
# etc.
ROOT = Path(
    r"C:\Users\Murrell\Box\SSPsyGene - WashU Mouse Assay Center\Behavior\DRACC_Share_Folder_WU-SMAC_Behavior_Data"
)

# New directory where everything will be collected
DESTINATION = ROOT / "Collected_L3_L4"


# ONLY search these assay folders
ASSAY_FOLDERS = {
    "1_Bandit 100-0",
    "2_FR1",
    "3_BEAM",
    "4_Bandit 80-20",
    "5_PR",
}


# ============================================================
# CREATE DESTINATION
# ============================================================

DESTINATION.mkdir(exist_ok=True)


# ============================================================
# SEARCH GENE FOLDERS
# ============================================================

for gene_folder in ROOT.iterdir():

    if not gene_folder.is_dir():
        continue

    # Don't search the output directory we are creating
    if gene_folder == DESTINATION:
        continue

    for assay_name in ASSAY_FOLDERS:

        assay_folder = gene_folder / assay_name

        if not assay_folder.exists():
            continue

        print(f"\nChecking: {assay_folder}")


        # ====================================================
        # L4 FOLDER
        # ====================================================

        l4_folder = assay_folder / "L4"

        if l4_folder.is_dir():

            new_name = (
                f"{gene_folder.name}__"
                f"{assay_name}__L4"
            )

            destination = DESTINATION / new_name

            print(f"  Moving L4: {l4_folder}")
            print(f"          -> {destination}")

            shutil.move(
                str(l4_folder),
                str(destination)
            )


        # ====================================================
        # L3 FOLDER
        # ====================================================

        l3_folder = assay_folder / "L3"

        if l3_folder.is_dir():

            new_name = (
                f"{gene_folder.name}__"
                f"{assay_name}__L3"
            )

            destination = DESTINATION / new_name

            print(f"  Moving L3 folder: {l3_folder}")
            print(f"                 -> {destination}")

            shutil.move(
                str(l3_folder),
                str(destination)
            )


        # ====================================================
        # L3 CSV FILES
        # e.g. MYT1L_001_Bandit80_L3.csv
        # ====================================================

        for csv_file in assay_folder.glob("*_L3.xlsx"):

            new_name = (
                f"{gene_folder.name}__"
                f"{assay_name}__"
                f"{csv_file.name}"
            )

            destination = DESTINATION / new_name

            print(f"  Moving L3 CSV: {csv_file}")
            print(f"              -> {destination}")

            shutil.move(
                str(csv_file),
                str(destination)
            )


print("\nDone!")
print(f"Everything was moved to:\n{DESTINATION}")