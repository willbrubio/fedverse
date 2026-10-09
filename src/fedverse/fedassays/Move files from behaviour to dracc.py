from pathlib import Path
import shutil
import re
import time

# ============================================================
# SETTINGS
# ============================================================

SOURCE_ROOT = Path(
    r"C:\Users\Murrell\Box\SSPsyGene - WashU Mouse Assay Center\Behavior"
)

DEST_ROOT = Path(
    r"C:\Users\Murrell\Box\SSPsyGene - WashU Mouse Assay Center\Behavior\DRACC_Share_Folder_WU-SMAC_Behavior_Data"
)

ASSAY_FOLDERS = [
    "1_Bandit100_0",
    "2_FR1",
    "3_BEAM",
    "4_Bandit80_20",
    "5_PR",
]


# ============================================================
# CHOOSE WHAT TO COPY
# ============================================================

# OPTIONS:
#
# "L0_L1"
#     L0 -> *_L0.zip
#     L1 -> *_L1.zip
#
# "L3"
#     L3 -> *_L3.csv
#
# "L4"
#     L4 -> stats_table folder
#     L4 -> *_L4.svg
#
COPY_MODE = "L4"


# Only genes 001 through 020
MIN_GENE = 1
MAX_GENE = 11

MAX_RETRIES = 5
RETRY_DELAY = 3


# ============================================================
# COPY ONE FILE WITH RETRIES
# ============================================================

def copy_file_with_retry(source_file, destination_file):

    destination_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            shutil.copy2(
                source_file,
                destination_file
            )

            print(f"      COPIED: {source_file.name}")

            return True

        except OSError as e:

            print(
                f"      RETRY {attempt}/{MAX_RETRIES}: "
                f"{source_file.name}"
            )

            print(f"         {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    print(
        f"      FAILED AFTER {MAX_RETRIES} ATTEMPTS: "
        f"{source_file}"
    )

    return False


# ============================================================
# COPY A WHOLE FOLDER WITH FILE-BY-FILE RETRIES
# ============================================================

def copy_folder_with_retry(source_folder, destination_folder):

    copied_count = 0
    failed_count = 0

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for item in source_folder.rglob("*"):

        relative_path = item.relative_to(source_folder)

        destination_item = (
            destination_folder
            / relative_path
        )

        # Create subfolders
        if item.is_dir():

            destination_item.mkdir(
                parents=True,
                exist_ok=True
            )

            continue

        # Copy each file
        success = copy_file_with_retry(
            item,
            destination_item
        )

        if success:
            copied_count += 1
        else:
            failed_count += 1

    return copied_count, failed_count


# ============================================================
# GET FILES/FOLDERS TO COPY
# ============================================================

def get_items_to_copy(source_assay):

    jobs = []

    # --------------------------------------------------------
    # L0 + L1
    # --------------------------------------------------------

    if COPY_MODE == "L0_L1":

        for level in ["L0", "L1"]:

            source_folder = source_assay / level

            if not source_folder.is_dir():
                continue

            matching_files = [
                file
                for file in source_folder.iterdir()
                if (
                    file.is_file()
                    and file.suffix.lower() == ".zip"
                    and file.stem.lower().endswith(
                        f"_{level.lower()}"
                    )
                )
            ]

            for file in matching_files:

                jobs.append({
                    "type": "file",
                    "source": file,
                    "level": level,
                })


    # --------------------------------------------------------
    # L3
    # --------------------------------------------------------

    elif COPY_MODE == "L3":

        source_folder = source_assay / "L3"

        if not source_folder.is_dir():
            return jobs

        matching_files = [
            file
            for file in source_folder.iterdir()
            if (
                file.is_file()
                and file.suffix.lower() == ".csv"
                and file.stem.lower().endswith("_l3")
            )
        ]

        for file in matching_files:

            jobs.append({
                "type": "file",
                "source": file,
                "level": "L3",
            })


    # --------------------------------------------------------
    # L4
    # --------------------------------------------------------

    elif COPY_MODE == "L4":

        source_folder = source_assay / "L4"

        if not source_folder.is_dir():
            return jobs

        # ====================================================
        # 1. stats_table folder
        # ====================================================

        stats_folder = source_folder / "stats_table"

        if stats_folder.is_dir():

            jobs.append({
                "type": "folder",
                "source": stats_folder,
                "level": "L4",
            })

        # ====================================================
        # 2. *_L4.svg
        # ====================================================

        matching_svgs = [
            file
            for file in source_folder.iterdir()
            if (
                file.is_file()
                and file.suffix.lower() == ".svg"
                and file.stem.lower().endswith("_l4")
            )
        ]

        for file in matching_svgs:

            jobs.append({
                "type": "file",
                "source": file,
                "level": "L4",
            })


    else:

        raise ValueError(
            "COPY_MODE must be "
            "'L0_L1', 'L3', or 'L4'"
        )

    return jobs


# ============================================================
# MAIN COPY
# ============================================================

files_copied = 0
files_failed = 0
folders_copied = 0


for gene_folder in SOURCE_ROOT.iterdir():

    if not gene_folder.is_dir():
        continue

    gene_name = gene_folder.name

    # Must start with three digits
    match = re.match(
        r"^(\d{3})",
        gene_name
    )

    if not match:
        continue

    gene_number = int(
        match.group(1)
    )

    # ONLY genes 001 through 020
    if not MIN_GENE <= gene_number <= MAX_GENE:
        continue


    print("\n" + "=" * 70)
    print(f"GENE: {gene_name}")
    print("=" * 70)


    # ========================================================
    # ASSAYS
    # ========================================================

    for assay_name in ASSAY_FOLDERS:

        source_assay = (
            gene_folder
            / assay_name
        )

        if not source_assay.is_dir():
            continue


        jobs = get_items_to_copy(
            source_assay
        )


        if not jobs:

            print(
                f"  NO MATCHING ITEMS: "
                f"{assay_name}"
            )

            continue


        print(
            f"\n  ASSAY: {assay_name}"
        )


        # ====================================================
        # COPY EACH ITEM
        # ====================================================

        for job in jobs:

            source_item = job["source"]
            level = job["level"]
            item_type = job["type"]


            # ------------------------------------------------
            # FILE
            # ------------------------------------------------

            if item_type == "file":

                destination_file = (
                    DEST_ROOT
                    / gene_name
                    / assay_name
                    / level
                    / source_item.name
                )

                success = copy_file_with_retry(
                    source_item,
                    destination_file
                )

                if success:
                    files_copied += 1
                else:
                    files_failed += 1


            # ------------------------------------------------
            # FOLDER
            # ------------------------------------------------

            elif item_type == "folder":

                destination_folder = (
                    DEST_ROOT
                    / gene_name
                    / assay_name
                    / level
                    / source_item.name
                )

                print(
                    f"      COPYING FOLDER: "
                    f"{source_item.name}"
                )

                copied, failed = copy_folder_with_retry(
                    source_item,
                    destination_folder
                )

                files_copied += copied
                files_failed += failed
                folders_copied += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Copy mode: {COPY_MODE}")
print(f"Files copied: {files_copied}")
print(f"Files failed: {files_failed}")

if COPY_MODE == "L4":
    print(f"stats_table folders copied: {folders_copied}")

print(
    f"\nDestination:\n{DEST_ROOT}"
)