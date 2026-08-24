from pathlib import Path
import shutil

# Main Behavior folder
behavior_path = Path(
    r"C:\Users\Murrell\Box\SSPsyGene - WashU Mouse Assay Center\Behavior"
)

# New destination folder
old_behavior_path = behavior_path / "Old_behaviour"

# Task folders to copy from each line
task_folders = [
    "1_Bandit100_0",
    "2_FR1",
    "3_BEAM",
    "4_Bandit80_20",
    "5_PR",
]

# Save the line-folder list before creating Old_behaviour
line_folders = [
    folder
    for folder in behavior_path.iterdir()
    if folder.is_dir() and folder.name != "Old_behaviour"
]

old_behavior_path.mkdir(exist_ok=True)

for line_folder in line_folders:
    # Create a matching line folder inside Old_behaviour
    destination_line = old_behavior_path / line_folder.name
    destination_line.mkdir(exist_ok=True)

    for task_name in task_folders:
        source = line_folder / task_name
        destination = destination_line / task_name

        if source.is_dir():
            print(f"Copying: {line_folder.name} / {task_name}")

            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
            )
        else:
            print(f"Not found: {line_folder.name} / {task_name}")

print("\nFinished copying task folders.")