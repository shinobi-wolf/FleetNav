import os
import csv
import shutil

# Define paths
csv_file_path = 'policy_experiment_map.csv'
runs_path = 'runs'

# Read policy names from the CSV file
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    valid_policies = {row['Policy Name'] for row in csv_reader}

# Delete folders in runs/algo that are not in the valid policies
for folder_name in os.listdir(runs_path):
    folder_path = os.path.join(runs_path, folder_name)
    if os.path.isdir(folder_path) and folder_name not in valid_policies:
        print(f"Deleting folder: {folder_path}")
        shutil.rmtree(folder_path)  # Use os.rmdir for empty folders or shutil.rmtree for non-empty folders
