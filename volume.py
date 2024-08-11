"""
Copy a directory from one volume to another.
I don't think you can use * in modal volume commands, so need to copy each file individually.
Probably a better way to do this though.

python volume.py cp
python volume.py rm
"""
import os
from tqdm import tqdm

def automate_volume_copy():
    source_dir = "fineweb-edu-sample-10BT-chunked-500-HF4-torched"
    destination_dir = "fineweb-edu-sample-10BT-chunked-500-HF4-torched-shuffled"

    # Use tqdm to create a progress bar for the file copying process
    for i in tqdm(range(100), desc="Copying files"):
        file_index = f"{i:05d}"
        source_file = os.path.join(source_dir, f"shard_{file_index}.pt")
        destination_file = os.path.join(destination_dir, f"shard_{file_index}.pt")
        
        command = f"modal volume cp embeddings {source_file} {destination_file}"
        os.system(command)  # Execute the command

def automate_volume_rm():
    source_dir = "fineweb-edu-sample-10BT-chunked-500-HF4-torched"

    # Use tqdm to create a progress bar for the file copying process
    for i in tqdm(range(100), desc="Deleting files"):
        file_index = f"{i:05d}"
        source_file = os.path.join(source_dir, f"shard_{file_index}.pt")
        
        command = f"modal volume rm embeddings {source_file}"
        os.system(command)  # Execute the command


import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Copy or remove files in a volume.")
    parser.add_argument("command", choices=["cp", "rm"], help="Specify 'cp' to copy files or 'rm' to remove files.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    command = args.command
    if command == "cp":
        automate_volume_copy()
    elif command == "rm":
        automate_volume_rm()
    else:
        print("Invalid command. Use 'cp' to copy or 'rm' to remove.")
        sys.exit(1)

if __name__ == "__main__":
    main()
