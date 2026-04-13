#!/usr/bin/env python3
"""
V10 - The "Merge-Linker" Script

This script gives up on fixing symlinks and instead *creates* new, 
physical datasets by "merging" two sources using hardlinks.

It takes a "problem" name (e.g., 'DCFace_balanced_10K_Wrafdb'), finds its
paired source ('DCFace_balanced_10K'), and merges all files from
the paired source AND the fixed source ('rafdb_train') into a new
destination directory.

This uses hardlinks (os.link) to be instantaneous and save disk space.
"""

import argparse
import os
import sys
from pathlib import Path
import shutil

# --- User's list of datasets to create ---
# (I removed the duplicate 'sampled_SynFace70V_DA_10_codeformer_Wrafdb')
DATASET_BASENAMES = [
    #"DCFace_balanced_10K_Wrafdb",
    #"DigiFace_balanced_10K_Wrafdb",
    #"EmoNet_Face_Big_balanced_10K_Wrafdb",
    "fineface_images_Wrafdb",
    "finefaceV2_images_Wrafdb",
    #"sampled_SynFace70V_DA_10_codeformer_V2_Wrafdb",
    #"sampled_SynFace70V_DA_10_codeformer_Wrafdb",
    "stable_generated_Wrafdb",
]

KEYWORD = "_Wrafdb"

def copy_dataset_files(source_dir, dest_dir, dry_run=False, collision_log=None):
    """
    Recursively scans source_dir and hardlinks all files into dest_dir,
    preserving the subdirectory structure.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    linked_count = 0
    
    print(f"  Scanning source: {source_dir}...")
    for src_file in source_path.rglob('*'):
        if not src_file.is_file():
            continue

        # Get the relative path of the file (e.g., "class_01/image.jpg")
        rel_path = src_file.relative_to(source_path)
        
        # Create the full destination path
        dest_file = dest_path / rel_path
        
        # Ensure the destination subdirectory exists
        dest_file_dir = dest_file.parent
        if not dest_file_dir.exists():
            if not dry_run:
                dest_file_dir.mkdir(parents=True, exist_ok=True)

        # Check for collisions BEFORE linking
        if dest_file.exists():
            # File already exists, this is a collision
            if collision_log:
                collision_log.append(f"COLLISION (SKIPPED): {dest_file} (from {src_file})")
            continue

        # Create the hardlink
        if not dry_run:
            try:
                shutil.copy2(src_file, dest_file)
            except OSError as e:
                print(f"    ERROR linking {src_file}: {e}", file=sys.stderr)
        
        linked_count += 1
    
    return linked_count

def main():
    parser = argparse.ArgumentParser(description='Create hybrid datasets using hardlinks')
    parser.add_argument('--root-dir', default="/path/to/storage_erdi/syntface_datasets", 
                        help='The top-level directory containing all source datasets (e.g., syntface_datasets)')
    parser.add_argument('--fixed-source-dir', default="/path/to/storage_erdi/syntface_datasets/rafdb_train",
                        help='The path to the fixed, second source (e.g., .../rafdb_train)')
    parser.add_argument('--output-dir', default="/path/to/hybrid_datasets",
                        help='The top-level directory to store the new hybrid datasets (e.g., /path/to/hybrid_datasets)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Only show what would be done, do not create dirs or links')
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    fixed_source_path = Path(args.fixed_source_dir)
    output_path = Path(args.output_dir)

    if not root_path.is_dir():
        print(f"Error: Root source directory not found: {root_path}", file=sys.stderr)
        sys.exit(1)
    if not fixed_source_path.is_dir():
        print(f"Error: Fixed source directory not found: {fixed_source_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"--- Starting Hybrid Dataset Creation ---")
    if args.dry_run:
        print("[DRY-RUN MODE] No files or directories will be created.\n")

    if not output_path.exists() and not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # --- Loop through the user's hardcoded list ---
    for job_name in DATASET_BASENAMES:
        print(f"--- Processing: {job_name} ---")
        
        # 1. Find the Paired Source
        keyword_index = job_name.find(KEYWORD)
        if keyword_index == -1:
            print(f"  Warning: Keyword '{KEYWORD}' not in '{job_name}'. Skipping.")
            continue
        
        paired_name = job_name[:keyword_index]
        paired_source_path = root_path / paired_name
        
        # 2. Define all paths
        dest_dir_path = output_path / job_name
        source_A = paired_source_path
        source_B = fixed_source_path
        
        if not source_A.is_dir():
            print(f"  Error: Paired source not found: {source_A}. Skipping job.")
            continue

        print(f"  Destination: {dest_dir_path}")
        print(f"  Source A (Paired): {source_A}")
        print(f"  Source B (Fixed):  {source_B}")

        if not dest_dir_path.exists() and not args.dry_run:
            dest_dir_path.mkdir(parents=True, exist_ok=True)
            
        collisions = []

        # 3. Copy files from Source A
        print(f"  Copying files from Source A...")
        count_A = copy_dataset_files(source_A, dest_dir_path, args.dry_run, collisions)
        print(f"  Copied {count_A} files from Source A.")

        # 4. Copy files from Source B
        print(f"  Copying files from Source B...")
        count_B = copy_dataset_files(source_B, dest_dir_path, args.dry_run, collisions)
        print(f"  Copied {count_B} files from Source B.")

        print(f"  Job '{job_name}' complete. Total new links: {count_A + count_B}")
        
        if collisions:
            print(f"  WARNING: {len(collisions)} filename collisions were detected and skipped.")
            # Log first 10 collisions
            for log_entry in collisions[:10]:
                print(f"    {log_entry}")
            if len(collisions) > 10:
                print(f"    ... and {len(collisions) - 10} more.")
        print("\n")

    print("--- All Jobs Complete ---")
    if args.dry_run:
        print("Dry-run finished. No files were changed.")
    else:
        print(f"All hybrid datasets created in {output_path}")

if __name__ == '__main__':
    main()