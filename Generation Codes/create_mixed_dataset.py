import os
import shutil
import random
from pathlib import Path

# --- Configuration ---
SOURCE_ROOT = Path('/path/to/storage_erdi/syntface_datasets')
OUTPUT_ROOT = Path('./mixed_syntface_dataset')  # Will be created in current directory

DATASETS = [
    "DCFace_balanced_10K",
    "DigiFace_balanced_10K",
    "EmoNet_Face_Big_balanced_10K",
    "fineface_images",
    "finefaceV2_images",
    "GANmut-F",
    "GANmut-V",
    "stable_generated"
]

# Mapping: "Output Folder Name" : ["List of possible source folder names"]
CLASS_MAPPINGS = {
    "angry":    ["angry", "anger"],          
    "disgust":  ["disgust", "disgusted"],    
    "fear":     ["fear", "fearful"],         
    "happy":    ["happy"],
    "neutral":  ["neutral"],
    "sad":      ["sad"],
    "surprise": ["surprise", "surprised"]    
}

SAMPLES_TRAIN = 1250
SAMPLES_VAL = 125
TOTAL_SAMPLES = SAMPLES_TRAIN + SAMPLES_VAL

def get_valid_subfolder(dataset_path, folder_variants):
    """Checks which variant of the class folder exists in the dataset."""
    # FIX: Loop over the argument 'folder_variants', not 'variants'
    for variant in folder_variants:
        p = dataset_path / variant
        if p.exists() and p.is_dir():
            return p
    return None

def create_dataset():
    # 1. Setup Output Directory Structure
    for split in ['train', 'val']:
        for class_name in CLASS_MAPPINGS.keys():
            (OUTPUT_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)
            
    print(f"Created output directory structure at: {OUTPUT_ROOT.resolve()}")
    
    # 2. Iterate over Datasets and Process
    for dataset_name in DATASETS:
        dataset_path = SOURCE_ROOT / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found. Skipping.")
            continue
            
        print(f"\nProcessing Dataset: {dataset_name}...")
        
        for target_class, folder_variants in CLASS_MAPPINGS.items():
            # Find the actual folder name for this class in this dataset
            source_class_dir = get_valid_subfolder(dataset_path, folder_variants)
            
            if source_class_dir is None:
                print(f"  [!] Could not find folder for class '{target_class}' in {dataset_name}. Checked: {folder_variants}")
                continue
                
            # Collect all valid image files
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            all_images = [
                f for f in source_class_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            
            # Check if we have enough data
            if len(all_images) < TOTAL_SAMPLES:
                print(f"  [!] Warning: '{target_class}' has only {len(all_images)} images (Needed {TOTAL_SAMPLES}). Using ALL available.")
                selected_files = all_images
                random.shuffle(selected_files)
                
                # If short on data, prioritize training set
                train_files = selected_files[:SAMPLES_TRAIN] 
                val_files = selected_files[SAMPLES_TRAIN:]
            else:
                # Random sample without replacement
                selected_files = random.sample(all_images, TOTAL_SAMPLES)
                train_files = selected_files[:SAMPLES_TRAIN]
                val_files = selected_files[SAMPLES_TRAIN:]

            # Helper function to copy files
            def copy_files(file_list, split_name):
                dest_dir = OUTPUT_ROOT / split_name / target_class
                for src_file in file_list:
                    # Rename to avoid collisions: "DatasetName_OriginalName.jpg"
                    new_filename = f"{dataset_name}_{src_file.name}"
                    shutil.copy2(src_file, dest_dir / new_filename)

            # Perform Copy
            copy_files(train_files, 'train')
            copy_files(val_files, 'val')
            
            print(f"  -> {target_class}: Copied {len(train_files)} train, {len(val_files)} val.")

    print("\n--- Processing Complete ---")
    print(f"Dataset saved to: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    create_dataset()