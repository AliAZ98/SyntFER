import sys
import os
import torch
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from torchvision import transforms

# -------------------
# 1. CONFIGURATION
# -------------------
SYNT_ROOT = "/path/to/storage_erdi/syntface_datasets/"

dataset_dirs = {
    "RafDB": "/path/to/FER/raf_db/align_data/train",
    "DCFace": SYNT_ROOT+"DCFace_balanced_10K",
    "DigiFace": SYNT_ROOT+"DigiFace_balanced_10K",
    "EmoNet_Face_Big": SYNT_ROOT+"EmoNet_Face_Big_balanced_10K",
    "fineface": SYNT_ROOT+"fineface_images",
    "finefaceV2": SYNT_ROOT+"finefaceV2_images",
    "GANmut-F": "/path/to/GANmut_local/linked_datasets/GANmut-F",
    "GANmut-V": "/path/to/GANmut_local/linked_datasets/GANmut-V",
    "Stable Diffusion": "/path/to/storage_erdi/syntface/stable_fer/stable_generated",

}

dataset_dirs = {
    "Stable Diffusion": "/path/to/storage_erdi/syntface/stable_fer/stable_generated",
}


OUT_ROOT = "./facexformer_attributes_results"
FACEX_REPO_ID = "kartiknarayan/facexformer"
FACEX_LOCAL_DIR = "./facexformer"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add FaceXFormer to path
sys.path.append(os.path.abspath(FACEX_LOCAL_DIR))

# -------------------
# 2. CLASS MAPPINGS
# -------------------
AGE_LABELS = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
GENDER_LABELS = ['Male', 'Female']
RACE_LABELS = ['White', 'Black', 'Indian', 'Asian', 'Others']

# -------------------
# 3. MODEL SETUP
# -------------------
print(f"Loading FaceXFormer on {DEVICE}...")

try:
    from network.models.facexformer import FaceXFormer
except ImportError:
    print("Error: 'facexformer' folder not found. Please clone the repo.")
    sys.exit(1)

def get_facexformer():
    # Ensure weights exist
    weights_path = "facexformer/ckpts/model.pt"
    if not os.path.exists(weights_path):
         print("Downloading weights...")
         hf_hub_download(repo_id=FACEX_REPO_ID, filename="ckpts/model.pt", local_dir=FACEX_LOCAL_DIR)

    from network.models.facexformer import FaceXFormer
    model = FaceXFormer().to(DEVICE)
    
    # Simple backbone load
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict_backbone'], strict=False)
    
    # Load heads if available (important for non-backbone layers)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    model.eval()
    return model

model = get_facexformer()

preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------
# 4. HELPER FUNCTIONS
# -------------------

# --- FIX: Robust symlink resolution with prefix remap (no filesystem changes) ---
SYMLINK_REMAP = [
    ("/path/to/", "/storage/omer_lisans/"),
]

def resolve_symlink_with_remap(path: str):
    # If it's a normal existing file, use it
    if os.path.exists(path):
        return path

    # If it's a symlink (possibly broken), try to read its target and remap
    if os.path.islink(path):
        try:
            target = os.readlink(path)

            # Handle relative symlinks
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(os.path.dirname(path), target))

            # If target exists as-is
            if os.path.exists(target):
                return target

            # Try remapping prefixes
            for old_prefix, new_prefix in SYMLINK_REMAP:
                if target.startswith(old_prefix):
                    candidate = new_prefix + target[len(old_prefix):]
                    if os.path.exists(candidate):
                        return candidate
        except OSError:
            pass

    # Last resort: realpath (may still be broken)
    rp = os.path.realpath(path)
    return rp if os.path.exists(rp) else None
# ------------------------------------------------------------------------------

def load_image_safe(path):
    try:
        fixed_path = resolve_symlink_with_remap(path)
        if fixed_path is None:
            return None
        return Image.open(fixed_path).convert("RGB")
    except Exception:
        return None

def prepare_dummy_labels(batch_size):
    return {
        "segmentation": torch.zeros([batch_size, 224, 224]),
        "lnm_seg": torch.zeros([batch_size, 5, 2]),
        "landmark": torch.zeros([batch_size, 68, 2]),
        "headpose": torch.zeros([batch_size, 3]),
        "attribute": torch.zeros([batch_size, 40]),
        "a_g_e": torch.zeros([batch_size, 3]),
        "visibility": torch.zeros([batch_size, 29])
    }

@torch.inference_mode()
def run_inference_batch(pil_images, image_paths):
    if not pil_images: return []

    input_tensor = torch.stack([preprocess(img) for img in pil_images]).to(DEVICE)
    bs = input_tensor.shape[0]

    task = torch.full((bs,), 4, dtype=torch.long).to(DEVICE) 
    labels = prepare_dummy_labels(bs)
    labels = {k: v.to(DEVICE) for k, v in labels.items()}

    results = model(input_tensor, labels, task)
    
    age_logits = results[4].cpu().numpy()
    gender_logits = results[5].cpu().numpy()
    race_logits = results[6].cpu().numpy()

    batch_results = []
    
    for i, path in enumerate(image_paths):
        parent_dir = os.path.dirname(path)
        folder_name = os.path.basename(parent_dir)
        
        pred_age = AGE_LABELS[np.argmax(age_logits[i])]
        pred_gender = GENDER_LABELS[np.argmax(gender_logits[i])]
        pred_race = RACE_LABELS[np.argmax(race_logits[i])]

        batch_results.append({
            "filename": os.path.basename(path),
            "class_folder": folder_name,
            "pred_age": pred_age,
            "pred_gender": pred_gender,
            "pred_race": pred_race,
            "logit_age": str(list(age_logits[i])),
            "logit_gender": str(list(gender_logits[i])),
            "logit_race": str(list(race_logits[i])),
            "full_path": path
        })
        
    return batch_results

# -------------------
# 5. MAIN LOOP (PER CLASS)
# -------------------
os.makedirs(OUT_ROOT, exist_ok=True)
csv_headers = ["filename", "pred_age", "pred_gender", "pred_race", 
               "logit_age", "logit_gender", "logit_race", "full_path"]

for dname, dpath in dataset_dirs.items():
    if not os.path.exists(dpath):
        print(f"Skipping {dname} (Path not found)")
        continue

    out_dir = os.path.join(OUT_ROOT, dname)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nScanning Dataset: {dname}")
    
    try:
        subitems = sorted(os.listdir(dpath))
        class_folders = [c for c in subitems if os.path.isdir(os.path.join(dpath, c))]
    except OSError:
        class_folders = []

    if not class_folders:
        print(f" -> No class folders found in {dpath}. Treating as single class 'root'.")
        class_folders = ["."]

    for class_name in class_folders:
        class_path = os.path.join(dpath, class_name)
        
        safe_cls = "".join([c if c.isalnum() else "_" for c in class_name])
        if safe_cls == ".": safe_cls = "root"
        csv_path = os.path.join(out_dir, f"{safe_cls}_results.csv")

        if os.path.exists(csv_path):
            print(f" [Skip] {class_name} - CSV exists.")
            continue

        # --- FIX: Robust Linked Image Gathering ---
        img_paths = []
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        # followlinks=True is critical for GANmut symlinked folders
        for root, dirs, files in os.walk(class_path, followlinks=True):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    img_paths.append(os.path.join(root, f))
        
        img_paths = sorted(img_paths)
        # ------------------------------------------

        if not img_paths:
            print(f" [WARN] No images found in {class_name} (checked recursively)")
            continue

        print(f" -> Processing Class: '{class_name}' ({len(img_paths)} images)")

        class_results = []
        for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc=f"   Infering {class_name}", leave=False):
            batch_paths = img_paths[i : i + BATCH_SIZE]
            batch_imgs = []
            valid_paths = []
            
            for p in batch_paths:
                img = load_image_safe(p)
                if img is not None:
                    batch_imgs.append(img)
                    valid_paths.append(p)
            
            if batch_imgs:
                results = run_inference_batch(batch_imgs, valid_paths)
                class_results.extend(results)

        if class_results:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(class_results)
            print(f"    -> Saved {csv_path}")
        
        del class_results
        del img_paths

print("\nAll processing complete.")
