import os
import gc
import json
import torch
from PIL import ImageOps
from fineface import FineFacePipeline
from tqdm import tqdm

# -----------------------
# AU mappings for emotions
# (only supported AUs: 1,2,4,5,6,9,12,15,17,20,25,26)
# -----------------------
EMOTION_TO_AUS = {
    "happy": {"AU6": 2.5, "AU12": 4.0, "AU25": 1.5},
    "sad": {"AU1": 2.5, "AU4": 2.0, "AU15": 3.0, "AU17": 1.5},
    "angry": {"AU4": 3.5, "AU5": 0.8, "AU17": 1.2, "AU25": 0.3},
    "disgusted": {"AU9": 3.5, "AU4": 1.2, "AU17": 0.8, "AU25": 0.8},
    "fearful": {"AU1": 1.8, "AU2": 1.6, "AU5": 2.5, "AU20": 1.8, "AU25": 1.2, "AU26": 1.2},
    "surprised": {"AU1": 2.5, "AU2": 2.5, "AU5": 2.8, "AU25": 1.8, "AU26": 2.5},
    "neutral": {}  # no AUs for neutral
}

SUPPORTED_AUS = {f"AU{k}" for k in [1,2,4,5,6,9,12,15,17,20,25,26]}

def sanitize_aus(au_dict):
    return {k: float(v) for k, v in au_dict.items() if k in SUPPORTED_AUS and v != 0.0}

# -----------------------
# Main
# -----------------------
def main(json_file, output_dir="fineface_outputs", seed=42, scale=1.0):
    os.makedirs(output_dir, exist_ok=True)

    # Load FineFace
    pipe = FineFacePipeline()
    torch.manual_seed(seed)

    # Load JSON data
    with open(json_file, "r") as f:
        records = json.load(f)

    for rec in tqdm(records):
        idx = rec.get("id", "00000")
        gender = rec.get("gender", "person")
        age = rec.get("age", "")
        race = rec.get("race", "")
        emotion = rec.get("emotion", "neutral").lower()
        identity_trait = rec.get("identity_trait", "").replace(" ", "_")
        
        # Build neutral subject prompt (no emotion words)
        prompt = (
            f"A close-up portrait of a {age} {gender} with {race} ethnicity, "
            f"{identity_trait}, {rec.get('head_pose','')}, photorealistic, natural lighting"
        )

        # Get AU configuration
        base_aus = EMOTION_TO_AUS.get(emotion, {})
        aus = sanitize_aus({k: v * scale for k, v in base_aus.items()})

        if not aus:
            print(f"[!] No AUs for emotion '{emotion}', defaulting to neutral.")

        # Generate
        try:
            out = pipe(prompt, aus)
            image = out.images[0]
        except Exception as e:
            print(f"[!] Generation error for {idx} ({emotion}): {e}")
            continue

        # Save
        filename = f"face_{idx}_{gender}_{age}_{race}_{emotion}_{identity_trait}.png"
        filepath = os.path.join(output_dir, filename)
        image = ImageOps.exif_transpose(image).copy()
        image.save(filepath)
        print(f"[✓] Saved: {filepath}  | AUs: {aus}")

        # Cleanup
        del image
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main("../stable_fer/generated_prompts.json")
