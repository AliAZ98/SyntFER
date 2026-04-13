import os
import re
import gc
import json
import argparse
import random
from typing import Dict, List

import torch
from PIL import ImageOps
from fineface import FineFacePipeline
from tqdm import tqdm

# -----------------------
# FineFace-supported AUs and descriptions
# -----------------------
SUPPORTED_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
AU_TO_DESC = {
    "AU1": "inner brow raiser",
    "AU2": "outer brow raiser",
    "AU4": "brow lowerer",
    "AU5": "upper lid raiser",
    "AU6": "cheek raiser",
    "AU9": "nose wrinkler",
    "AU12": "lip corner puller",
    "AU15": "lip corner depressor",
    "AU17": "chin raiser",
    "AU20": "lip stretcher",
    "AU25": "lips parted",
    "AU26": "jaw drop",
}

# -----------------------
# Emotion → AU base intensities (tune as needed)
# -----------------------
EMOTION_TO_AUS: Dict[str, Dict[str, float]] = {
    "happy":      {"AU6": 2.5, "AU12": 4.0, "AU25": 1.5},
    "sad":        {"AU1": 2.5, "AU4": 2.0, "AU15": 3.0, "AU17": 1.5},
    "angry":      {"AU4": 3.5, "AU5": 0.8, "AU17": 1.2, "AU25": 0.3},  # AU7 not available; AU5+AU17 as light proxy
    "disgusted":  {"AU9": 3.5, "AU4": 1.2, "AU17": 0.8, "AU25": 0.8},
    "fearful":    {"AU1": 1.8, "AU2": 1.6, "AU5": 2.5, "AU20": 1.8, "AU25": 1.2, "AU26": 1.2},
    "surprised":  {"AU1": 2.5, "AU2": 2.5, "AU5": 2.8, "AU25": 1.8, "AU26": 2.5},
    "neutral":    {},
}

SUPPORTED_AU_SET = {f"AU{k}" for k in SUPPORTED_AUS}

def sanitize_aus(au_dict: Dict[str, float]) -> Dict[str, float]:
    """Keep only supported AUs and non-zero values."""
    return {k: float(v) for k, v in au_dict.items() if k in SUPPORTED_AU_SET and float(v) != 0.0}

def scale_and_jitter_aus(au_dict: Dict[str, float], scale: float, jitter: float) -> Dict[str, float]:
    """Scale and (optionally) randomly jitter AU intensities."""
    out = {}
    for k, v in au_dict.items():
        val = v * scale
        if jitter > 0:
            val += random.uniform(-jitter, jitter)
        out[k] = max(0.0, val)
    return out

# -----------------------
# Prompt cleaning & AU text appending
# -----------------------
EMOTION_WORDS = r"(happy|sad|angry|surprised|fearful|disgusted|neutral|expression|facial expression|emotion|emotional)"

CLEAN_PATTERNS = [
    # Remove clauses like: "clearly showing ... facial expression (...)."
    r"(?:clearly\s+showing|showing|displaying)\b[^.]*?\bfacial\s+expression[^.]*\.",
    # Remove parentheses bits that look like AU lists inside the expression clause
    r"\([^)]*?(AU\s*\d+|inner|outer|brow|lid|lip|cheek|chin|jaw|eyes|mouth)[^)]*\)",
    # Remove trailing explicit emotion sentences
    r"[^.]*\b(?:{words})\b[^.]*\.".format(words=EMOTION_WORDS),
]

def clean_prompt_text(original: str) -> str:
    """Remove emotion-descriptive clauses; keep subject/context."""
    text = original
    for pat in CLEAN_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # Collapse extra spaces/commas
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s*\.\s*\.\s*", ". ", text)
    text = text.strip()
    # Ensure sentence ends with a period if it looks truncated
    if len(text) and text[-1] not in ".!?":
        text += "."
    return text

def au_dict_to_text(au_dict: Dict[str, float]) -> str:
    """Readable AU list for appending to prompt."""
    if not au_dict:
        return "neutral (no action units applied)"
    parts = []
    for k, v in sorted(au_dict.items()):
        label = AU_TO_DESC.get(k, k)
        parts.append(f"{k} ({label})={v:.2f}")
    return "; ".join(parts)

# -----------------------
# File naming
# -----------------------
def filename_from_flags(idx: str, gender: str, age: str, race: str, emotion: str, trait: str) -> str:
    trait_tag = (trait or "none").replace(" ", "_").replace("'", "")
    return f"face_{idx}_{gender}_{age}_{race}_{emotion}_{trait_tag}.png"

# -----------------------
# Main generation
# -----------------------
def main(
    json_path: str,
    output_dir: str = "fineface_outputs",
    seed: int = 42,
    scale: float = 1.0,
    jitter: float = 0.0,
    overwrite: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    with open(json_path, "r", encoding="utf-8") as f:
        records: List[dict] = json.load(f)

    pipe = FineFacePipeline()

    for rec in tqdm(records):
        idx = rec.get("id", "00000")
        gender = rec.get("gender", "person")
        age = rec.get("age", "")
        race = rec.get("race", "")
        emotion = (rec.get("emotion", "neutral") or "neutral").lower()
        head_pose = rec.get("head_pose", "")
        trait = rec.get("identity_trait", "")

        # 1) Start from the existing prompt, but remove emotion wording.
        base_prompt = rec.get("prompt", "")
        cleaned = clean_prompt_text(base_prompt)

        # 2) Build AUs from emotion label
        base_aus = EMOTION_TO_AUS.get(emotion, {})
        aus = sanitize_aus(scale_and_jitter_aus(base_aus, scale=scale, jitter=jitter))

        # 3) Append readable AU guidance to the end (supporting, not dominating)
        au_text = au_dict_to_text(aus)
        if au_text:
            final_prompt = f"{cleaned} The facial expression should follow these action units: {au_text}."
        else:
            final_prompt = cleaned  # neutral

        # Per-image deterministic seed (helps reproducibility across runs)
        img_seed = seed + hash((idx, gender, age, race, emotion, trait)) % (2**31 - 1)
        torch.manual_seed(img_seed)
        random.seed(img_seed)

        # Generate
        try:
            out = pipe(final_prompt, aus)
            image = out.images[0]
        except Exception as e:
            print(f"[!] Generation error for {idx} ({emotion}): {e}")
            continue

        # Save
        fname = filename_from_flags(idx, gender, age, race, emotion, trait)
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath) and not overwrite:
            print(f"[skip] Exists: {fpath}")
        else:
            try:
                image = ImageOps.exif_transpose(image).copy()
                image.save(fpath)
                print(f"[✓] Saved: {fpath}")
            except Exception as e:
                print(f"[!] Save error {fpath}: {e}")

        # Cleanup
        del image
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate faces with FineFace using emotion AUs and cleaned JSON prompts.")
    parser.add_argument("--json", type=str, default="../stable_fer/generated_prompts.json", help="Path to JSON file.")
    parser.add_argument("--out", type=str, default="fineface_outputsV2", help="Output folder.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0, help="Global AU intensity multiplier.")
    parser.add_argument("--jitter", type=float, default=0.0, help="Random AU jitter per unit in [-jitter,+jitter].")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(
        json_path=args.json,
        output_dir=args.out,
        seed=args.seed,
        scale=args.scale,
        jitter=args.jitter,
        overwrite=args.overwrite,
    )
