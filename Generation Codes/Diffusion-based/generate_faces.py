from diffusers import DiffusionPipeline
import random
import torch
from PIL import ImageOps
import os
import gc

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

# Load pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/realistic-vision-v51",
    torch_dtype=torch.float16
).to("cuda")

# Prompt elements
genders = ["male", "female"]
ages = ["child", "young", "middle-aged", "adult", "old"]
races = ["white", "black", "asian", "middle-eastern", "latino"]
emotions = ["surprised", "fearful", "disgusted", "happy", "sad", "angry", "neutral"]
head_poses = [
    "slightly turned to the left", "slightly turned to the right",
    "looking slightly to the left", "looking slightly to the right",
    "head tilted slightly to the left", "head tilted slightly to the right"
]

# === Emotion cue dictionaries ===

# 1. AU-based (technical)
emotion_cues_au = {
    "happy": "cheek raiser, lip corner puller, lips parted",
    "sad": "inner brow raiser, brow lowerer, lip corner depressor, chin raiser",
    "angry": "brow lowerer, lid tightener, chin raiser",
    "disgusted": "brow lowerer, cheek raiser, lid tightener, nose wrinkler",
    "fearful": "inner brow raiser, outer brow raiser, lip stretcher, upper lid raiser, lips parted",
    "surprised": "inner and outer brow raiser, upper lid raiser, lips parted, jaw drop",
    "neutral": "no specific muscle movements, relaxed face"
}

# 2. Descriptive (simple)
emotion_cues_descriptive = {
    "happy": "wide smile, raised cheeks",
    "sad": "teary eyes, downturned lips",
    "angry": "furrowed brows, clenched jaw",
    "disgusted": "wrinkled nose, upper lip raised",
    "fearful": "wide eyes, tense mouth",
    "surprised": "raised eyebrows, open mouth",
    "neutral": "relaxed facial muscles, calm eyes"
}

# 3. Combined/Intermediate (semi-technical, expressive)
emotion_cues_combined = {
    "happy": "smiling with parted lips and slightly raised cheeks",
    "sad": "subtly drooped mouth corners and raised inner brows",
    "angry": "intense brow tension, tightened eyes and chin pressure",
    "disgusted": "upper lip lifted with nose wrinkling and narrowed eyes",
    "fearful": "eyes wide open, brows lifted, mouth slightly parted",
    "surprised": "eyes and mouth wide open, brows arched upward",
    "neutral": "soft gaze, symmetric facial muscles, no expression signs"
}

cue_sets = {
    "au": emotion_cues_au,
    "descriptive": emotion_cues_descriptive,
    "combined": emotion_cues_combined
}

# Identity traits by (gender, age)
identity_traits_by_group = {
    ("male", "child"): [
        "short straight hair", "messy hair", "side-parted hair", "buzz cut", "spiky hair", "wearing a cap"
    ],
    ("female", "child"): [
        "braided hair", "short bob cut", "ponytail", "shoulder-length hair", "curly hair", "with a headband"
    ],
    ("male", "young"): [
        "buzz cut", "short spiky hair", "clean-shaven", "short beard", "goatee", "wearing a hoodie", "with earrings",
        "undercut hairstyle", "faded haircut", "with a nose ring"
    ],
    ("female", "young"): [
        "long straight hair", "ponytail", "with earrings", "wearing glasses", "fringe hairstyle", "with freckles",
        "shoulder-length waves", "colored highlights", "braided ponytail", "with a headscarf"
    ],
    ("male", "adult"): [
        "clean-shaven", "short beard", "moustache", "buzz cut", "wearing glasses", "medium-length hair",
        "slight stubble", "side-parted haircut", "with a crew cut", "with a five o'clock shadow"
    ],
    ("female", "adult"): [
        "with bangs", "long curly hair", "with earrings", "wearing glasses", "medium bob haircut",
        "with layered hairstyle", "curly shoulder-length hair", "with nose ring", "natural waves", "wearing lipstick"
    ],
    ("male", "middle-aged"): [
        "moustache and beard", "receding hairline", "wearing glasses", "clean-shaven", "short grey hair",
        "salt-and-pepper beard", "thinning hair", "wearing a dress shirt", "comb-over hairstyle", "with forehead wrinkles"
    ],
    ("female", "middle-aged"): [
        "shoulder-length hair", "wearing a scarf", "short curls", "with earrings", "grey streaks in hair",
        "tied-back hair", "glasses and lipstick", "loose bun hairstyle", "wavy layered hair", "light makeup"
    ],
    ("male", "old"): [
        "bald with white beard", "grey hair with glasses", "wrinkled face", "receding hairline", "clean-shaven elderly",
        "deep forehead lines", "white moustache", "aged skin", "sunspots on cheeks", "wearing a sweater"
    ],
    ("female", "old"): [
        "grey bun hairstyle", "short grey curls", "with earrings", "wrinkled face", "wearing a scarf",
        "deep smile lines", "wearing large glasses", "white wavy hair", "wearing a cardigan", "thin lips and age spots"
    ]
}

# Prompt generator
def get_random_prompt():
    gender = random.choice(genders)
    age = random.choice(ages)
    race = random.choice(races)
    emotion = random.choice(emotions)
    head_pose = random.choice(head_poses)

    # Identity
    identity_options = identity_traits_by_group.get((gender, age), ["neutral appearance"])
    identity_trait = random.choice(identity_options)

    # Random cue set
    cue_key = random.choice(["au", "descriptive", "combined"])
    emotion_cue = cue_sets[cue_key][emotion]

    prompt = (
        f"A close-up portrait clearly showing an intense and highly expressive {emotion} facial expression ({emotion_cue}). "
        f"The subject is a {age} {gender} with {race} ethnicity. "
        f"{head_pose}, {identity_trait}, captured in a real-world environment."
    )

    flags = {
        "gender":gender,
        "age":age,
        "race":race,
        "emotion":emotion,
        "identity_trait":identity_trait,
        "emotion_cue":emotion_cue
    }

    return prompt, flags

# Generation
batch_size = 25
os.makedirs("samples", exist_ok=True)

prompt_data = [get_random_prompt() for _ in range(batch_size)]
prompts = [p[0] for p in prompt_data]

# Print prompts
for i, p in enumerate(prompts):
    print(f"Prompt {i+1}: {p}")

# Generate images
with torch.autocast("cuda"):
    output = pipe(prompts, num_inference_steps=50, guidance_scale=10)

# Save images
for i, (img, (_, flags)) in enumerate(zip(output.images, prompt_data)):
    try:
        img = ImageOps.exif_transpose(img).copy()
        gender, age, race, emotion = flags["gender"], flags["age"], flags["race"], flags["emotion"]
        filename = f"samples/face_{i+1:03d}_{gender}_{age}_{race}_{emotion}.png"
        img.save(filename)
    except Exception as e:
        print(f"[!] Failed to save image {i+1}: {e}")

    del img
    torch.cuda.empty_cache()
    gc.collect()
