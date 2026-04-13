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

# Count total variation
num_variations = len(genders) * len(ages) * len(races) * len(emotions)
print(f"Total possible prompt variations: {num_variations} *", len(head_poses))

# Emotion-specific facial cues
emotion_cues = {
    "happy": "cheek raiser, lip corner puller, lips parted",                      # AUs 6+7+12+25
    "sad": "inner brow raiser, brow lowerer, lip corner depressor, chin raiser",  # AUs 1+4+15+17
    "angry": "brow lowerer, lid tightener, chin raiser (+ sometimes lip tightener)",  # AUs 4+7+17, alt: 4+5+7+23
    "disgusted": "brow lowerer, cheek raiser, lid tightener, nose wrinkler",      # AUs 4+6+7+9 (alt: 7+9+10+17)
    "fearful": "inner brow raiser, outer brow raiser, lip stretcher, upper lid raiser, lips parted",  # AUs 1+2+20+25 (alt: 4+5+20)
    "surprised": "inner and outer brow raiser, upper lid raiser, lips parted, jaw drop",  # AUs 1+2+5+25+26
    "neutral": "no specific muscle movements, relaxed face"                       # default
}


# Prompt generator with metadata return
def get_random_prompt():
    gender = random.choice(genders)
    age = random.choice(ages)
    race = random.choice(races)
    emotion = random.choice(emotions)
    head_pose = random.choice(head_poses)
    facial_cue = emotion_cues[emotion]

    prompt = (
        f"A portrait of a {age} {gender} with {race} ethnicity, "
        f"showing a vivid, clearly recognizable {emotion} facial expression "
        f"({facial_cue}), {head_pose}, captured in diverse real-world conditions"
    )
    return prompt, gender, age, race, emotion

# Generate prompts
batch_size = 25
prompt_data = [get_random_prompt() for _ in range(batch_size)]

prompts = [p[0] for p in prompt_data]
for i, p in enumerate(prompts):
    print(f"Prompt {i+1}: {p}")

# Set random seed for reproducibility
random_seed = 43
random.seed(random_seed)
torch.manual_seed(random_seed)

# Run batched generation
with torch.autocast("cuda"):
    output = pipe(prompts, num_inference_steps=50, guidance_scale=7.5)

# Save all images
os.makedirs("samples3", exist_ok=True)
for i, (img, (_, gender, age, race, emotion)) in enumerate(zip(output.images, prompt_data)):
    try:
        img = ImageOps.exif_transpose(img).copy()
        filename = f"samples3/face_{i+1:03d}_{gender}_{age}_{race}_{emotion}.png"
        img.save(filename)
    except Exception as e:
        print(f"[!] Failed to save image {i+1}: {e}")

    # Memory cleanup
    del img
    torch.cuda.empty_cache()
    gc.collect()
