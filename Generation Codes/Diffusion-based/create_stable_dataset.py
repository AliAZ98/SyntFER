import pandas as pd
import os
import gc
import torch
from PIL import ImageOps
from diffusers import DiffusionPipeline

# Load prompt data
data = pd.read_csv("generated_prompts.csv")

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/realistic-vision-v51",
    torch_dtype=torch.float16
).to("cuda")

# Output directory
output_dir = "samples"
os.makedirs(output_dir, exist_ok=True)

# Generation loop
for i, row in data.iterrows():
    prompt = row["prompt"]
    gender = row["gender"]
    age = row["age"]
    race = row["race"]
    emotion = row["emotion"]
    trait = row["identity_trait"].replace(" ", "_")
    
    filename = f"face_{int(row['id']):05d}_{gender}_{age}_{race}_{emotion}_{trait}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=10).images[0]

        image = ImageOps.exif_transpose(image).copy()
        image.save(filepath)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"[!] Error on {filename}: {e}")

    del image
    torch.cuda.empty_cache()
    gc.collect()
