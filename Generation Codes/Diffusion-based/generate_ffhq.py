from diffusers import StableDiffusionXLInstructPix2PixPipeline
import random
import torch
from PIL import Image, ImageOps
import os
import gc
from glob import glob
from tqdm import tqdm

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

# Load the InstructPix2Pix pipeline
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    "diffusers/sdxl-instructpix2pix-768", torch_dtype=torch.float16
).to("cuda")

# Emotions and facial cues
emotion_cues = {
    "happy": "wide smile, raised cheeks, slight eye crinkle",
    "sad": "downturned mouth corners, drooped eyelids, inner brow raiser",
    "angry": "furrowed brows, tightened lips, lowered jaw",
    "surprised": "raised eyebrows, wide-open eyes, parted lips, jaw drop",
    "disgusted": "wrinkled nose, raised upper lip, narrowed eyes",
    "fearful": "wide eyes, tense mouth, slightly raised eyebrows",
    "neutral": "relaxed facial muscles, calm eyes, no strong expression"
}

# Prepare images
images = glob("ffhq_256/*.png")
sorted(images)
print(f"Found {len(images)} images in ffhq_images directory.")

# Output directory
output_dir = "edited_ffhq"
os.makedirs(output_dir, exist_ok=True)

# Loop through all images and all emotions
for img_path in tqdm(images[:3], desc="Images"):
    img = Image.open(img_path).convert("RGB").resize((768, 768))
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for emotion, cues in emotion_cues.items():
        """
        edit_instruction = (
            f"Modify the person's facial expression to look {emotion} without changing the identity or the rest of the scene. "
            f"Keep every distinctive feature exactly the same (skin tone, hairstyle, facial hair, glasses, head pose), "
            f"and maintain realistic lighting, shading, and background. "
            f"Focus only on adjusting the facial muscles: {cues}. "
            #f"Result should be a high-quality, photorealistic image of the same individual showing {emotion}."
        )
        """
        edit_instruction = (
            f"Modify the person's facial expression to look {emotion} keeping the identity related features. "
            f"Focus mainly on adjusting the facial muscles: {cues}."
        )

        # Edit the image
        with torch.autocast("cuda"):
            edited = pipe(
                prompt=edit_instruction,
                image=img,
                height=768,
                width=768,
                guidance_scale=5.0,
                image_guidance_scale=1.0,
                num_inference_steps=50,
            ).images[0]

        # Save the edited image
        filename = f"{base_name}_{emotion}.png"
        edited = ImageOps.exif_transpose(edited).copy()
        edited.save(os.path.join(output_dir, filename))

        # Free memory
        del edited
        torch.cuda.empty_cache()
        gc.collect()
