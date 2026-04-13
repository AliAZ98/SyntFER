from diffusers import DiffusionPipeline,StableDiffusionInstructPix2PixPipeline,StableDiffusionXLInstructPix2PixPipeline
import random
import torch
import PIL
from PIL import ImageOps

pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realistic-vision-v51")
pipe = pipe.to("cuda")

import random

genders = ["male", "female"]
ages = ["child", "young", "middle-aged", "adult", "old"]
races = ["white", "black", "asian", "middle-eastern", "latino"]
emotions = ["surprised", "fearful", "disgusted", "happy", "sad", "angry", "neutral"]
head_poses = [
    "slightly turned to the left",
    "slightly turned to the right",
    "looking slightly to the left",
    "looking slightly to the right",
    "head tilted slightly to the left",
    "head tilted slightly to the right"
]

def get_random_prompt():
    gender = random.choice(genders)
    age = random.choice(ages)
    race = random.choice(races)
    emotion = random.choice(emotions)
    head_pose = random.choice(head_poses)

    segments = [
    f"A portrait of a {age} {gender} with {race} ethnicity, "
    f"showing a natural, unposed {emotion} expression, "
    f"{head_pose}, captured in diverse real-world conditions"
    ]
    prompt = ", ".join(segments)

    return prompt

for i in range(10):
    prompt = get_random_prompt()
    print(f"Prompt {i+1}: {prompt}")
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image = ImageOps.exif_transpose(image)
    image.save(f"samples/random_face_{i+1}.png")
