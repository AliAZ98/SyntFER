# This script generates the following dataset:
# 70K FIX-EMOTION DOMAIN-ADAPT 10K-EACH-EMOTION

import sys
sys.path.append("..")
import random
import os
import numpy as np

# Set random seed for reproducibility
seed = 42
random.seed(seed)  # For Python's built-in random
np.random.seed(seed)  # For NumPy


# CHECK!
# There should be no 'edited_images' folder in 'GANmut' folder or 'edited_images' should be empty.

# DOMAIN-ADAPT: notebook_utils
from utils.notebook_utils4 import GANmut

# FIX-EMOTION: GGANmut (Gaussian)
G = GANmut(G_path='../learned_generators/gaus_2d/1800000-G.ckpt',model='gaussian')
fix_emotion_parameter_dict = { \
        'neutral':[-0.2812, -0.4329], \
        'happy': [ 0.3152, -0.1149], \
        'sad': [-1.0000, -0.6416], \
        'surprise': [ 0.9617, -0.4510], \
        'fear': [-0.1805, -1.0000], \
        'disgust': [-0.9823, -0.9984], \
        'anger' :[ 0.8572, -0.9997]}

# Emotion Indices
emotion_indices = { \
        'surprise': 1, \
        'fear': 2, \
        'disgust': 3, \
        'happy': 4, \
        'sad': 5, \
        'anger': 6, \
        'neutral': 7 }

# FFHQ Retrieval
ffhq_input_images = [f for f in os.listdir('./../../FFHQ_thumbnails256x256/') if f.endswith('.png')]

# Shuffle the ffhq_input_images
random.shuffle(ffhq_input_images)

# Split ffhq_input_images into 7 arrays of 10K images each
emotion_image_arrays = {
    emotion: ffhq_input_images[i * 10000: (i + 1) * 10000]
    for i, emotion in enumerate(emotion_indices.keys())
}

# Generate 10K synthetic images for each emotion
for emotion, image_list in emotion_image_arrays.items():
    emotion_params = fix_emotion_parameter_dict[emotion]
    emotion_index = emotion_indices[emotion]
    
    for image_path in image_list:
        full_image_path = '../../FFHQ_thumbnails256x256/' + image_path
        
        try:
            G.emotion_edit(
                img_path=full_image_path,
                x=emotion_params[0],
                y=emotion_params[1],
                save=True,
                emotion_name=emotion,
                emotion_index=emotion_index
            )

        except Exception as e:
            print(f"Error while generating {emotion} from {os.path.split(full_image_path)[-1]}: {e}")
            print("Error:", e)

# Shuffle the class label file
file_path = './../edited_images/list_patition_label.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
random.shuffle(lines)
with open(file_path, 'w') as file:
    file.writelines(lines)