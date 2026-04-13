# This script generates the following dataset:
# 490K VARIABLE-EMOTION DOMAIN-ADAPT 70K-EACH-EMOTION

import sys
sys.path.append("..")
import random
import os
import numpy as np
import math
from tqdm import tqdm
# Set random seed for reproducibility
seed = 42
random.seed(seed)  # For Python's built-in random
np.random.seed(seed)  # For NumPy

# CHECK!
# There should be no 'edited_images' folder in 'GANmut' folder or 'edited_images' should be empty.

# No-DOMAIN-ADAPT: notebook_utils
from utils.notebook_utils3 import GANmut

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

# Generate 490K synthetic images for each emotion of each image (7*70K)

# For each face image in FFHQ dataset
for image_path in tqdm(ffhq_input_images):

    # For each one of seven (7) emotion
    for emotion in emotion_indices.keys():
        emotion_params = fix_emotion_parameter_dict[emotion]
        # Get index of emotion
        emotion_index = emotion_indices[emotion]

        full_image_path = '../../FFHQ_thumbnails256x256/' + image_path
        
        try:
            # theta_emotion you would need to do atan2(y,x) where x,y are coordinate of the unit vectors you obtain from print_axes(). 
            # rho_emotion is your choice, it depends on the intensity you want.
            
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
