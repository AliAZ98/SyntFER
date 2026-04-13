# This script generates the following dataset:
# 490K VARIABLE-EMOTION DOMAIN-ADAPT 70K-EACH-EMOTION

import sys
sys.path.append("..")
import random
import os
import numpy as np
import math

# Set random seed for reproducibility
seed = 42
random.seed(seed)  # For Python's built-in random
np.random.seed(seed)  # For NumPy

# CHECK!
# There should be no 'edited_images' folder in 'GANmut' folder or 'edited_images' should be empty.

# DOMAIN-ADAPT: notebook_utils
from utils.notebook_utils4 import GANmut

# RANDOM-EMOTION: GANmut (Linear)
G = GANmut(G_path='../learned_generators/lin_2d/1000000-G.ckpt',model='linear')
# latent_space_mean_vectors dictionary holds the mean vectors in latent space corresponding standard AffectNet emotions.
latent_space_mean_vectors = {'happy':[ 0.9918,  0.1275], \
                             'sad':[-0.9996, -0.0287], \
                             'surprise':[-0.1363,  0.9907], \
                             'fear':[-0.6859,  0.7277], \
                             'disgust':[-0.7135, -0.7007], \
                             'anger':[-0.2205, -0.9754]}

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
for image_path in ffhq_input_images:

    # For each one of seven (7) emotion
    for emotion in emotion_indices.keys():

        # Get index of emotion
        emotion_index = emotion_indices[emotion]

        full_image_path = '../../FFHQ_thumbnails256x256/' + image_path
        
        try:
            # theta_emotion you would need to do atan2(y,x) where x,y are coordinate of the unit vectors you obtain from print_axes(). 
            # rho_emotion is your choice, it depends on the intensity you want.

            # For neutral
            if emotion == 'neutral':
                G.emotion_edit(img_path=full_image_path, theta=random.uniform(-np.pi, np.pi) ,rho =random.uniform(0, 0.10), save = True, emotion_name=emotion, emotion_index = emotion_index)

            # For other emotions
            else:

                # Get emotion vector
                emotion_vector = latent_space_mean_vectors[emotion]

                G.emotion_edit(img_path=full_image_path, theta=math.atan2(emotion_vector[1], emotion_vector[0]) ,rho =random.uniform(0.50, 1.20), save = True, emotion_name=emotion, emotion_index = emotion_index)

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