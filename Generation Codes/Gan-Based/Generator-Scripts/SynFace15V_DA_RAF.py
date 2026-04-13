# This script generates the following dataset:
# 15K VARIABLE-EMOTION DOMAIN-ADAPT CLASS-DIST:RAF-DB 

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

# CLASS-DIST:RAF-DB
rafdb_emotion_class_distribution_size_and_index = { \
        'surprise': [1619,1], \
        'fear': [355,2], \
        'disgust': [877,3], \
        'happy': [5957,4], \
        'sad': [2460, 5], \
        'anger': [867, 6], \
        'neutral': [3204,7] }

# FFHQ Retrieval of Input Images
ffhq_input_images = [f for f in os.listdir('./../../FFHQ_thumbnails256x256/') if f.endswith('.png')]

# Iterate through class distribution dictionary and get emotion size and index
for emotion, size_and_index_list in rafdb_emotion_class_distribution_size_and_index.items():

    # Iterate as much as emotion size
    for image in range(size_and_index_list[0]):

        # Choose pseudo-random input image
        random_image = random.choice(ffhq_input_images)

        # Remove the input image to avoid reuse of same face
        ffhq_input_images.remove(random_image)

        # Construct path of that image
        image_path = '../../FFHQ_thumbnails256x256/' + random_image

        try:
            # theta_emotion you would need to do atan2(y,x) where x,y are coordinate of the unit vectors you obtain from print_axes(). 
            # rho_emotion is your choice, it depends on the intensity you want.

            # For neutral
            if emotion == 'neutral':
                G.emotion_edit(img_path=image_path, theta=random.uniform(-np.pi, np.pi) ,rho =random.uniform(0, 0.10), save = True, emotion_name=emotion, emotion_index = size_and_index_list[1])

            # For other emotions
            else:
                G.emotion_edit(img_path=image_path, theta=math.atan2(latent_space_mean_vectors[emotion][1],latent_space_mean_vectors[emotion][0]) ,rho =random.uniform(0.50, 1.20), save = True, emotion_name=emotion, emotion_index = size_and_index_list[1])

        # Print to log file if any error occurs
        except Exception as e:
            print(f"Error while generating {emotion} from {os.path.split(image_path)[-1]}")

# Shuffle the class label file
file_path = './../edited_images/list_patition_label.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
random.shuffle(lines)
with open(file_path, 'w') as file:
    file.writelines(lines)