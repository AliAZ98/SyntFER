# This version is the version that detects face with dlib on the process, not taking location from detections.pkl
# This script generates the following dataset:
# 15K FIX-EMOTION NO-DOMAIN-ADAPT CLASS-DIST:RAF-DB 

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

# NO-DOMAIN-ADAPT: notebook_utils
from utils.notebook_utils import GANmut

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

# CLASS-DIST:RAF-DB
rafdb_emotion_class_distribution_size_and_index = { \
        'surprise': [1619,1], \
        'fear': [355,2], \
        'disgust': [877,3], \
        'happy': [5957,4], \
        'sad': [2460, 5], \
        'anger': [867, 6], \
        'neutral': [3204,7] }

# FFHQ Retrieval
ffhq_input_images = [f for f in os.listdir('./../../FFHQ_thumbnails256x256/') if f.endswith('.png')]

for emotion, size_and_index_list in rafdb_emotion_class_distribution_size_and_index.items():
    for image in range(size_and_index_list[0]):
        random_image = random.choice(ffhq_input_images)
        ffhq_input_images.remove(random_image)
        image_path = '../../FFHQ_thumbnails256x256/' + random_image
        try: 
                G.emotion_edit(img_path=image_path, x = fix_emotion_parameter_dict[emotion][0], y = fix_emotion_parameter_dict[emotion][1], save = True, emotion_name=emotion, emotion_index = size_and_index_list[1])
        except Exception as e:
              print(f"Error while generating {emotion} from {os.path.split(image_path)[-1]}")

# Shuffle the class label file
file_path = './../edited_images/list_patition_label.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()
random.shuffle(lines)
with open(file_path, 'w') as file:
    file.writelines(lines)