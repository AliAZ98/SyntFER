import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
import random
from torchvision import datasets
from torchvision.datasets import DatasetFolder, ImageFolder


class Affectdataset(data.Dataset):
    def __init__(self, root=None, dataidxs=None, train=True, transform=None, basic_aug=False, download=False):
        #self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform

        if self.train:
            self.dataset = datasets.ImageFolder(root='/path/to/train_set', transform=None)
        else:
            self.dataset = datasets.ImageFolder(root='/path/to/val_set', transform=None)
            #/path/to/mix_dataset/mixed_syntface_dataset/val  /path/to/SynFaceResearch/GANmut/linked_datasets/rafdb_testset

        if self.dataidxs is not None:
            self.dataset.imgs = [self.dataset.imgs[i] for i in self.dataidxs]
            self.dataset.samples = [self.dataset.samples[i] for i in self.dataidxs]

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return [x[1] for x in self.dataset.samples]

    def __getitem__(self, idx):
        #print(1)
        image, target = self.dataset[idx]
        image = np.array(image)
        
        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        if self.transform is not None:
            image = self.transform(image)

        return image, target


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)


# from collections import Counter

# # Assuming you have already initialized the Affectdataset with the correct parameters
# affect_dataset = Affectdataset(root=None, train=True)

# # Get the labels of all images
# labels = affect_dataset.get_labels()

# # Use Counter to count the occurrences of each class label
# label_counts = Counter(labels)

# # Print the number of images for each class
# for label, count in label_counts.items():
#     print(f'Class {label}: {count} images')
