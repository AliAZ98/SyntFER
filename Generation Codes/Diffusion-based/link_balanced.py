from glob import glob
from tqdm import tqdm
import os
import numpy as np
from shutil import copyfile

emotions = ["surprised", "fearful", "disgusted", "happy", "sad", "angry", "neutral"]
datasets_root = "./"

def load_sorted_keys(scores_file_path, round_decimals=None):
    scores = np.load(scores_file_path, allow_pickle=True)

    sorted_map = {}
    counts = {}
    for emotion in emotions:
        em_scores = scores[emotion].item()  # dict: key -> score
        # sort once
        sorted_keys = sorted(em_scores, key=em_scores.get, reverse=True)
        sorted_map[emotion] = sorted_keys
        counts[emotion] = len(sorted_keys)
        print(f"{emotion}: {counts[emotion]} items")

    min_count = min(counts.values())
    if round_decimals:
        min_count = (min_count // round_decimals) * round_decimals
    return sorted_map, min_count


def make_balanced_link(sorted_map, link_name, max_per_class):
    link_root = os.path.join(datasets_root, link_name)
    os.makedirs(link_root, exist_ok=True)
    for emotion in emotions:
        os.makedirs(os.path.join(link_root, emotion), exist_ok=True)

    for emotion in tqdm(emotions, desc=f"Linking for {link_name}"):
        selected_keys = sorted_map[emotion][:max_per_class]
        emotion_dir = os.path.join(link_root, emotion)

        for key in selected_keys:
            key = key.replace("\\", "/")
            filename_only = os.path.basename(key)
            src_path = os.path.join(datasets_root, key)
            if not os.path.exists(src_path):
                print(f"Could not find image {src_path}. Skipping")
                continue

            link_path = os.path.join(emotion_dir, filename_only)
            if not os.path.exists(link_path):
                copyfile(src_path, link_path)
    print(f"[{link_name}] Done.")


if __name__ == "__main__":
    scores_file_path = "DigiFace_Infer_emotion_scores.npz"
    sorted_map, min_count = load_sorted_keys(scores_file_path, round_decimals=1000)
    print(f"Minimum class size: {min_count}")
    make_balanced_link(sorted_map, "DigiFace_Balanced_10K", max_per_class=10000)
