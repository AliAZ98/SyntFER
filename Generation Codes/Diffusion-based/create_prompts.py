import itertools
import pandas as pd
import json

# Prompt attributes
genders = ["male", "female"]
ages = ["child", "young", "middle-aged", "adult", "old"]
races = ["white", "black", "asian", "middle-eastern", "latino"]
emotions = ["surprised", "fearful", "disgusted", "happy", "sad", "angry", "neutral"]
head_poses = [
    "slightly turned to the left", "slightly turned to the right",
    "looking slightly to the left", "looking slightly to the right",
    "head tilted slightly to the left", "head tilted slightly to the right"
]
cue_keys = ["au", "descriptive", "combined"]

# Emotion cue sets
emotion_cues_au = {
    "happy": "cheek raiser, lip corner puller, lips parted",
    "sad": "inner brow raiser, brow lowerer, lip corner depressor, chin raiser",
    "angry": "brow lowerer, lid tightener, chin raiser",
    "disgusted": "brow lowerer, cheek raiser, lid tightener, nose wrinkler",
    "fearful": "inner brow raiser, outer brow raiser, lip stretcher, upper lid raiser, lips parted",
    "surprised": "inner and outer brow raiser, upper lid raiser, lips parted, jaw drop",
    "neutral": "no specific muscle movements, relaxed face"
}

emotion_cues_descriptive = {
    "happy": "wide smile, raised cheeks",
    "sad": "teary eyes, downturned lips",
    "angry": "furrowed brows, clenched jaw",
    "disgusted": "wrinkled nose, upper lip raised",
    "fearful": "wide eyes, tense mouth",
    "surprised": "raised eyebrows, open mouth",
    "neutral": "relaxed facial muscles, calm eyes"
}

emotion_cues_combined = {
    "happy": "smiling with parted lips and slightly raised cheeks",
    "sad": "subtly drooped mouth corners and raised inner brows",
    "angry": "intense brow tension, tightened eyes and chin pressure",
    "disgusted": "upper lip lifted with nose wrinkling and narrowed eyes",
    "fearful": "eyes wide open, brows lifted, mouth slightly parted",
    "surprised": "eyes and mouth wide open, brows arched upward",
    "neutral": "soft gaze, symmetric facial muscles, no expression signs"
}

cue_sets = {
    "au": emotion_cues_au,
    "descriptive": emotion_cues_descriptive,
    "combined": emotion_cues_combined
}

# Identity traits by (gender, age)
identity_traits_by_group = {
    ("male", "child"): [
        "short straight hair", "messy hair", "side-parted hair", "buzz cut", "spiky hair", "wearing a cap"
    ],
    ("female", "child"): [
        "braided hair", "short bob cut", "ponytail", "shoulder-length hair", "curly hair", "with a headband"
    ],
    ("male", "young"): [
        "buzz cut", "short spiky hair", "clean-shaven", "short beard", "goatee", "wearing a hoodie", "with earrings",
        "undercut hairstyle", "faded haircut", "with a nose ring"
    ],
    ("female", "young"): [
        "long straight hair", "ponytail", "with earrings", "wearing glasses", "fringe hairstyle", "with freckles",
        "shoulder-length waves", "colored highlights", "braided ponytail", "with a headscarf"
    ],
    ("male", "adult"): [
        "clean-shaven", "short beard", "moustache", "buzz cut", "wearing glasses", "medium-length hair",
        "slight stubble", "side-parted haircut", "with a crew cut", "with a five o'clock shadow"
    ],
    ("female", "adult"): [
        "with bangs", "long curly hair", "with earrings", "wearing glasses", "medium bob haircut",
        "with layered hairstyle", "curly shoulder-length hair", "with nose ring", "natural waves", "wearing lipstick"
    ],
    ("male", "middle-aged"): [
        "moustache and beard", "receding hairline", "wearing glasses", "clean-shaven", "short grey hair",
        "salt-and-pepper beard", "thinning hair", "wearing a dress shirt", "comb-over hairstyle", "with forehead wrinkles"
    ],
    ("female", "middle-aged"): [
        "shoulder-length hair", "wearing a scarf", "short curls", "with earrings", "grey streaks in hair",
        "tied-back hair", "glasses and lipstick", "loose bun hairstyle", "wavy layered hair", "light makeup"
    ],
    ("male", "old"): [
        "bald with white beard", "grey hair with glasses", "wrinkled face", "receding hairline", "clean-shaven elderly",
        "deep forehead lines", "white moustache", "aged skin", "sunspots on cheeks", "wearing a sweater"
    ],
    ("female", "old"): [
        "grey bun hairstyle", "short grey curls", "with earrings", "wrinkled face", "wearing a scarf",
        "deep smile lines", "wearing large glasses", "white wavy hair", "wearing a cardigan", "thin lips and age spots"
    ]
}

# Generate all prompts
records = []
index = 1

for gender, age, race, emotion, head_pose, cue_key in itertools.product(genders, ages, races, emotions, head_poses, cue_keys):
    emotion_cue = cue_sets[cue_key][emotion]
    identity_traits = identity_traits_by_group.get((gender, age), ["neutral appearance"])
    for identity_trait in identity_traits:
        prompt = (
            f"A close-up portrait clearly showing an intense and highly expressive {emotion} facial expression ({emotion_cue}). "
            f"The subject is a {age} {gender} with {race} ethnicity. "
            f"{head_pose}, {identity_trait}, captured in a real-world environment."
        )
        record = {
            "id": f"{index:05d}",
            "prompt": prompt,
            "gender": gender,
            "age": age,
            "race": race,
            "emotion": emotion,
            "head_pose": head_pose,
            "cue_key": cue_key,
            "identity_trait": identity_trait
        }
        records.append(record)
        index += 1

# Save to CSV and JSON
df = pd.DataFrame(records)
df.to_csv("generated_prompts.csv", index=False)
with open("generated_prompts.json", "w") as f:
    json.dump(records, f, indent=2)

print("Files generated: generated_prompts.csv and generated_prompts.json")
