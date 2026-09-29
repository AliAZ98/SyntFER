# Hugging Face release plan for SyntFER curated datasets

This repository is prepared to publish the curated SyntFER datasets on the Hugging Face Hub.

## Recommended Hub layout

Use one dataset repository:

```text
<HF_USERNAME>/SyntFER-Curated
```

with one Hugging Face configuration (subset) per curated dataset:

- `StableDiffusion-Curated`
- `FineFace-Curated`
- `FineFaceV2-Curated`
- `GANmut-F-Curated`
- `GANmut-V-Curated`
- `Mixed-SYN-Curated`
- `Mixed-SYN-C-Curated`
- `Mixed-SYN-Star-Curated`

This gives users one canonical dataset page while still allowing each variant to be loaded independently.

Example:

```python
from datasets import load_dataset

dataset = load_dataset(
    "<HF_USERNAME>/SyntFER-Curated",
    "StableDiffusion-Curated",
)
```

## Expected local folder structure

Each curated variant should use the same seven FER class names:

```text
StableDiffusion-Curated/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

The same class layout should be used for every other curated variant.

If a dataset has explicit splits, use:

```text
FineFace-Curated/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── ...
```

## Upload

1. Create a Hugging Face account.
2. Create a write token in Hugging Face settings.
3. Install the required packages:

```bash
pip install -U datasets huggingface_hub pillow
```

4. Authenticate:

```bash
hf auth login
```

5. Run the helper script for each local curated dataset folder:

```bash
python tools/upload_syntfer_to_hf.py \
  --dataset-root "/path/to/StableDiffusion-Curated" \
  --repo-id "<HF_USERNAME>/SyntFER-Curated" \
  --config-name "StableDiffusion-Curated"
```

Repeat for the other configurations.

## Before publication

Please verify for every curated dataset:

- image count per expression class;
- exact source/generation method;
- whether images are fully synthetic or derived/edited from another source;
- any upstream license or redistribution restriction;
- whether the release may be public or should be gated;
- paper citation and authors;
- intended use and limitations;
- demographic attribute handling, if applicable.

The repository MIT license covers the code in SyntFER. Dataset redistribution rights should be documented separately because source/model/data licenses may impose different conditions.

## Suggested dataset card contents

The Hugging Face dataset card should document:

- paper: *On Applicability of Synthetic Datasets for Facial Expression Recognition*;
- arXiv: 2605.17483;
- seven labels: angry, disgust, fear, happy, neutral, sad, surprise;
- construction protocol for each configuration;
- class counts;
- source model/dataset dependencies;
- limitations and responsible-use notes;
- citation;
- GitHub project link.

After the Hub repository is live, replace the Google Drive-only entries in the main README with direct Hugging Face links and `load_dataset` examples.
