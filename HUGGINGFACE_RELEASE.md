# Hugging Face release plan for SyntFER curated datasets

This repository is prepared to publish the curated SyntFER datasets on the Hugging Face Hub.

## Recommended Hub layout

Use one dataset repository:

```text
AliAz98/SyntFER-Curated
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
    "AliAz98/SyntFER-Curated",
    "StableDiffusion-Curated",
)
```

## Source files and verification status

The intended Hub destination is `AliAz98/SyntFER-Curated`. This document does not imply that the Hub repository has been created or populated.

The project README currently links all eight curated variants to this shared source folder:

[Download source datasets from Google Drive](https://drive.google.com/drive/folders/1L1HW3rY7l398RMKeZp4e9eV-uZqvoS_j)

Individual archive download links have not yet been verified. Do not treat the shared folder URL as a direct archive download.

Ali Azmoudeh confirms that these datasets were curated by him and his colleague and that all released images are synthetic. The source models and any source-image editing steps should still be described for each variant.

As of 2026-09-29, the repository links and release helper have been inspected, but the Drive files have not been accessed. Therefore the following remain **unverified**:

- actual archive names and their mapping to the eight configurations;
- folder structure and split names;
- image counts for each expression class and split;
- image readability and duplicate files;
- correspondence between the downloadable files and the paper's released variants.

The layouts below are expected layouts, not observations from the downloaded data. Compute the counts from the actual files before including them in a dataset card.

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

1. Use the Hugging Face account `AliAz98`.
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
  --repo-id "AliAz98/SyntFER-Curated" \
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
