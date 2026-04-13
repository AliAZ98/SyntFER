# SyntFER72

![Abstract overview / teaser](Figures/overall%20data%20generation.png)


# “On Applicability of Synthetic Datasets for Facial Expression Recognition” (Anonymous submission).
## Abstract
Facial Expression Recognition faces two core challenges. The first is class imbalance in public datasets, which skews the learning process and weakens generalization. The second is related to privacy and data collection constraints, which limit the sharing of facial images and restrict the creation of large, balanced datasets. To address these issues, we examine three complementary strategies for constructing privacy-preserving FER datasets in the standard seven discrete facial expression classes setting. Our strategies are: (i) pseudo-labeling large unlabeled face collections with a teacher model under a confidence-thresholding scheme, (ii) prompt-driven synthesis using diffusion models conditioned on demographic attributes, and (iii) task-aware GAN-based expression editing that modifies facial expression while preserving identity and realism. For training and evaluation, we employed widely adopted datasets, including AffectNet, RAF-DB, and FER2013. We utilized the synthetic datasets DigiFace, DCFace, and Emonet-Face BIG as unlabeled sources for pseudo-labeling. Additionally, we utilized the FFHQ dataset as the source for generative synthesis. The main experiments are conducted using a classic CNN backbone, IR50, and we also explore a more complex architecture, POSTERv1, to assess its feasibility and robustness. Using cross-dataset evaluations, we analyze the trade-offs each strategy presents in curated datasets. The findings demonstrate how synthetic data can effectively substitute or be combined with real datasets to mitigate imbalance and privacy limitations.


---

## 1- Training and Testing

### A) Training on IR50 and POSTERv1

#### A.1) IR50 training (within this repository)
- The IR50 training/evaluation utilities are organized under:
  - `Train-Test IR50/`
- Typical workflow:
  1. Create a Python environment and install dependencies.
  2. Prepare your dataset folder(s) (e.g., RAF-DB and/or curated synthetic datasets).
  3. Run the provided training script(s) under `Train-Test IR50/` (see the folder for configs/scripts).

**Notes**
- Make sure your dataset directory structure matches what the training scripts expect.
- If paths are hard-coded in any config/script, update them to your local setup.

#### A.2) POSTERv1 training (official implementation + our integration)
- POSTER official implementation:
  - https://github.com/zczcwh/POSTER
- Suggested workflow:
  1. Clone the official POSTER repo and set it up following their instructions.
  2. Use the same curated datasets described in Section 2 (or RAF-DB) for training/evaluation.
  3. If you use any adapter code in this repo (e.g., dataset lists, preprocessing, splits), keep the dataset root paths consistent across both repos.

---

### B) Testing notebook
- A testing / evaluation notebook is provided in the repository (see the notebooks under `Train-Test IR50/`).
- Typical usage:
  1. Download or place the trained checkpoint in the expected checkpoints directory.
  2. Open the notebook and set:
     - `checkpoint_path`
     - `dataset_root`
  3. Run all cells to reproduce evaluation metrics and/or plots.

---

## 2- Generated datasets and sources

### a) Original synthetic dataset sources (external)
Download the original datasets from their official sources:

- **DCFace**
  - Official repository: https://github.com/mk-minchul/dcface

- **DigiFace-1M**
  - Official repository: https://github.com/microsoft/DigiFace1M
  - Project page: https://microsoft.github.io/DigiFace1M/

- **EmoNet-Face (BIG)**
  - Official repository: https://github.com/laion-ai/emonet-face
  - HuggingFace dataset (EmoNet-Face BIG): https://huggingface.co/datasets/laion/emonet-face-big

---

### b) Curated datasets (this work)
The curated datasets listed below will be uploaded soon.  
For the three external sources above (DCFace, DigiFace-1M, EmoNet-Face BIG), the “curated download” is not provided here (N/A), since they should be downloaded from their official sources.

| Curated dataset name | Source repo / HF / panel | Link to download curated dataset |
|---|---|---|
| DCFace | Official source (see above) | N/A |
| DigiFace-1M | Official source (see above) | N/A |
| EmoNet-Face BIG | Official source (see above) | N/A |
| StableDiffusion-Curated | This repository | **Will be uploaded soon** |
| FineFace-Curated | This repository | **Will be uploaded soon** |
| FineFaceV2-Curated | This repository | **Will be uploaded soon** |
| GANmut-F (Fixed-intensity) Curated | This repository | **Will be uploaded soon** |
| GANmut-V (Variate-intensity) Curated | This repository | **Will be uploaded soon** |
| Mixed-SYN Curated | This repository | **Will be uploaded soon** |
| Mixed-SYN-C Curated | This repository | **Will be uploaded soon** |
| Mixed-SYN* Curated | This repository | **Will be uploaded soon** |

> **Note:** “Curated” refers to the processed/balanced subsets prepared for FER training under the protocols described in the paper.

---

## Repository map (quick glance)
- `Figures/` — paper figures used in this README
- `Generation Codes/` — synthetic curation / generation utilities
- `Train-Test IR50/` — IR50 training + evaluation + notebooks

---
## References (External Projects Used)

This repository builds on the following publicly available methods/tools (used in our dataset curation pipelines described in the paper).

### GAN-based expression editing
- **GANmut (CVPR 2021)** — expression editing via an interpretable polar control space  
  - Official implementation: https://github.com/stefanodapolito/GANmut  
  - Paper (arXiv): https://arxiv.org/abs/2406.11079

- **CodeFormer (NeurIPS 2022)** — blind face restoration used as a refinement step after GAN-based editing  
  - Official implementation: https://github.com/sczhou/CodeFormer  
  - Paper (arXiv): https://arxiv.org/abs/2206.11253  

### Diffusion-based face generation
- **Stable Diffusion / Latent Diffusion** — text-to-image latent diffusion backbone used for photorealistic portrait generation  
  - Official Stable Diffusion repo (CompVis): https://github.com/CompVis/stable-diffusion  
  - Latent Diffusion paper (arXiv): https://arxiv.org/abs/2112.10752  
  - Diffusers library (Hugging Face): https://github.com/huggingface/diffusers  

  **Checkpoint used in the paper:** Realistic-Vision v5.1  
  - Model page: https://huggingface.co/stablediffusionapi/realistic-vision-v51

- **FineFace** — AU-aware diffusion model for expression-controlled synthesis  
  - Official repository: https://github.com/tvaranka/fineface  
  - Paper (arXiv): https://arxiv.org/abs/2407.20175

> Please follow each project’s license and citation requirements when using these external tools/models.
>
> 
## Contact
For double-blind review, please use the anonymous contact channel specified in the submission system.

## NOTE: The repository will be transferred  to the official account if the paper is accepted.

