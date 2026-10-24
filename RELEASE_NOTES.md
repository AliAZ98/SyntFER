# SyntFER v1.0.0 — Initial Research Code Release

This is the first public GitHub release of the research code accompanying:

**On Applicability of Synthetic Datasets for Facial Expression Recognition**
Ali Azmoudeh, Erdi Sarıtaş, Ömer Yıldırım, Hazım Kemal Ekenel

Paper: https://arxiv.org/abs/2605.17483

## Included in the source archive

- **Generation Codes/**: utilities for synthetic facial-expression generation and dataset curation, including diffusion-based and GAN-based approaches, hybrid/mixed dataset preparation, restoration, and quality analysis.
- **Train-Test IR50/**: IR50 facial expression recognition training, preprocessing, model utilities, environment requirements, and evaluation notebook.
- **Figures/**: research overview visualization.
- **SupplementaryMaterials.pdf**: supplementary materials accompanying the research.
- **README.md**: method overview, dataset-source links, usage guidance, and citation.

## What this release does not bundle

- Large curated image datasets and third-party source datasets. See the README for external dataset sources and curated dataset access links, subject to the respective usage terms.
- Pretrained model weights and checkpoints that are not already committed.
- The standalone POSTERv1 upstream training repository; follow the official project linked in the README.

## Notes

This release snapshots the code on `main` at the time of publication. Adjust hard-coded dataset and checkpoint paths and install the dependencies needed by the relevant pipeline before running the scripts. Different generation pipelines rely on upstream model implementations; follow their licenses and installation instructions.

Please cite the paper if you use the repository or curated data:

```bibtex
@misc{azmoudeh2026applicabilitysyntheticdatasetsfacial,
  title={On Applicability of Synthetic Datasets for Facial Expression Recognition},
  author={Ali Azmoudeh and Erdi Sarıtaş and Ömer Yıldırım and Hazım Kemal Ekenel},
  year={2026},
  eprint={2605.17483},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.17483}
}
```
