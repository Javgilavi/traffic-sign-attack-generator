# Traffic-Sign Adversarial Dataset Generator

This project generates literature-inspired shadow and occlusion variants for traffic-sign recognition experiments on GTSRB. It is designed for your team workflow: you create attacked images and a manifest, and your teammates evaluate their model on the generated dataset.

The implementation is grounded in the papers already in this folder:

- `2203.03818v3.pdf`: shadow-based physical attack using triangular shadows and LAB lightness attenuation.
- `2307.08278v1.pdf`: survey context for traffic-sign attacks, including shadows, stickers, and physical robustness.
- `2512.00765v2.pdf`: edge-aligned patch idea for stealthier boundary occlusion.

## What This Project Does

- Loads GTSRB from either the Kaggle-style flat CSV layout or the official benchmark folder layout.
- Generates artificial shadows using a ShadowAttack-style triangular mask applied in LAB space.
- Generates occlusions using three physical-patch styles:
  - `edge_patch`: annular boundary mask inspired by edge-patch literature.
  - `center_sticker`: localized sticker-like occlusion.
  - `diagonal_band`: tape-like occlusion across the sign.
- Applies optional EOT-like post-transforms: small rotation, brightness, contrast, and blur.
- Exports all generated images plus a `manifest.csv` with exact attack parameters.

## Important Scope Note

This project creates physically plausible adversarial evaluation data. Because your teammates' model is not part of this repository, the generator does not optimize perturbations against a target classifier. That means the images are literature-informed attack candidates, not guaranteed misclassifications for every model. Once the classifier exists, the same pipeline can be extended with model-aware optimization.

## Project Layout

```text
configs/                         Example YAML configuration
docs/                            Methodology and team usage notes
traffic_sign_attacks/            Generator package
tests/                           Synthetic end-to-end tests
2203.03818v3.pdf                 Shadow attack paper
2307.08278v1.pdf                 Survey paper
2512.00765v2.pdf                 Edge patch paper
```

## Installation

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

The generator supports both common GTSRB layouts:

1. Kaggle-style flat layout

```text
GTSRB/
  Train.csv
  Test.csv
  Train/ or train/
  Test/ or test/
```

2. Official benchmark layout

```text
GTSRB/
  Train/
    00000/
      GT-00000.csv
      *.ppm
  GT-final_test.csv
  Final_Test/Images/
```

Edit [configs/gtsrb_attack_dataset.yaml](/home/javigil/MSc_Robotics/1_year/CV/adversarial-attack/configs/gtsrb_attack_dataset.yaml) and set `dataset_root` to your local GTSRB path.

## Run

```bash
python3 -m traffic_sign_attacks --config configs/gtsrb_attack_dataset.yaml
```

The command writes:

- attacked images under `generated/<split>/<attack>/<class_id>/`
- metadata in `generated/manifest.csv`

## Output Manifest

Each row in `manifest.csv` includes:

- original image path
- generated image path
- class id and label
- sign shape used to build the mask
- ROI coordinates
- serialized attack parameters
- serialized transform parameters

This makes it easy for your teammates to evaluate clean vs attacked accuracy and to reproduce any specific sample.

## Recommended Team Workflow

1. Generate `test` split attacks first.
2. Evaluate the recognition model on the clean test set.
3. Evaluate on shadow-only data.
4. Evaluate on occlusion-only data.
5. Break down results by class and by attack style using `manifest.csv`.
6. If you need training-time robustness experiments later, generate a smaller attacked `train` subset and mix it into training.

## Method Choices From The Papers

- Shadow masks are triangular because the shadow paper reports triangles are already effective while staying natural.
- Shadow intensity is applied only to LAB lightness, following the paper's physical modeling choice.
- Default shadow coefficient range is centered around the paper's reported realistic mean near `k = 0.43`.
- The edge occlusion uses a ring mask because the edge-patch paper argues border-conforming patches are stealthier than central patches.
- Post-transforms approximate EOT-style robustness with small viewpoint and lighting changes.

More detail is in [docs/methodology.md](/home/javigil/MSc_Robotics/1_year/CV/adversarial-attack/docs/methodology.md) and [docs/team_guideline.md](/home/javigil/MSc_Robotics/1_year/CV/adversarial-attack/docs/team_guideline.md).

## Tests

Run the synthetic end-to-end test suite with:

```bash
python3 -m unittest discover -s tests -v
```
