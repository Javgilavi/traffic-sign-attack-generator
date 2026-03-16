# Traffic Sign Attack Generator

Generate physical adversarial attack images for traffic sign datasets using shadows and occlusions.

The project takes traffic sign images plus a CSV manifest, applies configurable attack variants, and writes attacked images together with a `manifest.csv` that records the parameters used for every output.

## Example Results

| Shadow | Occlusion |
| --- | --- |
| ![Shadow attack example](example_result/shadow_stop.png) | ![Occlusion attack example](example_result/occlusion_stop.png) |
| ![Shadow attack example](example_result/shadow_yield.png) | ![Occlusion attack example](example_result/occlusion_yield.png) |

## What This Project Does

- Applies triangular shadow attacks inside the traffic sign region.
- Applies physical occlusion attacks with `edge_patch`, `center_sticker`, and `diagonal_band` styles.
- Optionally adds small physical transforms such as rotation, brightness, contrast, and blur.
- Exports attacked images and a manifest with attack parameters and ROI metadata.

This project generates attack images. It does not train a model or search for model-specific perturbations.

## Installation

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

The generator works with any traffic sign dataset that can be described by:

- a `dataset_root` directory
- a CSV manifest referenced by `annotation_file`
- image paths relative to `dataset_root` or absolute paths

Example layout:

```text
my_dataset/
  annotations.csv
  images/
    00001.png
    00002.png
```

Canonical CSV columns:

- Required: `path`, `class_id`
- Optional: `split`, `label`, `shape`, `width`, `height`, `roi_x1`, `roi_y1`, `roi_x2`, `roi_y2`, `sample_id`

Important behavior:

- If `width` and `height` are missing, the image size is read from disk.
- If ROI columns are missing, the full image is used as the attack region.
- If `shape` is missing, the generator uses `shape_map` from the YAML config when provided, otherwise it falls back to `box`.
- Supported shape names are `circle`, `triangle`, `triangle_inverted`, `diamond`, `octagon`, and `box`.
- If your dataset contains full-scene images, you should provide ROI coordinates so the attack stays on the sign.

Example row:

```csv
path,class_id,label,shape,split,roi_x1,roi_y1,roi_x2,roi_y2
images/00001.png,14,Stop,octagon,test,0,0,57,69
```

GTSRB-style manifests still work, but the project is no longer tied to GTSRB specifically.

## Configuration

Use the example config in [`configs/traffic_sign_attack_dataset.yaml`](configs/traffic_sign_attack_dataset.yaml).

Key fields:

- `dataset_root`: root directory of your dataset
- `annotation_file`: CSV manifest path relative to `dataset_root` or absolute
- `output_root`: where attacked images and the generated manifest are written
- `split`: `train`, `test`, or `all`
- `shape_map`: optional mapping from `class_id` to sign shape when the CSV does not provide a `shape` column
- `label_map`: optional mapping from `class_id` to a readable label when the CSV does not provide a `label` column

## Run

```bash
python3 -m traffic_sign_attacks --config configs/traffic_sign_attack_dataset.yaml
```

## Output

The command writes:

- attacked images under `generated/<split>/<attack>/<class_id>/`
- metadata in `generated/manifest.csv`

Each manifest row includes:

- `sample_id`
- `split`
- `class_id`
- `label`
- `shape`
- `attack`
- `variant_index`
- `original_path`
- `output_path`
- `roi_x1`, `roi_y1`, `roi_x2`, `roi_y2`
- serialized `attack_parameters`
- serialized `transform_parameters`

## Project Layout

```text
configs/                  Example YAML config
example_result/           Small sample outputs used in this README
traffic_sign_attacks/     Generator package
```
