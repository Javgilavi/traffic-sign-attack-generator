# Methodology

## Goal

Create a reproducible dataset-generation pipeline for physically plausible traffic-sign perturbations based on:

- artificial shadows
- artificial occlusions and sticker-like patches

The generated data is meant to stress-test a traffic-sign classifier trained by your teammates on GTSRB.

## Literature Basis

### 1. Shadow Attack

Source: `2203.03818v3.pdf`

Key implementation choices adopted here:

- model the perturbation as a polygon shadow, with triangles as the default mask
- restrict the shadow to the sign region
- apply the perturbation in LAB space by attenuating the `L` channel only
- keep shadow strength in a realistic range around the paper's reported mean coefficient
- simulate physical variability with small post-transforms

What is simplified:

- the original paper uses black-box optimization over triangle vertices
- this project samples physically plausible triangles instead of optimizing against a classifier

That simplification is intentional because the target model is not included in this repo.

### 2. Occlusion / Patch Attack

Sources:

- `2512.00765v2.pdf`
- `2307.08278v1.pdf`

Key implementation choices adopted here:

- use sign-shape-aware masks rather than arbitrary full-image cutouts
- include edge-conforming annular patches for stealthier physical occlusion
- include sticker-like central patches and diagonal tape-style bands for stronger failure cases
- use muted colors and adjustable opacity to stay physically plausible

What is simplified:

- the edge-patch paper learns the patch content with a generator and classifier loss
- this project generates deterministic, configurable physical occlusion masks without training a patch network

## Supported Sign Geometry

The pipeline maps each GTSRB class to a coarse sign shape:

- circle
- triangle
- inverted triangle
- diamond
- octagon
- box fallback

These masks are used to constrain both the shadow and the occlusion to the traffic sign rather than the full crop.

## Physical Robustness Approximation

The papers use EOT or similar physical robustness ideas. This project approximates that at data generation time using:

- small rotation
- brightness scaling
- contrast scaling
- Gaussian blur

This is not a full physical simulator, but it is a reasonable first robustness layer for dataset creation.

## Output Philosophy

The pipeline exports all parameters needed for traceability:

- attack type
- mask geometry
- opacity or shadow coefficient
- output path
- ROI
- transform settings

That makes later ablation studies possible without regenerating the full dataset blindly.
