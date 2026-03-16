# Team Guideline

## Recommended Experiment Order

1. Baseline

- evaluate the classifier on the clean GTSRB test set

2. Single-factor robustness

- evaluate on shadow-only generated samples
- evaluate on occlusion-only generated samples

3. Attack-style breakdown

- compare `edge_patch`, `center_sticker`, and `diagonal_band`
- compare weak vs strong shadow coefficients

4. Class-wise analysis

- identify which sign groups are most vulnerable:
  - circular speed-limit signs
  - warning triangles
  - stop / yield / priority exceptions

## Useful Metrics

- overall accuracy on clean and attacked data
- per-class accuracy
- attack success rate if you define a clean-to-attacked misclassification metric
- confusion matrix shifts caused by each attack family

## Practical Advice

- Start with the `test` split. Do not generate a huge attacked training set until you know which perturbations matter.
- Keep the manifest alongside results. It gives you exact parameters for every generated sample.
- If a specific attack family is too weak or too strong, tune the YAML ranges instead of hardcoding changes in Python.
- If your teammates later expose a prediction API, add a scoring loop on top of this generator and search for the strongest shadow placement per image.
