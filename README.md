# Do Vision Transformers generalise better to unseen camera traps?

MAAI 901 Deep Learning for Visual Data, Applied Vision Transformer Report · Aravind · University of Wollongong in Dubai · October 2026

This repository holds the code and outputs for the report. It compares **ViT-B/16**, **ResNet-50** and **CLIP ViT-B/16 (zero-shot)** on the Caltech Camera Traps 20 (CCT20) benchmark. Each model is tested on cameras seen in training (`cis_test`) and on cameras never seen in training (`trans_test`).

- **H1:** the ViT loses a smaller share of its macro F1 than ResNet-50 at unseen cameras (paired bootstrap, 1,000 resamples).
- **H2:** at unseen cameras, the ViT puts less attention-rollout mass inside the animal's bounding box when it is wrong (one-sided Mann-Whitney U).

## Results (run of 21 October 2026, seed 42)

| Model | Macro F1 seen | Macro F1 unseen | Relative gap | Accuracy seen → unseen |
| --- | --- | --- | --- | --- |
| ResNet-50 | 0.723 | 0.467 | 0.354 | 0.765 → 0.579 |
| ViT-B/16 | 0.663 | 0.404 | 0.391 | 0.682 → 0.488 |
| CLIP zero-shot | 0.330 | 0.316 | 0.042 | 0.316 → 0.362 |

- **H1 not supported.** ResNet gap minus ViT gap = −0.037 (95% CI −0.051 to −0.022): the ViT's gap is larger.
- **H2 supported.** Median in-box attention 0.090 when correct vs 0.064 when wrong (n = 3,000, p = 1.9 × 10⁻³⁴). It still holds after dividing by box area (p = 1.0 × 10⁻¹²) and within every box-size quartile.

## Repository layout

```
notebooks/cct20_vit_camera_traps.ipynb   full pipeline, executed, with all outputs
results/                                  CSV tables (scores, H1 bootstrap, H2 test, breakdowns, efficiency, training history)
results/predictions/                      class probabilities per test image (.npy) for each model
figures/                                  every figure used in the report and slides
requirements.txt
```

## How to reproduce

1. Open `notebooks/cct20_vit_camera_traps.ipynb` in Google Colab and choose a T4 GPU (Runtime → Change runtime type).
2. Run all cells. The notebook downloads CCT20 annotations and the small-image archive from LILA BC itself; no manual download or API key is needed.
3. A full run took about an hour on a T4 (ResNet-50 28.7 min and ViT 33.8 min of training, plus evaluation). Set `SMOKE_TEST = True` for a quick end-to-end check.
4. Optional: set `RUN_BIT_CONTROL = True` to add a ResNet-50 pretrained on ImageNet-21k (BiT), which separates architecture from pretraining scale. This was not run for the report.

Settings: 224 × 224 input, 10 epochs per model, batch size 64, AdamW with cosine schedule, learning rate 3 × 10⁻⁴ (ResNet-50) and 3 × 10⁻⁵ (ViT), best checkpoint chosen on seen-camera validation macro F1, seed 42.

## Data and licences

CCT20 (Beery, Van Horn & Perona, 2018) is distributed by [LILA BC](https://lila.science/datasets/caltech-camera-traps) under the Community Data License Agreement (permissive, v1.0). Images are not included in this repository. Pretrained weights come from Hugging Face (`google/vit-base-patch16-224-in21k`, `openai/clip-vit-base-patch16`) and torchvision (ResNet-50, ImageNet-1k).

## Reference

Beery, S., Van Horn, G. and Perona, P. (2018) 'Recognition in terra incognita', *ECCV 2018*, pp. 472–489.
