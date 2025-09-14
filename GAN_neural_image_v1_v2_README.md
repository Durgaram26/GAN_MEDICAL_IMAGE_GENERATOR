# GAN Neural Image — v1 & v2 (Medical Image Generation)

This document describes two GAN-based image generation projects (v1 and v2) included in this repository. Both versions are designed for medical image synthesis and experimentation (e.g., chest X‑rays, MRI slices). The README explains repository layout, setup, training, inference, and tips for reproducibility.

## Overview

- **v1**: Baseline GAN (e.g., DCGAN/standard GAN) used for initial medical image synthesis experiments.
- **v2**: Improved model (e.g., StyleGAN2-ADA or a modified architecture) with better fidelity and training stability. v2 includes training logs and checkpointed models in `final_round`/`final_round_1` subfolders.

Both versions aim to generate realistic medical images for augmentation, research, or visualization purposes. Use responsibly and do not deploy for clinical decision-making.

## Repository locations

- `final_round/` — contains v2 models, templates, and generated images.
- `final_round_1/final_round_1/` — alternate v2 location with models and generated images.
- `Datrex/` and `Hackathon/medsynthgen/` — may contain related scripts, notebooks, or saved models for preparation, training, and inference.

## Quick setup

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install common dependencies (adjust versions per your environment):

```bash
pip install torch torchvision numpy matplotlib Pillow tqdm
```

For v2 (StyleGAN2-ADA) you may need additional dependencies (e.g., `diffusers`, `torchvision`, or the official StyleGAN2-ADA repo requirements). If there is a `requirements.txt` in the model folders, prefer using it.

## Training (high level)

v1 (baseline GAN):

- Prepare a dataset of aligned, same-sized medical images (e.g., 256x256 or 512x512), placed in a folder structure expected by the training script.
- Configure hyperparameters (batch size, learning rate, number of epochs) in the training script or a `parameters.json` file.
- Start training with the provided training script (example):

```bash
python train_gan_v1.py --data ./datasets/medical_images --epochs 100 --batch-size 32
```

v2 (StyleGAN2-ADA or advanced):

- Use the v2 training script or the StyleGAN2-ADA repo commands. Ensure you use the `--gpus`/`--batch` flags appropriate for your hardware.
- Example (StyleGAN2-ADA CLI style):

```bash
python train.py --outdir=./results --data=./datasets/medical_images.zip --gpus=1 --cfg=auto
```

Training artifacts (checkpoints, parameters) are saved under `results/` or `generated_images/` folders inside the model directories.

## Inference / Generate images

- Use the included generator model (e.g., `generator_343.pth` or `stylegan_final_model.pth`) to sample images.
- Example script usage:

```bash
python generate_images.py --model final_round/models/generator_343.pth --num-images 10 --outdir generated_images/sample_run
```

- Some folders include zipped generated outputs (e.g., `generated_images/*/*.zip`) for quick inspection.

## Files to look for

- `generator_343.pth`, `stylegan_final_model.pth`, `discriminator_StyleGAN2_ADA.pth` — model weights for inference/training continuation.
- `parameters.json` — training configuration (learning rate, image size, augmentation settings).
- `training_log.json` — training progress and metrics.
- `generated_images/` — sample outputs produced during or after training.

## Reproducing results

- Save the environment (`pip freeze > requirements.txt`) after installing the exact packages used for training.
- Use deterministic seeds (if provided) in training scripts to make experiments reproducible.
- Keep checkpoints regularly and log metrics (e.g., FID, IS) for comparison.

## Tips and best practices

- Use mixed precision (AMP) when training large models to save memory and accelerate training.
- Apply domain-specific data augmentation carefully — preserve medically relevant features.
- Validate generated images with domain experts before using them in downstream tasks.

## License & ethical considerations

- These models are for research and educational use only. Generated images are synthetic and should not be used for medical diagnosis.
- Add an appropriate `LICENSE` file if you plan to distribute models or code publicly. 