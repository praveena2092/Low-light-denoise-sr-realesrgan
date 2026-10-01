# Low-Light Image Denoising + 4× Super-Resolution (Real-ESRGAN fine-tune)

Joint denoising and 4× super-resolution of noisy, low-light images by fine-tuning a pretrained **Real-ESRGAN (RRDBNet, `x4plus`)** model.

## Task

Given low-resolution, noisy, low-light images, produce clean images at 4× the input resolution. Quality is measured with PSNR against the ground-truth high-resolution images.

| | |
|---|---|
| Train pairs | 1,105 |
| Validation pairs | 267 |
| Test images | 60 (inputs only) |
| Resolution | LR 160×256 → HR 640×1024 |

## Approach

1. **Pairing** – noisy inputs (`train/train`, `val/val`) are matched to clean targets (`gt/`) by numeric ID; the pairing logic also has fallbacks for other folder layouts.
2. **Pretrained start** – load the official `RealESRGAN_x4plus` weights into `basicsr`'s `RRDBNet` (23 blocks, 64 features). Real-ESRGAN was trained with a synthetic noise + blur + downsampling pipeline, a close match to this task.
3. **Fine-tuning** – L1 loss, Adam (lr 1e-4, cosine annealing), mixed precision, 64×64 random LR patches with horizontal flips, batch 2 × 4 gradient-accumulation steps (effective 8), 8 epochs.
4. **Model selection** – best checkpoint by validation PSNR.
5. **Inference** – tiled (256 px tiles, 16 px padding) to bound GPU memory; outputs saved as PNGs, then converted to the competition CSV format with the provided helper script.

## Results

| Metric | Value |
|---|---|
| Best validation PSNR | **39.04 dB** (epoch 8) |
| Train L1 (final epoch) | 0.0092 |
| Time per epoch | ~2 min on one GPU |

PSNR rose steadily and was still creeping up at epoch 8 (38.94 → 39.04 dB over epochs 4–8), so longer training may help.

**Caveat:** the validation loader uses a *center 64×64 LR crop* of each image (256×256 HR), so the reported PSNR is on crops, not full images. Full-image PSNR may differ.

## Repository layout

```
notebooks/realesrgan_lowlight_denoise_sr.ipynb   # pairing → fine-tune → tiled inference → submission
data/                                             # put dataset here (git-ignored)
requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
```

Open the notebook and set the data paths. It has a `SAMPLE_MODE` flag: run once with `True` (a few images, 2 epochs) to smoke-test the whole pipeline, then set it to `False` for the full run. A GPU is strongly recommended. The notebook also contains an optional cell to upload the checkpoint to the Hugging Face Hub (set `HF_TOKEN` and `REPO_ID`; never hardcode a token).

## Notes

- `basicsr`/`realesrgan` are installed with `--no-deps`, and a small shim restores `torchvision.transforms.functional_tensor`, which newer torchvision removed.
- Ideas to improve: more epochs and larger patches, a perceptual or GAN loss, self-ensemble (flip/rotate test-time augmentation), and evaluating PSNR on full images.

## Acknowledgements

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) and [BasicSR](https://github.com/XPixelGroup/BasicSR).

## License

MIT (code). Pretrained weights and datasets follow their own licenses.
