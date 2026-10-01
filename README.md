# FlowDPS-TR

This repository is a fork of
[FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems (ICCV2025)](https://github.com/FlowDPS-Inverse/FlowDPS)
with proximal optimization from **[ACCV2026] Local Gaussian Conditioning and
Covariance-Scaled Trust Regions for Flow Matching**.

The default solver in `solve.py` refines each predicted clean latent by minimizing
a measurement loss plus a Gaussian penalty around the flow model's prediction.
It blends the optimized latent into the flow trajectory, optionally applies
renoising, and reports PSNR after reconstruction. The original FlowDPS repository
provides the SD3 model integration, degradation operators, and baseline solvers.

The default method is selected with `--method gaussian_flowdps`. The original
solvers remain available through `--method` for comparison.

The overview below is from the original FlowDPS repository.
![FlowDPS overview](assets/main.jpg)

## Quick Start

### Environment Setup

First, clone this repository and install requirements.

```
git clone https://github.com/kirkegaardlab/FlowDPS-TR.git
cd FlowDPS-TR
conda create -n flowdps python==3.10
conda activate flowdps
pip install -r requirements.txt
```

> The provided requirements.txt targets the original CUDA 11.8 environment. Use a
> PyTorch/torchvision build compatible with your GPU; recent GPUs may need a newer
> CUDA build. The SD3 model requires access to
> [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
> or an existing local Hugging Face cache.

Motion blur uses the kernel implementation already included in `utils/motionblur.py`.

### Examples

You can quickly check the results using the following examples.

**Example 1. Super-resolution x 12 (avg-pool) / Dog**
```
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory;
```

**Example 2. Super-resolution x 12 (bicubic) / Animal**
```
python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory;
```
> The prompt (after "a high quality photo of") is extracted by DAPE from the given measurement.

**Example 3. Motion Deblur / Human**
```
python solve.py \
    --img_size 768 \
    --img_path samples/ffhq_example.png \
    --prompt "a photo of a closed face" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory;
```


The following figure shows results from the original FlowDPS implementation;
the Gaussian solver can produce different reconstructions.
![expect](assets/expected.jpg)


## How to choose task and solver

You can freely change the task and solver using the following arguments:
- `task` : sr_avgpool / sr_bicubic / deblur_gauss / deblur_motion
- `method` : gaussian_flowdps (default in `solve.py`) / psld / flowchef / flowdps

If you want to change the amount of degradation, change `deg_scale`. For SR tasks, it refers to the downscaling factor, and for deblurring tasks, it refers to the kernel size. 

Proximal optimization defaults are `--step_size 0.1`, `--gauss_inner_steps 15`,
`--gauss_obs_std 0.001`, `--gauss_prior_scale 1.0`, `--gauss_prior_floor 0.015`,
`--gauss_blend 1.5`, `--gauss_blend_power 1.0`, `--gauss_residual_mode pinv`,
`--gauss_loss l2`, and `--measurement_noise 0.005`. Renoising is enabled; use
`--no_renoise` to disable it. Run `python solve.py --help` for all options.
When selecting an original solver, also set its original step size and noise,
for example `--method flowdps --step_size 15 --measurement_noise 0.03`.

Results are saved to `workdir_gaussian_flowdps/{input,recon,label}/` by default;
use `--workdir` to choose another destination. `--prompt` supplies one prompt for
all images; `--prompt_file` supplies prompts using the original DAPE text format.

Run the model-free regression checks with `python -m unittest discover -s tests -v`.

## Efficient inference

If you use `--efficient_memory`, the text encoder will pre-compute text embeddings and be removed from the GPU.

This allows us to solve inverse problem with a single GPU with VRAM of 24GB.
