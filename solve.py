"""FlowDPS-TR: a fork of FlowDPS with proximal optimization from
[AACV2026] Local Gaussian Conditioning and Covariance-Scaled Trust Regions
for Flow Matching.

The solver is based on exactgauss/FlowDPS/solve_gaussian_flowdps_v2.py and
reuses the original FlowDPS model integration and degradation operators.
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from munch import munchify
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from util import get_img_list, process_text, set_seed
from functions.degradation import get_degradation
from sd3_sampler import SD3Euler, get_solver, register_solver

TensorTree = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]


def tree_add_noise(y: TensorTree, std: float) -> TensorTree:
    if std <= 0:
        return y
    if isinstance(y, (tuple, list)):
        return type(y)(tree_add_noise(v, std) for v in y)
    return y + std * torch.randn_like(y)


def tree_to_device(y: TensorTree, device: torch.device) -> TensorTree:
    if isinstance(y, (tuple, list)):
        return type(y)(tree_to_device(v, device) for v in y)
    return y.to(device=device)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred01 = ((pred.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    target01 = ((target.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    mse = (pred01 - target01).pow(2).flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return torch.where(mse == 0, torch.full_like(psnr, float("inf")), psnr)


@torch.no_grad()
def precompute(args, prompts: List[str], solver):
    prompt_emb_set = []
    pooled_emb_set = []
    n = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:n]:
        prompt_emb, pooled_emb = solver.encode_prompt(prompt, batch_size=1)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)
    return prompt_emb_set, pooled_emb_set


@register_solver("gaussian_flowdps")
class SD3GaussianFlowDPS(SD3Euler):
    """Extend the FlowDPS sampler with local Gaussian proximal optimization."""

    def __init__(self, model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers", device="cuda"):
        super().__init__(model_key=model_key, device=device)
        self.vae.requires_grad_(False)

    def _residual_pair(self, x: torch.Tensor, operator, measurement: TensorTree, task: str, mode: str):
        if mode == "measurement":
            return operator.A(x), measurement

        if mode == "pinv" or (mode == "hybrid" and "sr" in task):
            if "sr" in task and hasattr(operator, "A_pinv"):
                return operator.A_pinv(operator.A(x)), operator.A_pinv(measurement)
            return operator.At(operator.A(x)), operator.At(measurement)

        if mode == "hybrid":
            return (operator.A(x), operator.At(operator.A(x))), (measurement, operator.At(measurement))

        raise ValueError(f"Unknown residual mode: {mode}")

    def _loss_from_pair(self, pred: TensorTree, target: TensorTree, obs_std: float, loss_type: str) -> torch.Tensor:
        if isinstance(pred, (tuple, list)):
            return sum(self._loss_from_pair(p, t, obs_std, loss_type) for p, t in zip(pred, target))
        pred = pred.float()
        target = target.to(device=pred.device, dtype=pred.dtype)
        r = (pred - target).reshape(pred.shape[0], -1)
        if loss_type == "l2":
            return 0.5 * (r / obs_std).pow(2).mean()
        if loss_type == "charbonnier":
            return torch.sqrt((r / obs_std).pow(2) + 1e-6).mean()
        raise ValueError(f"Unknown loss type: {loss_type}")

    def _constraint_avg_l2(self, pred: TensorTree, target: TensorTree) -> torch.Tensor:
        if isinstance(pred, (tuple, list)):
            values = [self._constraint_avg_l2(p, t) for p, t in zip(pred, target)]
            return torch.stack(values).sum()
        pred = pred.float()
        target = target.to(device=pred.device, dtype=pred.dtype)
        r = (pred - target).reshape(pred.shape[0], -1)
        return torch.linalg.vector_norm(r, dim=1).mean()

    def gaussian_clean_update(
        self,
        z0t: torch.Tensor,
        operator,
        measurement: TensorTree,
        task: str,
        sigma: float,
        inner_steps: int,
        prox_lr: float,
        obs_std: float,
        prior_scale: float,
        prior_floor: float,
        residual_mode: str,
        loss_type: str,
    ) -> Tuple[torch.Tensor, float]:
        """Refine the predicted clean latent with a Gaussian proximal penalty."""
        base = z0t.detach()
        base32 = base.float()
        prior_std = max(float(prior_floor), float(prior_scale) * float(sigma))
        obs_std = max(float(obs_std), 1e-6)

        delta = torch.zeros_like(base32, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=prox_lr, betas=(0.5, 0.9))

        for _ in range(inner_steps):
            opt.zero_grad(set_to_none=True)
            z_trial = (base32 + delta).to(dtype=self.dtype)
            x_trial = self.decode(z_trial).float()
            pred, target = self._residual_pair(x_trial, operator, measurement, task, residual_mode)
            data_loss = self._loss_from_pair(pred, target, obs_std, loss_type)
            # The local Gaussian prior penalizes displacement from the flow prediction.
            prior_loss = 0.5 * (delta / prior_std).pow(2).mean()
            loss = data_loss + prior_loss
            # Only optimize the latent, not the fixed degradation operator.
            delta.grad, = torch.autograd.grad(loss, delta)
            opt.step()

        z_updated = (base32 + delta.detach()).to(dtype=base.dtype)
        with torch.no_grad():
            x_updated = self.decode(z_updated).float()
            pred, target = self._residual_pair(x_updated, operator, measurement, task, residual_mode)
            constraint_avg_l2 = float(self._constraint_avg_l2(pred, target).detach().cpu())

        return z_updated, constraint_avg_l2

    def sample(
        self,
        measurement,
        operator,
        task,
        prompts: List[str],
        NFE: int,
        img_shape: Optional[Tuple[int, int]] = None,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        step_size: float = 0.05,
        latent: Optional[torch.Tensor] = None,
        prompt_emb: Optional[List[torch.Tensor]] = None,
        null_emb: Optional[List[torch.Tensor]] = None,
        gauss_inner_steps: int = 3,
        gauss_obs_std: float = 0.08,
        gauss_prior_scale: float = 1.0,
        gauss_prior_floor: float = 0.015,
        gauss_blend: float = 1.0,
        gauss_blend_power: float = 1.0,
        gauss_residual_mode: str = "pinv",
        gauss_loss: str = "l2",
        renoise: bool = True,
    ):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]
            prompt_emb = prompt_emb.to(self.transformer.device)
            pooled_emb = pooled_emb.to(self.transformer.device)

            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]
            null_prompt_emb = null_prompt_emb.to(self.transformer.device)
            null_pooled_emb = null_pooled_emb.to(self.transformer.device)

        z = self.initialize_latent((imgH, imgW), batch_size) if latent is None else latent.to(self.transformer.device, dtype=self.dtype)
        measurement = tree_to_device(measurement, self.transformer.device)

        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc="SD3-Gaussian-FlowDPS")
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                    pred_v = pred_null_v + cfg_scale * (pred_v - pred_null_v)

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.zeros_like(sigma)
            sigma_f = float(sigma.detach().cpu())
            sigma_next_f = float(sigma_next.detach().cpu())

            z0t = z - sigma * pred_v
            z1t = z + (1.0 - sigma) * pred_v

            z0_prox, constraint_avg_l2 = self.gaussian_clean_update(
                z0t=z0t,
                operator=operator,
                measurement=measurement,
                task=task,
                sigma=sigma_f,
                inner_steps=gauss_inner_steps,
                prox_lr=step_size,
                obs_std=gauss_obs_std,
                prior_scale=gauss_prior_scale,
                prior_floor=gauss_prior_floor,
                residual_mode=gauss_residual_mode,
                loss_type=gauss_loss,
            )
            pbar.set_postfix({"constraint_l2": f"{constraint_avg_l2:.4e}"})

            blend = max(0.0, min(1.0, gauss_blend * (sigma_f ** gauss_blend_power)))
            z0y = z0t + blend * (z0_prox - z0t)

            if renoise and sigma_next_f > 0:
                a = math.sqrt(max(sigma_next_f, 0.0))
                b = math.sqrt(max(1.0 - sigma_next_f, 0.0))
                noise = a * z1t + b * torch.randn_like(z1t)
                z = z0y + sigma_next * (noise - z0y)
            else:
                z = z0y + sigma_next * (z1t - z0y)

            z = z.detach()

        with torch.no_grad():
            return self.decode(z)


def run(args):
    solver = get_solver(args.method)
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)

    solver.text_enc_1.to("cuda")
    solver.text_enc_2.to("cuda")
    solver.text_enc_3.to("cuda")

    if args.efficient_memory:
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
            null_emb, null_pooled_emb = solver.encode_prompt([""], batch_size=1)
        del solver.text_enc_1, solver.text_enc_2, solver.text_enc_3
        torch.cuda.empty_cache()
        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
        null_embs = [null_emb, null_pooled_emb]
    else:
        prompt_embs = [None] * len(prompts)
        null_embs = None

    print("Prompts are processed.")

    solver.vae.to("cuda")
    solver.transformer.to("cuda")

    deg_config = munchify({"channels": 3, "image_size": args.img_size, "deg_scale": args.deg_scale})
    operator = get_degradation(args.task, deg_config, solver.transformer.device)

    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    psnr_values = []
    pbar = tqdm(get_img_list(args.img_path), desc="Solving")
    for i, path in enumerate(pbar):
        img = tf(Image.open(path).convert("RGB"))
        img = img.unsqueeze(0).to(solver.vae.device)
        img = img * 2 - 1

        # Observations are fixed, even when A contains trainable torch modules.
        with torch.no_grad():
            y = operator.A(img)
            y = tree_add_noise(y, args.measurement_noise)

        prompt = prompts[i] if len(prompts) > 1 else prompts[0]
        p_emb = prompt_embs[i] if len(prompt_embs) > 1 else prompt_embs[0]

        sample_kwargs = dict(
            measurement=y,
            operator=operator,
            prompts=prompt,
            NFE=args.NFE,
            img_shape=(args.img_size, args.img_size),
            cfg_scale=args.cfg_scale,
            step_size=args.step_size,
            task=args.task,
            prompt_emb=p_emb,
            null_emb=null_embs,
        )
        if args.method == "gaussian_flowdps":
            sample_kwargs.update(
                gauss_inner_steps=args.gauss_inner_steps,
                gauss_obs_std=args.gauss_obs_std,
                gauss_prior_scale=args.gauss_prior_scale,
                gauss_prior_floor=args.gauss_prior_floor,
                gauss_blend=args.gauss_blend,
                gauss_blend_power=args.gauss_blend_power,
                gauss_residual_mode=args.gauss_residual_mode,
                gauss_loss=args.gauss_loss,
                renoise=not args.no_renoise,
            )

        out = solver.sample(**sample_kwargs)
        psnr_value = float(compute_psnr(out, img).mean().detach().cpu())
        psnr_values.append(psnr_value)
        pbar.set_postfix({"psnr": f"{psnr_value:.4f}"})

        inp = operator.At(y).reshape(img.shape) if hasattr(operator, "At") else operator.A_pinv(y).reshape(img.shape)
        save_image(inp, args.workdir.joinpath(f"input/{i:04d}.png"), normalize=True)
        save_image(out, args.workdir.joinpath(f"recon/{i:04d}.png"), normalize=True)
        save_image(img, args.workdir.joinpath(f"label/{i:04d}.png"), normalize=True)

        if args.num_samples > 0 and (i + 1) == args.num_samples:
            break

    if psnr_values:
        mean_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Mean PSNR: {mean_psnr:.4f} dB")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--NFE", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--workdir", type=Path, default=Path("workdir_gaussian_flowdps"))
    parser.add_argument("--img_path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--task", type=str, default="sr_avgpool")
    parser.add_argument("--method", type=str, default="gaussian_flowdps")
    parser.add_argument("--deg_scale", type=int, default=12)
    parser.add_argument("--measurement_noise", type=float, default=0.005)
    parser.add_argument("--step_size", type=float, default=0.05)  # 0.05 = nice quality,  0.1 = good metrics.
    parser.add_argument("--gauss_inner_steps", type=int, default=15)
    parser.add_argument("--gauss_obs_std", type=float, default=0.001)
    parser.add_argument("--gauss_prior_scale", type=float, default=1.0)
    parser.add_argument("--gauss_prior_floor", type=float, default=0.015)
    parser.add_argument("--gauss_blend", type=float, default=1.5)
    parser.add_argument("--gauss_blend_power", type=float, default=1.0)
    parser.add_argument("--gauss_residual_mode", type=str, default="pinv", choices=["pinv", "measurement", "hybrid"])
    parser.add_argument("--gauss_loss", type=str, default="l2", choices=["l2", "charbonnier"])
    parser.add_argument("--no_renoise", default=False, action="store_true")
    parser.add_argument("--efficient_memory", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.workdir.joinpath("input").mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath("recon").mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath("label").mkdir(parents=True, exist_ok=True)
    run(args)
