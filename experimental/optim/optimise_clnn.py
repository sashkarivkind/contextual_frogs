# -*- coding: utf-8 -*-
"""
Translation of the provided OCaml (Forward_torch/Sofo) code to Python + PyTorch.

I kept the overall module structure:
- Gaussian
- Variational
- GenerativeModel
- Model (wrapping gen + var)
- ELBO and training loop
…and tried to mirror variable names and logic exactly where reasonable.

Where the OCaml code relies on library helpers (einsum strings, broadcast_to, linsolve_triangular, etc.)
I used equivalent PyTorch ops.

Places that looked questionable in the original are marked with
'QUESTION'/'NOTE' comments so you can clarify (see bottom of file too).
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

from gaussian import Gaussian, GaussianParams

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from toymodels import ToyObs

# ------------------------------
# Utilities & defaults
# ------------------------------

def get_device(cuda_index: int = 1) -> torch.device:
    """Mimic Torch.Device.Cuda 1 default; fallback to CPU if not available."""
    if torch.cuda.is_available() and cuda_index < torch.cuda.device_count():
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def ones_like_shape(shape: Tuple[int, ...], device):
    return torch.ones(shape, device=device)


def randn_like_shape(shape: Tuple[int, ...], device):
    return torch.randn(shape, device=device)


def zeros_like_shape(shape: Tuple[int, ...], device):
    return torch.zeros(shape, device=device)


def broadcast_to(x: torch.Tensor, size: Tuple[int, ...]) -> torch.Tensor:
    return x.expand(size)


def norm(x: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.sqrt(torch.mean(x ** 2)).item()



# ------------------------------
# Globals mirroring the OCaml defaults
# ------------------------------

FUDGE = 1e-4
CUDA_INDEX_DEFAULT = 1  # Torch.Device.Cuda 1
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------
# Gaussian distributions
# ------------------------------


# ------------------------------
# Variational parameters
# ------------------------------

class Variational(nn.Module):
    """
    OCaml:
    type 'g p = { x : 'g }
    Here 'g is GaussianParams.
    """
    def __init__(self, t: int, device: torch.device, args=None):
        super().__init__()
        mu = torch.zeros(t, device=device)
        sigma12 = torch.eye(t, device=device)
        self.x = GaussianParams(mu, sigma12)


# ------------------------------
# Generative model
# ------------------------------

class GenerativeModel(nn.Module):
    """
    OCaml generative parameters:
      log_learning_rate, log_learning_rate_decay, sigma_b, output_scale,
      log_weight_decay, sigma_a, sigma_x  (all scalars in the OCaml init)
    """
    def __init__(self, device: torch.device, args=None):
        super().__init__()
        # match OCaml init scales
        optimize_noises = not args.model == "toy" or args.optimize_toy_noises
        self.log_learning_rate = nn.Parameter(torch.full((1,), -6.0, device=device))  # bounded below -3 in OCaml; we ignore bound
        self.log_learning_rate_decay = nn.Parameter(torch.full((1,), 1e-5, device=device))
        self.sigma_b = nn.Parameter(torch.full((1,), 0.1, device=device))
        self.output_scale = nn.Parameter(torch.full((1,), 1.0, device=device))
        self.log_weight_decay = nn.Parameter(torch.full((1,), -0.001, device=device))
        self.sigma_a = nn.Parameter(torch.full((1,), 0.1, device=device)) if optimize_noises else torch.tensor(args.toymodel_OUsigma_obs, 
                                                                                                                       device=device, 
                                                                                                                       requires_grad=False)
        self.sigma_x = nn.Parameter(torch.full((1,), 0.1, device=device)) if optimize_noises else torch.tensor(args.toymodel_OUsigma_process, 
                                                                                                                       device=device, 
                                                                                                                       requires_grad=False)

    @staticmethod
    def better_relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def _sample_biases_and_w_in(self, n, bs, device):
        noise = torch.randn(n, device=device)          # no grad needed
        biases = self.sigma_b.reshape(1) * noise       # grad flows to sigma_b
        w_in = torch.randn(n, device=device)
        return biases, w_in


    def f(self,
          n: int,
          noises: List[torch.Tensor],   # list of [bs] tensors (one per time step)
          ys: List[Optional[torch.Tensor]],  # list of optional target tensors (shape [1] per OCaml)
          model_setting: str) -> List[torch.Tensor]:
        """
        Port of the OCaml `Generative_model.f`.
        Returns a list of a_means (each shape [bs]) for each time step.
        """
        assert len(noises) == len(ys), "noises and ys must have same length"
        bs = noises[0].shape[0]
        device = self.log_learning_rate.device

        biases, w_in = self._sample_biases_and_w_in(n=n, bs=bs, device=device)

        # state init
        w_out = torch.zeros(bs, n, device=device)
        u = torch.zeros(bs, device=device)
        x = torch.zeros(bs, device=device)
        e = torch.zeros(bs, device=device)
        lr = torch.exp(self.log_learning_rate).expand(bs)

        a_means: List[torch.Tensor] = []

        for y, noise_x in zip(ys, noises):
            if model_setting == "toy":
                x = args.toymodel_OUphi * x + noise_x
                a_means.append(1.0 * x)

            elif model_setting == "default":
                if y is None:
                    x = u + noise_x
                    h = self.better_relu(biases + (x.unsqueeze(1) * w_in.unsqueeze(0)))
                    u = torch.einsum("kj,kj->k", w_out, h)  # rowwise dot
                    a_mean = (self.output_scale * u).squeeze()
                    w_out = (torch.exp(self.log_weight_decay) * w_out)
                    a_means.append(a_mean)
                else:
                    x = u + e + noise_x
                    h = self.better_relu(biases + (x.unsqueeze(1) * w_in.unsqueeze(0)))
                    u = torch.einsum("kj,kj->k", w_out, h)
                    a_mean = (self.output_scale * u).squeeze()
                    # update w_out
                    # OCaml: e = broadcast_to(size u) y - u
                    e = y.to(device).expand_as(u) - u
                    dw_out = (e.unsqueeze(1) * h) * lr.unsqueeze(1)
                    w_out = w_out + dw_out
                    norms = torch.sqrt(FUDGE + torch.einsum("ki->k", dw_out ** 2))
                    lr = lr * torch.exp(-(torch.exp(self.log_learning_rate_decay) * norms))
                    w_out = (torch.exp(self.log_weight_decay) * w_out)
                    a_means.append(a_mean)
            else:
                raise ValueError("unknown model setting")

        return a_means


# ------------------------------
# Model wrapper (gen + var)
# ------------------------------

class FullModel(nn.Module):
    def __init__(self, t: int, device: torch.device, args=None):
        super().__init__()
        self.gen = GenerativeModel(device, args=args)
        self.var = Variational(t, device, args=args)


# ------------------------------
# ELBO / Objective
# ------------------------------

def kl_schedule(iter_t: int) -> float:
    return min(1.0, float(iter_t) / 2000.0)


def neg_elbo(beta: float,
             n: int,
             bs: int,
             theta: FullModel,
             ys: List[Optional[torch.Tensor]],
             a: List[Optional[torch.Tensor]],
             klmethod: str) -> torch.Tensor:
    """
    Returns scalar loss: (beta * KL) - log_likelihood, averaged like the OCaml does.
    """
    device = next(theta.parameters()).device

    # sample noise x_t ~ q (via reparam)
    x_samples = Gaussian.sample(theta.var.x, bs)   # [bs, t]
    # split along time to a list of [bs] tensors
    noises = [x_samples[:, t_idx] for t_idx in range(x_samples.shape[1])]

    # likelihood term
    # propagate through model
    a_means = theta.gen.f(n=n, noises=noises, ys=ys, model_setting=args.model)
    # print(f"a_means shape: {[z.shape for z in a_means]}")
    count = 0
    quad_and_const = None
    for a_mean, a_opt in zip(a_means, a):
        if a_opt is None:
            continue
        # d = a_mean - broadcast(y)
        d = a_mean - a_opt.to(device).expand_as(a_mean)
        # mean over batch, following OCaml: -0.5 * (mean(sqr(d)) / mean(sqr sigma_a)) + const
        term = -0.5 * torch.mean(d ** 2) / (theta.gen.sigma_a ** 2) \
               -0.5 * math.log(2.0 * math.pi)
        count += 1
        if quad_and_const is None:
            quad_and_const = term
        else:
            quad_and_const = quad_and_const + term

    if quad_and_const is None:
        # log_lik = torch.tensor(0.0, device=device)
        raise ValueError("no observed data points for likelihood term")
    else:
        log_det_term = (-0.5 * float(count)) * torch.log(theta.gen.sigma_a ** 2 + 0.0)
        log_lik = log_det_term + quad_and_const

    # KL term
    mu_q = theta.var.x.mu.detach() if not args.enable_kl_grad else theta.var.x.mu
    sigma12_q = theta.var.x.sigma12.detach() if not args.enable_kl_grad else theta.var.x.sigma12
    if klmethod == "analytical":
        kl = Gaussian.gaussian_kl_full_vs_diag(
            mu_p=mu_q,
            sigma12_p=sigma12_q,
            diag_sigma2_q=(theta.gen.sigma_x ** 2).expand_as(theta.var.x.mu)
        )
        # (scalar)
    elif klmethod == "montecarlo":
        # log q (stop-grad through q params, like OCaml)
        q_theta_detached = GaussianParams(mu_q, sigma12_q)
        log_q = Gaussian.log_density(q_theta_detached, x_samples)

        # log p for diagonal Gaussian prior N(0, diag(sigma_x^2))
        x_term = (-0.5 / float(bs)) * torch.sum(x_samples ** 2) / torch.sum(FUDGE + (theta.gen.sigma_x ** 2))
        twopi = 2.0 * math.pi
        log_p = x_term - (0.5 * float(len(a))) * torch.sum(torch.log(FUDGE + twopi * (theta.gen.sigma_x ** 2)))
        kl = log_q - log_p
    else:
        raise ValueError("unknown kl method")

    neg_elbo_val = (beta * kl) - log_lik
    # OCaml finishes with: any ((1. / len(ys)) * neg_elbo)
    # Here kl and log_lik are scalars, so we just scale:
    # loss = neg_elbo_val / float(len(ys))
    loss = neg_elbo_val 
    return loss, log_lik, kl


# ------------------------------
# Training / IO
# ------------------------------

def save_results(theta: FullModel, outdir: Path, ys: List[Optional[torch.Tensor]], n: int, incl_matrices: bool = True):
    outdir.mkdir(parents=True, exist_ok=True)

    # Save parameters
    torch.save(theta.state_dict(), outdir.joinpath("params.pt"))

    # Save posterior matrices (var.x)
    with torch.no_grad():
        if incl_matrices:
            sigma12 = theta.var.x.sigma12
            sigma = sigma12 @ sigma12.transpose(-1, -2)

            np.savetxt(outdir.joinpath("post_sigma.txt"), sigma.detach().cpu().numpy())
            np.savetxt(outdir.joinpath("post_sigma12.txt"), sigma12.detach().cpu().numpy())

        mu = theta.var.x.mu.reshape(-1, 1)
        np.savetxt(outdir.joinpath("post_mu.txt"), mu.detach().cpu().numpy())

        # predictions using mean noise = mu split across time
        noises = [theta.var.x.mu[t_idx:t_idx+1] for t_idx in range(theta.var.x.mu.shape[0])]
        # Broadcast each to [bs] via tiny helper: we’ll use bs=1 for deterministic mean path
        noises = [z.expand(1) for z in noises]
        a_pred_list = theta.gen.f(n=n, noises=noises, ys=ys, model_setting=args.model)
        pred_a = torch.cat([z.reshape(1, 1) for z in a_pred_list], dim=0)  # [T,1]
        np.savetxt(outdir.joinpath("pred_a.txt"), pred_a.detach().cpu().numpy())


def build_targets_from_txt(txt_path: Path,
                           device: torch.device) -> List[Optional[torch.Tensor]]:
    """
    OCaml: load_txt then map z -> None if NaN else Some (ones ~scale:z [1])
    Here: Optional[Tensor] with shape [1].
    """
    arr = np.loadtxt(str(txt_path))
    lst: List[Optional[torch.Tensor]] = []
    for z in arr.tolist():
        if isinstance(z, float) and (np.isnan(z)):
            lst.append(None)
        else:
            t = torch.tensor([float(z)], device=device)
            lst.append(t)
    return lst


def build_piecewise_ys(t: int, device: torch.device) -> List[Optional[torch.Tensor]]:
    """
    Mirrors the OCaml ys construction in your snippet.
    Each Some z becomes a tensor of shape [1] with that scalar.
    """
    ys: List[Optional[torch.Tensor]] = []
    for i in range(t):
        if i < 50:
            z = 0.0
        elif i < 175:
            z = 1.0
        elif i < 190:
            z = -1.0
        elif i < 192:
            ys.append(None)
            continue
        elif i < 194:
            z = 1.0
        else:
            ys.append(None)
            continue
        ys.append(torch.tensor([z], device=device))
    return ys


def main(args):
    device = get_device(args.cuda_index)
    outdir = Path(args.out_dir)

    # Data & targets
    if not args.model == "toy":
        a_list = build_targets_from_txt(Path(args.data), device=device)
    else:
        toy_obs = ToyObs(n_steps=args.t_episode, 
                         phi=args.toydata_OUphi, 
                         sigma_process=args.toydata_OUsigma_process, 
                         sigma_obs=args.toydata_OUsigma_obs, 
                         device=device, stationary_init=False)
        ys_torch, xs_torch = toy_obs.get_obs(seed=SEED, return_latents=True)
        a_list = [ys_torch[t].reshape(1) for t in range(len(ys_torch))]
        print(f'prob density = {toy_obs.get_density(obs=ys_torch).item():.3e}')
        np.savetxt(outdir.joinpath('toysynth_a.txt'), ys_torch.cpu().numpy())

    t = len(a_list) if args.t_episode is None else args.t_episode
    print(f"Data length T = {t}")
    ys = build_piecewise_ys(t, device=device)

    # Model
    if args.reuse is not None and Path(args.reuse).exists():
        model = FullModel(t=t, device=device, args=args)
        sd = torch.load(args.reuse, map_location=device)
        model.load_state_dict(sd)
    else:
        model = FullModel(t=t, device=device, args=args)

    model.to(device)

    # Optimizer with OCaml-like schedule: 1e-3 / sqrt(1 + k/100)
    def lr_for_iter(k: int) -> float:
        return 5e-3 / math.sqrt(1.0 + (k / 100.0))

    opt = torch.optim.Adam(model.parameters(), lr=lr_for_iter(0))

    # Training loop
    loss_file = outdir.joinpath("loss.csv")
    if loss_file.exists():
        loss_file.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    bs = args.bs
    n = args.n
    max_iter = args.max_iter

    for it in range(max_iter + 1):
        # manual LR schedule to mirror OCaml changing 'config'
        for g in opt.param_groups:
            g["lr"] = lr_for_iter(it)

        beta = kl_schedule(it)

        opt.zero_grad(set_to_none=True)
        loss, ll, kl = neg_elbo(beta=beta,
                        n=n,
                        bs=bs,
                        theta=model,
                        ys=ys,
                        a=a_list,
                        klmethod=args.klmethod)
        loss_float = float(loss.detach().cpu())
        ll_float = float(ll.detach().cpu())
        kl_float = float(kl.detach().cpu())
        loss.backward()
        opt.step()

        if it % 10 == 0:
            save_results(model, outdir, ys, n, incl_matrices=(it % args.save_matrices_every == 0))
            print({"t": it, "loss": loss_float, "log_lik": ll_float, "kl": kl_float,})
            if args.print_params:
                print(f' gen params: lr={torch.exp(model.gen.log_learning_rate).item():.3e}, decay={torch.exp(model.gen.log_learning_rate_decay).item():.3e}, '
                    f'sigma_b={model.gen.sigma_b.item():.3e}, sigma_a={model.gen.sigma_a.item():.3e}, '
                    f'output_scale={model.gen.output_scale.item():.3e}, sigma_x={model.gen.sigma_x.item():.3e}, '
                    f'weight_decay={torch.exp(model.gen.log_weight_decay).item():.3e}')
            with loss_file.open("a", encoding="utf-8") as f:
                f.write(f"{it},{loss_float}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out-dir", type=str, default="./out", help="Output directory (OCaml: -d)")
    p.add_argument("--data", required=True, help="Path to .txt (one number per line)")
    p.add_argument("--model", choices=["default", "toy"], default="default", help="Model setting")
    p.add_argument("--klmethod", choices=["analytical", "montecarlo"], default="montecarlo")
    p.add_argument("--reuse", type=str, default=None, help="Load parameters from a previous run (.pt)")
    p.add_argument("--cuda-index", type=int, default=CUDA_INDEX_DEFAULT)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--t-episode", type=int, default=None, help="Length of an episode (default: full data length)")
    p.add_argument("--max-iter", type=int, default=10_000)
    p.add_argument("--enable-kl-grad", action="store_true", help="Enable KL divergence gradient")
    p.add_argument("--print-params", action="store_true", help="Print model params every 10 iters")
    p.add_argument("--save-matrices-every", type=int, default=10, help="Save posterior matrices every N iters")
    # Toy model params 
    p.add_argument("-tdp","--toydata-OUphi", type=float, default=0.9, help="Toy model OU phi data param")
    p.add_argument("-tdsp","--toydata-OUsigma_process", type=float, default=0.1, help="Toy model OU sigma process data param")
    p.add_argument("-tdso","--toydata-OUsigma_obs", type=float, default=0.05, help="Toy model OU sigma observation data param")
    # p.add_argument("--toymodel-as-data", action="store_true", help="Set model params = data params for toy model")
    p.add_argument("-tmp","--toymodel-OUphi", type=float,  help="Toy model OU phi param for model")
    p.add_argument("-tmsp","--toymodel-OUsigma_process", type=float, help="Toy model OU sigma process param for model")
    p.add_argument("-tmso","--toymodel-OUsigma_obs", type=float, help="Toy model OU sigma observation param for model")
    p.add_argument("--optimize-toy-noises", action="store_true", 
                   help="Optimize sigma_a and sigma_x even in toy model (default: use fixed data params)")
    args = p.parse_args()


    args.toymodel_OUphi = args.toydata_OUphi if args.toymodel_OUphi is None else args.toymodel_OUphi
    args.toymodel_OUsigma_process = args.toydata_OUsigma_process if args.toymodel_OUsigma_process is None else args.toymodel_OUsigma_process
    args.toymodel_OUsigma_obs = args.toydata_OUsigma_obs if args.toymodel_OUsigma_obs is None else args.toymodel_OUsigma_obs

    print(f'Setting model params = data params for toy model: phi={args.toymodel_OUphi}, sigma_process={args.toymodel_OUsigma_process}, sigma_obs={args.toymodel_OUsigma_obs}')
    main(args)
