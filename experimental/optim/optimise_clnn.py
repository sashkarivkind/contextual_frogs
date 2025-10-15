# -*- coding: utf-8 -*-
import sys
sys.path.append("../../") 

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

from gaussian import Gaussian, GaussianParams
import pandas as pd

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from toymodels import ToyObs
import json
from models import ElboGenerativeModelTop, ElboGenerativeModelDualRate
from types import SimpleNamespace
import pickle
# ------------------------------
# Utilities & defaults
# ------------------------------

# def get_device(cuda_index: int = 1) -> torch.device:
#     """Mimic Torch.Device.Cuda 1 default; fallback to CPU if not available."""
#     if torch.cuda.is_available() and cuda_index < torch.cuda.device_count():
#         return torch.device(f"cuda:{cuda_index}")
#     return torch.device("cpu")

def get_device(cuda_index: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if cuda_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{cuda_index} but only {torch.cuda.device_count()} visible."
        )
    return torch.device(f"cuda:{cuda_index}")

def mymask(t: int):
    '''
    Create a boolean mask such that: second half is true
     in the first half every 10th is true, rest false
    '''
    mask = torch.zeros(t, dtype=torch.bool)
    mask[t // 2:] = True
    mask[::10] = True
    return mask

def load_subject_data(filename, ff_mult = 1./0.15):
    df = pd.read_csv(filename)
    a = df.Adaptation.to_numpy()
    y = ff_mult * df.FieldConstants_1.to_numpy()
    channel_trials  = np.logical_not(np.isnan(a))
    y[channel_trials] = np.nan
    q = df.ControlPoint.to_numpy() if 'ControlPoint' in df.columns else np.zeros_like(a)
    q = np.float32(q)

    return a, y, q



# ------------------------------
# Variational parameters
# ------------------------------

class Variational(nn.Module):
    """
    OCaml:
    type 'g p = { x : 'g }
    Here 'g is GaussianParams.
    """
    def __init__(self, t: int, device: torch.device, args=None, scale_for_cholesky=None):
        super().__init__()
        mu = torch.zeros(t, device=device)
        sigma12 = torch.eye(t, device=device)
        self.x_ = GaussianParams(mu, sigma12)
        self.args = args
        self.scale_for_cholesky = scale_for_cholesky

    def x(self): 
        out = SimpleNamespace()
        if self.scale_for_cholesky is not None:
            out.sigma12 = self.x_.sigma12 * self.scale_for_cholesky.reshape(-1, 1)
            out.mu = self.x_.mu
        else:
            out = self.x_
        return out
    


# ------------------------------
# Generative model
# ------------------------------


# ------------------------------
# Model wrapper (gen + var)
# ------------------------------

class FullModel(nn.Module):
    def __init__(self, t: int, device: torch.device, args=None, fudge=1e-4):
        super().__init__()
        if args.model == "dual-rate":
            self.gen = ElboGenerativeModelDualRate(device, args=args)
        else:
            self.gen = ElboGenerativeModelTop(device, args=args)
        self.var = Variational(t,
                                device,
                                args=args,
                                scale_for_cholesky=self.gen.sigma_x if args.scale_cholesky else None)
        self.fudge = fudge
    #     self.scale_cholesky = args.scale_cholesky

    # def var(self,scale_for_cholesky: Optional[torch.Tensor] = None):
    #     return self.var_ * (scale_for_cholesky if scale_for_cholesky is not None else 1.0)

# ------------------------------
# ELBO / Objective
# ------------------------------

def kl_schedule(iter_t: int) -> float:
    return min(1.0, float(iter_t) / args.kl_warmup_iters) if args.kl_warmup_iters > 0 else 1.0


def neg_elbo(beta: float,
             n: int,
             bs: int,
             theta: FullModel,
             ys: List[Optional[torch.Tensor]],
             a: List[Optional[torch.Tensor]],
             klmethod: str,
             qs: Optional[List[Optional[torch.Tensor]]] = None,
             return_a_means_hook: bool = False,
             sample_from_prior: bool = False,
             debug_dict = None,
             ): #-> torch.Tensor: TODO: undebug

    device = next(theta.parameters()).device

    #debug printout all the inputs
    # print(f'neg_elbo inputs: beta={beta}, n={n}, bs={bs}, device={device}, ys={(ys)}, a={(a)}, klmethod={klmethod}, qs len={(len(qs) if qs is not None else "None")}')
    # print(f' qs{((qs) if qs is not None else "None")}')

    t = len(ys)
    # sample noise x_t ~ q (via reparam)
    if debug_dict is not None:
        # print('Using debug x_samples')
        x_samples = debug_dict['x_samples']
    elif sample_from_prior:
        x_samples = torch.randn((bs, t), device=device) * theta.gen.sigma_x  # [bs, t]
    else:
        x_samples = Gaussian.sample(theta.var.x(), bs)   # [bs, t]
    # split along time to a list of [bs] tensors
    noises = [x_samples[:, t_idx] for t_idx in range(x_samples.shape[1])]

    # likelihood term
    # propagate through model
    # print(f'qs: {qs}')
    a_means = theta.gen.f(n=n, noises=noises, ys=ys, qs=qs, model_setting=args.model) 
    # if return_a_means_hook:
    #     return a_means

    # print(f"a_means shape: {[z.shape for z in a_means]}")
    count = 0
    ds = []
    for a_mean, a_opt in zip(a_means, a):
        if a_opt is None or torch.isnan(a_opt).all():
            continue
        # d = a_mean - broadcast(y)
        d = a_mean - a_opt.to(device).expand_as(a_mean)
        ds.append(d)
        # mean over batch, following OCaml: -0.5 * (mean(sqr(d)) / mean(sqr sigma_a)) + const
        count += 1

    ds = torch.cat(ds, dim=0) 
    sigma_out2 = theta.gen.sigma_a ** 2 if not args.assume_opt_output_noise else torch.mean(ds ** 2)

    mean_quad_and_const = -0.5 * torch.mean(ds ** 2) / (sigma_out2) \
                    -0.5 * math.log(2.0 * math.pi)

    log_det_term = (-0.5 * float(count)) * torch.log(sigma_out2)
    log_lik = log_det_term + mean_quad_and_const * float(count)
    # KL term
    mu_q = theta.var.x().mu.detach() if not args.enable_kl_grad else theta.var.x().mu
    sigma12_q = theta.var.x().sigma12.detach() if not args.enable_kl_grad else theta.var.x().sigma12
    if klmethod == "analytical":
        kl = Gaussian.gaussian_kl_full_vs_diag(
            mu_p=mu_q,
            sigma12_p=sigma12_q,
            diag_sigma2_q=(theta.gen.sigma_x ** 2).expand_as(theta.var.x().mu)
        ) 

        # (scalar)
    elif klmethod == "montecarlo":
        # log q (stop-grad through q params, like OCaml)
        q_theta_detached = GaussianParams(mu_q, sigma12_q)
        log_q = Gaussian.log_density(q_theta_detached, x_samples)

        # log p for diagonal Gaussian prior N(0, diag(sigma_x^2))
        x_term = (-0.5 / float(bs)) * torch.sum(x_samples ** 2) / torch.sum(self.fudge + (theta.gen.sigma_x ** 2))
        twopi = 2.0 * math.pi
        log_p = x_term - (0.5 * float(len(a))) * torch.sum(torch.log(self.fudge + twopi * (theta.gen.sigma_x ** 2)))
        kl = log_q - log_p
    else:
        raise ValueError("unknown kl method")

    neg_elbo_val = (beta * kl) - log_lik 
    # OCaml finishes with: any ((1. / len(ys)) * neg_elbo)
    # Here kl and log_lik are scalars, so we just scale:
    # loss = neg_elbo_val / float(len(ys))
    loss = neg_elbo_val
    if not return_a_means_hook:
        return loss, log_lik, kl
    else:
        return loss, log_lik, kl, a_means

def mock_neg_elbo(beta: float,
             n: int,
             bs: int,
             theta: FullModel,
             ys: List[Optional[torch.Tensor]],
             a: List[Optional[torch.Tensor]],
             klmethod: str,
             return_a_means_hook: bool = False,
             debug_dict = None,
             ): # -> torch.Tensor:
    # return 0.0, 0.0, 0.0
    # return theta.gen.log_learning_rate.sum(), theta.gen.log_learning_rate.sum(), theta.gen.log_learning_rate.sum(),  # mock values for loss, log_lik, kl
    return theta.var.x().mu.sum(), theta.var.x().mu.sum(), theta.var.x().mu.sum()  # mock values for loss, log_lik, kl
# ------------------------------
# Training / IO
# ------------------------------

def save_results(theta: FullModel, 
                 outdir: Path, 
                 ys: List[Optional[torch.Tensor]], 
                 n: int, 
                 incl_matrices: bool = True,
                 a_list: Optional[List[Optional[torch.Tensor]]] = None,
                 qs: Optional[List[Optional[torch.Tensor]]] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Save parameters
    torch.save(theta.state_dict(), outdir.joinpath("params.pt"))

    # Save posterior matrices (var.x)
    with torch.no_grad():
        if incl_matrices:
            sigma12 = theta.var.x().sigma12
            sigma = sigma12 @ sigma12.transpose(-1, -2)

            np.savetxt(outdir.joinpath("post_sigma.txt"), sigma.detach().cpu().numpy())
            np.savetxt(outdir.joinpath("post_sigma12.txt"), sigma12.detach().cpu().numpy())

        mu = theta.var.x().mu.reshape(-1, 1)
        np.savetxt(outdir.joinpath("post_mu.txt"), mu.detach().cpu().numpy())

        # predictions using mean noise = mu split across time
        noises = [theta.var.x().mu[t_idx:t_idx+1] for t_idx in range(theta.var.x().mu.shape[0])]
        # Broadcast each to [bs] via tiny helper: we’ll use bs=1 for deterministic mean path
        noises = [z.expand(1) for z in noises]
        a_pred_list = theta.gen.f(n=n, noises=noises, ys=ys, model_setting=args.model,qs=qs)
        pred_a = torch.cat([z.reshape(1, 1) for z in a_pred_list], dim=0)  # [T,1]
        np.savetxt(outdir.joinpath("pred_a.txt"), pred_a.detach().cpu().numpy())
        if args.save_batch_of_trajs:
            with torch.no_grad():
                for sample_from_prior in [True, False]:
                    loss,ll,kl,a_means = neg_elbo(beta=1.0,
                                    n=n,
                                    bs=args.bs,
                                    theta=theta,
                                    ys=ys,
                                    a=a_list,
                                    klmethod=args.klmethod,
                                    qs=qs,
                                    sample_from_prior=sample_from_prior,
                                    return_a_means_hook=True)
                    file_name = "prior_a_batch.npz" if sample_from_prior else "pred_a_batch.npz"
                    np.savez(outdir.joinpath(f"{file_name}"), **{'as':[this_a.cpu().numpy() for this_a in a_means], 
                                                                    'loss':loss.cpu().numpy(), 'll':ll.cpu().numpy(), 'kl':kl.cpu().numpy()})
                    print(f'Saved batch of {args.bs} trajectories to {outdir.joinpath(f"{file_name}")}')
                print(f'eval (for sanity purposes): loss={loss.item():.3e}, nll={ll.item():.3e}, kl={kl.item():.3e}')
                exit(0)

def build_targets_from_txt_or_csv(file_path: Path,
                           device: torch.device) -> List[Optional[torch.Tensor]]:

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        arr = df.Adaptation.to_numpy()
        arr *= np.sign(np.nansum(arr))
    else:
        arr = np.loadtxt(str(file_path))
    lst: List[Optional[torch.Tensor]] = []
    for z in arr.tolist():
        if isinstance(z, float) and (np.isnan(z)):
            lst.append(None)
        else:
            t = torch.tensor([float(z)], device=device)
            lst.append(t)
    return lst


def build_piecewise_ys(t: int, device: torch.device, paradigm = 'er') -> List[Optional[torch.Tensor]]:
    """
    Mirrors the OCaml ys construction in your snippet.
    Each Some z becomes a tensor of shape [1] with that scalar.
    """
    ys: List[Optional[torch.Tensor]] = []
    if paradigm == 'er':
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
    elif paradigm == 'sr':
        for i in range(t):
            if i < 50:
                z = 0.0
            elif i < 175:
                z = 1.0
            elif i < 190:
                z = -1.0
            else:
                ys.append(None)
                continue
            ys.append(torch.tensor([z], device=device))
    else:
        raise ValueError("unknown paradigm")
    return ys

def eval_paradigms(model, playlist_file, args):
    with open(playlist_file, 'rb') as f:
        playlist = pickle.load(f)

    model.eval()
    with torch.no_grad():
        outputs = {}
        for paradigm_name,paradigm in playlist.items():
            print(f'evaluating paradigm {paradigm_name} with {len(paradigm)} steps')
            ys = paradigm
            noises = torch.randn((args.bs, len(ys)), device=next(model.parameters()).device) * model.sigma_x  # [bs, t]
            noises = [noises[:, t_idx] for t_idx in range(noises.shape[1])]
            model_setting = args.model
            outputs_ = model.f(args.n,
                                noises,
                                ys,  
                                model_setting,
                                qs=None,
                                )
            outputs[paradigm_name] = np.array([z.cpu().numpy().reshape(-1) for z in outputs_]) 
    return outputs

def main(args):
    device = get_device(args.cuda_index)
    # Create output directory if doesn't exist
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (outdir / 'config.json').open('w'), indent=2)


    # Data & targets
    if not args.model == "toy":
        if args.load_ys_from_file:
            a_list, ys, qs = load_subject_data(args.data)
            a_list = [torch.tensor([z], device=device) if not np.isnan(z) else None for z in a_list]
            ys = [torch.tensor([y], device=device) if not np.isnan(y) else None for y in ys]
            qs = [torch.tensor([q], device=device) if not np.isnan(q) else None for q in qs]
            # Enable q_scale tuning if any q is not zero, nan or None
            args.enable_q_scale_tuning = any([ (q is not None) and (not torch.isnan(q).all()) and (not (q==0.0).all()) for q in qs])
        else: #backward compatibility mode
            a_list = build_targets_from_txt_or_csv(Path(args.data), device=device)
            qs = [torch.zeros((1,), device=device) for _ in a_list]
            print('warning: setting qs to zero by default.')
    else:
        toy_obs = ToyObs(n_steps=args.t_episode, 
                         phi=args.toydata_OUphi, 
                         sigma_process=args.toydata_OUsigma_process, 
                         sigma_obs=args.toydata_OUsigma_obs, 
                         device=device, stationary_init=False, mask=None if not args.toydata_usemask else mymask(args.t_episode))
        ys_torch, xs_torch = toy_obs.get_obs(seed=args.seed, return_latents=True)
        a_list = [ys_torch[t].reshape(1) for t in range(len(ys_torch))]
        ref_density = toy_obs.get_density(obs=ys_torch).item()
        print(f'prob density = {ref_density:.3e}')
        np.savetxt(outdir.joinpath('toysynth_a.txt'), ys_torch.cpu().numpy())
        np.savetxt(outdir.joinpath('toysynth_density.txt'), np.array([ref_density]))

    t = len(a_list) if args.t_episode is None else args.t_episode
    print(f"Data length T = {t}")

    if not args.load_ys_from_file: #backward compatibility mode
        ys = build_piecewise_ys(t, device=device, paradigm = args.paradigm)

    #Load or init. TODO: refactor into a function
    if args.reuse is not None and Path(args.reuse).exists():
        model = FullModel(t=t, device=device, args=args)
        sd = torch.load(args.reuse, map_location=device)

        # Figure out the intended n in the checkpoint (safer if it differs from CLI)
        ckpt_n = None
        if "gen._w_in" in sd and isinstance(sd["gen._w_in"], torch.Tensor):
            ckpt_n = sd["gen._w_in"].numel()

        target_n = ckpt_n if ckpt_n is not None else args.n

        # Prime the fixed random-feature buffers so shapes & keys match
        model.gen._ensure_random_features(n=target_n, device=device)

        if ckpt_n is not None and args.n != ckpt_n:
            print(f"Warning: --n ({args.n}) != checkpoint n ({ckpt_n}). "
                f"Using checkpoint n={ckpt_n} for consistency.")
            # You may also want to set args.n = ckpt_n here if downstream code relies on it.

        model.load_state_dict(sd)  # strict=True default
    else:
        model = FullModel(t=t, device=device, args=args)


    model.to(device)

    # Optimizer with OCaml-like schedule: 1e-3 / sqrt(1 + k/100)
    def lr_for_iter(k: int) -> float:
        return args.lr / math.sqrt(1.0 + (k / args.lr_decay_iter_scale))

    opt = torch.optim.Adam(model.parameters(), lr=lr_for_iter(0),)

    if args.eval_only:
        outputs = eval_paradigms(model.gen, args.paradigm_file, args) #dictionary of np.arrays
        np.savez(outdir.joinpath("paradigms.npz"), **outputs)
        return

    # Training loop
    loss_file = outdir.joinpath("loss.csv")
    log_file = outdir.joinpath("log.csv")
    if loss_file.exists():
        loss_file.unlink()
    if log_file.exists():
        log_file.unlink()


    # outdir.mkdir(parents=True, exist_ok=True)

    bs = args.bs
    n = args.n
    max_iter = args.max_iter
    # debug_dict = {'x_samples': model.var.x().mu.reshape(1, -1).expand(bs, -1)} 
    for it in range(max_iter + 1):

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
                        qs=qs,
                        klmethod=args.klmethod,
                        debug_dict=None,
                        )
        if it % args.save_every == 0:
        # if it % 1000000 == 9999999999999999999:
            loss_float = float(loss.detach().cpu())
            ll_float = float(ll.detach().cpu())
            kl_float = float(kl.detach().cpu())
        loss.backward()
        opt.step()

        # if False: #TODO: undebug
        if it % args.save_every == 0: #TODO: cleanup and refactor
            print({"t": it, "loss": loss_float, "log_lik": ll_float, "kl": kl_float,})
            save_results(model, outdir, ys, n, incl_matrices=(it % args.save_matrices_every == 0),a_list=a_list, qs=qs)
            if args.print_params:
                print(f' gen params: lr={torch.exp(model.gen.log_learning_rate).item():.3e}, decay={torch.exp(model.gen.log_learning_rate_decay).item():.3e}, '
                    f'sigma_b={model.gen.sigma_b.item():.3e}, sigma_a={model.gen.sigma_a.item():.3e}, '
                    f'output_scale={model.gen.output_scale.item():.3e}, sigma_x={model.gen.sigma_x.item():.3e}, '
                    f'weight_decay={torch.exp(model.gen.log_weight_decay).item():.3e}')
            with loss_file.open("a", encoding="utf-8") as f:
                f.write(f"{it},{loss_float}\n")
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"{it},{loss_float},{ll_float},{kl_float}\n")




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out-dir", type=str, default="./out", help="Output directory (OCaml: -d)")
    p.add_argument("--data", required=True, help="Path to .txt (one number per line) or a csv file with Adaptation column for real data")
    p.add_argument("--model", choices=["default", "toy", "dual-rate"], default="default", help="Model setting")
    p.add_argument("--klmethod", choices=["analytical", "montecarlo"], default="montecarlo")
    p.add_argument("--reuse", type=str, default=None, help="Load parameters from a previous run (.pt)")
    p.add_argument("--cuda-index", type=int, default=0)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--t-episode", type=int, default=None, help="Length of an episode (default: full data length)")
    p.add_argument("--max-iter", type=int, default=10_000)
    p.add_argument("--save-every", type=int, default=10, help="Save every N iters")
    p.add_argument("--enable-kl-grad", action="store_true", help="Enable KL divergence gradient")
    p.add_argument("--scale-cholesky", action="store_true", help="Scale cholesky by sigma_x")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    p.add_argument("--lr-decay-iter-scale", type=float, default=100.0, help="Learning rate decay scale in iterations")
    p.add_argument("--kl-warmup-iters", type=int, default=2000, help="Number of iterations for KL warmup (0 = no warmup)")
    p.add_argument("--assume-opt-output-noise", action="store_true", help="Assume output var noise = sigma_a^2 (else use empirical)")
    p.add_argument("--adam-epsilon", type=float, default=1e-8, help="Adam epsilon")
    p.add_argument("--paradigm", choices=['NA','er','sr'], default='NA', help="paradigm er (evoked recovery) or sr (spontaneous recovery)")
    p.add_argument("--load-ys-from-file", action="store_true", help="Load ys and qs from data file (csv with Adaptation and FieldConstants_1 columns)")
    # filtering
    p.add_argument("--enable-qlpf", action="store_true", help="Enable q low-pass filter (tau_qlpf)")
    p.add_argument("--enable-ylpf", action="store_true", help="Enable y low-pass filter (tau_ylpf)")
    p.add_argument("--enable-q-scale-tuning", action="store_true", help="Enable tuning of q_scale (else fixed to 1.0)")


    # Logging / saving
    p.add_argument("--print-params", action="store_true", help="Print model params every 10 iters")
    p.add_argument("--save-matrices-every", type=int, default=10, help="Save posterior matrices every N iters")
    p.add_argument("--save-batch-of-trajs", action="store_true", help="Save a batch of trajectories and exit")
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
    p.add_argument("--toydata-usemask", action="store_true", help="Use a mask for the toy data (half observed, every 10th in first half)")
    
    #noise injection node:
    p.add_argument("--noise-injection-node", choices=['a','x','u'], default='x', help="Node to inject noise in: a (output) or x (state)")
    #post training eval
    p.add_argument("--eval-only", action="store_true", help="Only eval saved model on paradigms in --paradigm-file")
    p.add_argument("--paradigm-file", type=str, default="paradigms.pkl", help="Path to paradigms .pkl file")
    #weight decay fix
    p.add_argument("--model-tie-lr-weight-decay", action="store_true", help="Tie weight decay to learning rate")
    args = p.parse_args()


    args.toymodel_OUphi = args.toydata_OUphi if args.toymodel_OUphi is None else args.toymodel_OUphi
    args.toymodel_OUsigma_process = args.toydata_OUsigma_process if args.toymodel_OUsigma_process is None else args.toymodel_OUsigma_process
    args.toymodel_OUsigma_obs = args.toydata_OUsigma_obs if args.toymodel_OUsigma_obs is None else args.toymodel_OUsigma_obs

    print(f'Setting model params = data params for toy model: phi={args.toymodel_OUphi}, sigma_process={args.toymodel_OUsigma_process}, sigma_obs={args.toymodel_OUsigma_obs}')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
    print("Done.")
    
    exit(0) #TODO: there is some freeze that pervents exit
