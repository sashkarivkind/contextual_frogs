import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedElboGenerativeModelTop(nn.Module):
    """
    Same logic as ElboGenerativeModelTop, but *all* learnable scalars become vectors of shape [bs].
    Each batch entry therefore has its own:
      - log_learning_rate, log_learning_rate_decay
      - sigma_b, sp_weight_decay, output_scale, u_feedback_scale, q_scale, ...
      - (optional) sigma_a, sigma_x
      - (optional) tau* and direct_injection_scale
    """

    def __init__(self, device: torch.device, args=None, fudge=1e-4, batch_size: Optional[int] = None):
        super().__init__()

        # --- set default args ---
        default_args = {
            "injection_opt": 1,
            "skip_gain": 0.0,
            "channel_trial_extra_error": 0.0,
        }
        for key, value in default_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)

        # batch size: prefer explicit, else args.bs must exist
        if batch_size is None:
            if not hasattr(args, "bs"):
                raise ValueError("Provide batch_size or set args.bs")
            batch_size = int(args.bs)
        self.bs = int(batch_size)

        # --- randomize initial parameters (now per batch entry) ---
        # use torch.rand to stay on-device
        def randu(shape, low, high):
            return low + (high - low) * torch.rand(shape, device=device)

        # ranges mirror your original scalar init
        init_log_learning_rate = randu((self.bs,), -7.0, -4.5) if not args.zzz_legacy_init else torch.full((self.bs,), -6.0, device=device)
        init_log_learning_rate_decay = randu((self.bs,), -1.0, 1.0) if not args.zzz_legacy_init else torch.full((self.bs,), 0.0, device=device)
        init_sigma_b = randu((self.bs,), 0.05, 0.55) if not args.zzz_legacy_init else torch.full((self.bs,), 0.1, device=device)
        init_output_scale = randu((self.bs,), 0.8, 1.0) if not args.zzz_legacy_init else torch.full((self.bs,), 1.0, device=device)

        if args.model_tie_lr_weight_decay:
            init_sp_weight_decay = randu((self.bs,), -5.0, -3.0) if not args.zzz_legacy_init else torch.full((self.bs,), -4.0, device=device)
        else:
            init_sp_weight_decay = randu((self.bs,), -8.0, -6.0) if not args.zzz_legacy_init else torch.full((self.bs,), -7.0, device=device)

        init_sigma_a = randu((self.bs,), 0.02, 0.12) if not args.zzz_legacy_init else torch.full((self.bs,), 0.1, device=device)
        init_sigma_x = randu((self.bs,), 0.02, 0.12) if not args.zzz_legacy_init else torch.full((self.bs,), 0.1, device=device)

        optimize_noises = (not args.model == "toy") or getattr(args, "optimize_toy_noises", False)

        # --- per-batch parameters ---
        self.log_learning_rate = nn.Parameter(init_log_learning_rate)
        self.log_learning_rate_decay = nn.Parameter(init_log_learning_rate_decay)
        self.sigma_b = nn.Parameter(init_sigma_b)

        if args.enable_output_scale_tuning:
            self.output_scale = nn.Parameter(init_output_scale)
        else:
            self.output_scale = torch.ones(self.bs, device=device, requires_grad=False)

        if args.enable_u_feedback_scale_tuning:
            self.u_feedback_scale = nn.Parameter(torch.ones(self.bs, device=device))
        else:
            self.u_feedback_scale = torch.ones(self.bs, device=device, requires_grad=False)

        self.sp_weight_decay = nn.Parameter(init_sp_weight_decay)

        if args.enable_q_scale_tuning:
            self.q_scale = nn.Parameter(randu((self.bs,), 0.3, 1.5))
        else:
            self.q_scale = torch.ones(self.bs, device=device, requires_grad=False)

        if not args.assume_opt_output_noise:
            if optimize_noises:
                self.sigma_a = nn.Parameter(init_sigma_a)
            else:
                self.sigma_a = torch.full((self.bs,), float(args.toymodel_OUsigma_obs), device=device, requires_grad=False)

        if optimize_noises:
            self.sigma_x = nn.Parameter(init_sigma_x)
        else:
            self.sigma_x = torch.full((self.bs,), float(args.toymodel_OUsigma_process), device=device, requires_grad=False)

        # filters / injection (per batch)
        self.tauqlpf_m1 = nn.Parameter(torch.full((self.bs,), -1.0, device=device)) if args.enable_qlpf else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
        self.tauylpf_m1 = nn.Parameter(torch.full((self.bs,), -1.0, device=device)) if args.enable_ylpf else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
        self.tauelpf_m1 = nn.Parameter(torch.full((self.bs,),  1.0, device=device)) if args.enable_elpf else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)

        self.direct_injection_scale = nn.Parameter(randu((self.bs,), 0.05, 0.4)) if args.enable_direct_injection else torch.zeros(self.bs, device=device, requires_grad=False)

        # --- random features (shared across batch, as before) ---
        self.register_buffer("_z_biases", torch.empty(0))  # base N(0,1) for biases, shape [n]
        self.register_buffer("_w_in", torch.empty(0))      # random input features, shape [n]
        self.register_buffer("_w_inq", torch.empty(0))     # random input features for q, shape [n]

        self.args = args
        self.fudge = fudge

    @staticmethod
    def better_relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        # keep your original definition
        return torch.log1p(torch.exp(x))

    def _ensure_random_features(self, n: int, device):
        if self._w_in.numel() != n:
            self.register_buffer("_z_biases", torch.randn(n, device=device))
            self.register_buffer("_w_in", torch.randn(n, device=device))
            self.register_buffer("_w_inq", torch.randn(n, device=device))

    def get_biases_and_w_in(self, n: int, device):
        self._ensure_random_features(n, device)
        # per-batch sigma_b -> per-batch biases vector
        # biases: [bs, n]
        biases = self.sigma_b.unsqueeze(1) * self._z_biases.unsqueeze(0)
        return biases, self._w_in  # w_in: [n]

    def get_winq(self, n: int, device):
        self._ensure_random_features(n, device)
        return self._w_inq  # [n]

    def _expand_time_input(self, tval: torch.Tensor, bs: int) -> torch.Tensor:
        """
        Make sure per-time tensors are shape [bs].
        Accepts scalar [], [1], or [bs].
        """
        if tval is None:
            return torch.zeros(bs, device=self.log_learning_rate.device)
        if tval.ndim == 0:
            return tval.expand(bs)
        if tval.numel() == 1 and tval.shape[0] == 1:
            return tval.expand(bs)
        if tval.shape[0] != bs:
            raise ValueError(f"Expected time-slice with shape [bs]={bs}, got {tuple(tval.shape)}")
        return tval

    def f(
        self,
        n: int,
        noises: List[torch.Tensor],                # list of [bs]
        ys: List[Optional[torch.Tensor]],          # list of [bs] or None
        model_setting: str,
        qs: Optional[List[Optional[torch.Tensor]]] = None,  # list of [bs] or [1] or None
    ) -> List[torch.Tensor]:
        assert len(noises) == len(ys), "noises and ys must have same length"
        bs = noises[0].shape[0]
        if bs != self.bs:
            raise ValueError(f"Model was initialized with bs={self.bs}, but got input bs={bs}")

        device = self.log_learning_rate.device
        biases, w_in = self.get_biases_and_w_in(n=n, device=device)  # biases [bs,n], w_in [n]
        prescaled_w_inq = self.get_winq(n=n, device=device) if qs is not None else None  # [n] or None

        # state init (per batch)
        w_out = torch.zeros(bs, n, device=device)
        u = torch.zeros(bs, device=device)
        x = torch.zeros(bs, device=device)
        e = torch.zeros(bs, device=device)
        lr_mult = torch.ones(bs, device=device)
        qlp = torch.zeros(bs, device=device)
        ylp = torch.zeros(bs, device=device)
        elp = torch.zeros(bs, device=device)

        lr0 = torch.exp(self.log_learning_rate)  # [bs]
        lr = lr0 * lr_mult

        tauqlpf = 1.0 + self.softplus(self.tauqlpf_m1)  # [bs]
        tauylpf = 1.0 + self.softplus(self.tauylpf_m1)  # [bs]
        tauelpf = 1.0 + self.softplus(self.tauelpf_m1)  # [bs]

        inj_ = self.direct_injection_scale  # [bs]

        # default qs if none
        if qs is None:
            qs = [torch.zeros((1,), device=device)] * len(ys)

        a_means: List[torch.Tensor] = []

        for y, noise_x, q in zip(ys, noises, qs):
            # y handling
            if y is None:
                y = torch.full((bs,), np.double("nan"), device=device)
            else:
                y = self._expand_time_input(y, bs)

            q = self._expand_time_input(q if q is not None else torch.zeros((1,), device=device), bs)
            noise_x = self._expand_time_input(noise_x, bs)

            if model_setting == "toy":
                x = self.args.toymodel_OUphi * x + noise_x
                a_means.append(1.0 * x)
                continue

            if model_setting != "default":
                raise ValueError("unknown model setting")

            # q low-pass
            qlp = (1.0 - 1.0 / tauqlpf) * qlp + (1.0 / tauqlpf) * q

            # scaled q input: [bs,n]
            if prescaled_w_inq is not None:
                q_gain = (self.q_scale * qlp).unsqueeze(1)  # [bs,1]
                scaled_q_in = prescaled_w_inq.unsqueeze(0) * q_gain  # [bs,n]
            else:
                scaled_q_in = 0.0

            # x update
            x = (
                u * self.u_feedback_scale
                + e
                # TODO: add noise dimention and restore noise options+ (noise_x if self.args.noise_injection_node == "x" else 0.0)
            )

            mask = torch.isnan(y)  # [bs]

            if self.args.injection_opt == 1:
                raise ValueError("injection_opt 1 is deprecated, use injection_opt 2 instead")

            # hidden
            h = self.better_relu(
                biases
                + (x.unsqueeze(1) * w_in.unsqueeze(0))
                + scaled_q_in
            )  # [bs,n]

            u = torch.einsum("kj,kj->k", w_out, h) # TODO: add noise dimention and restore noise options+ (noise_x if self.args.noise_injection_node == "u" else 0.0)
            u = u + self.args.skip_gain * x

            a_mean = self.output_scale * u # TODO: add noise dimention and restore noise options+ (noise_x if self.args.noise_injection_node == "a" else 0.0)

            # channel trial handling
            y_ = torch.where(mask, u + self.args.channel_trial_extra_error, y)
            ylp = (1.0 - 1.0 / tauylpf) * ylp + (1.0 / tauylpf) * y_

            # error
            e = ylp - u
            elp = (1.0 - 1.0 / tauelpf) * elp + (1.0 / tauelpf) * e

            # injection_opt == 2: leak perturbation into hidden
            if self.args.injection_opt == 2:
                if self.args.noise_injection_node == "x":
                    raise ValueError("injection_opt2 incompatible with noise_injection_node 'x' yet")
                x_prime = u * self.u_feedback_scale + e
                h_prime = self.better_relu(
                    biases
                    + (x_prime.unsqueeze(1) * w_in.unsqueeze(0))
                    + scaled_q_in
                )
                inj_col = inj_.unsqueeze(1)  # [bs,1]
                h_bar = (1.0 - inj_col) * h + inj_col * h_prime
            else:
                h_bar = h

            # weight update
            dw_out = lr.unsqueeze(1) * elp.unsqueeze(1) * h_bar  # [bs,n]
            norms = torch.sqrt(self.fudge + torch.einsum("ki->k", dw_out ** 2))  # [bs]

            wd = self.softplus(self.sp_weight_decay)  # [bs]
            if self.args.model_tie_lr_weight_decay:
                decay = (lr * wd).unsqueeze(1)  # [bs,1]
            else:
                decay = wd.unsqueeze(1)  # [bs,1]

            dw_out = dw_out - decay * w_out
            w_out = w_out + dw_out

            # lr update
            lr_mult = lr_mult * torch.exp(-(torch.exp(self.log_learning_rate_decay) * norms))
            lr = lr0 * lr_mult

            a_means.append(a_mean)

        return a_means
