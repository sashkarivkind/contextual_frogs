
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
            "enable_sigma_b_tuning": True,
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
        
        if args.apply_lr_decay:
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
        self.sigma_b = nn.Parameter(init_sigma_b) if args.enable_sigma_b_tuning else torch.full((self.bs,), 0.1, device=device, requires_grad=False)

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
            elif self.args.injection_opt == 3:
                if self.args.noise_injection_node == "x":
                    raise ValueError("injection_opt3 incompatible with noise_injection_node 'x' yet")
                x_prime = u * self.u_feedback_scale + e 
                inj_col = inj_
                x_bar = (1.0 - inj_col) * x + inj_col * x_prime
                h_bar = self.better_relu(
                    biases
                    + (x_bar.unsqueeze(1) * w_in.unsqueeze(0))
                    + scaled_q_in
                )
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
            if self.args.apply_lr_decay:
                lr_mult = lr_mult * torch.exp(-(torch.exp(self.log_learning_rate_decay) * norms))
                lr = lr0 * lr_mult

            a_means.append(a_mean)

        return a_means


# class BatchedElboGenerativeModelTopMulti(nn.Module):
#     """
#     Same logic as ElboGenerativeModelTop, but *all* learnable scalars become vectors of shape [bs].
#     Each batch entry therefore has its own:
#       - log_learning_rate, log_learning_rate_decay
#       - sigma_b, sp_weight_decay, output_scale, u_feedback_scale, q_scale, ...
#       - (optional) sigma_a, sigma_x
#       - (optional) tau* and direct_injection_scale

#     Multirate extension:
#       - output synapses are split into m subpopulations per neuron:
#         w_out: [bs, n, m]
#       - learning rates and decays are per subpopulation:
#         log_learning_rate: [bs, m]
#         sp_weight_decay:  [bs, m]
#     """  #CHANGED

#     def __init__(self, device: torch.device, args=None, fudge=1e-4, batch_size: Optional[int] = None):
#         super().__init__()

#         # --- set default args ---
#         default_args = {
#             "injection_opt": 1,
#             "skip_gain": 0.0,
#             "channel_trial_extra_error": 0.0,
#             "multirate_m": 1,  
#             'lr_min_mult': 0.25,  #CHANGED: number of synapse subpopulations (m)
#             "weight_decay_mode": "softplus",  #CHANGED: new argument for weight decay mode
#             "nl_activation": "relu",  #CHANGED: new argument for nonlinearity
#             "disable_lpfs": False,  #CHANGED: new argument to disable LPFs
#             "direct_inj_limiter": 1.0,  #CHANGED: new argument for direct injection limiter
#             "lr_bound": None,  #CHANGED: new argument for learning rate bound
#             "enable_sigma_b_tuning": True,  #CHANGED: whether to make sigma_b learnable
#             "bound_weight_decay": False,  #CHANGED: whether to apply a bound to weight decay similar to lr_bound
#             "enable_weight_decay_exp": False,  #CHANGED: whether to enable weight_decay_exp 
#             "enable_weight_learning_exp": False,  #CHANGED: whether to enable weight_learning_exp
#             "enable_bias_update": False, 
#             'develop_b_tgt': 0.0, 
#             'enable_w_in_plasticity': False,
#             'x_update_mode': 'vanilla',  #CHANGED: new argument for x update mode; options: 'vanilla', 'two_lpfs'
#             'x_update_combine_mode': 'equal_mix',  #CHANGED: new argument for x update combine mode

#         }  #CHANGED
#         self.mult_activation_mode = isinstance(args.nl_activation, list)
#         self.device = device
#         for key, value in default_args.items():
#             if not hasattr(args, key):
#                 setattr(args, key, value)

#         if args.disable_lpfs:  #CHANGED
#             assert not (args.enable_qlpf or args.enable_ylpf or args.enable_elpf), "Cannot enable LPFs if disable_lpfs is set"

#         # batch size: prefer explicit, else args.bs must exist
#         if batch_size is None:
#             if not hasattr(args, "bs"):
#                 raise ValueError("Provide batch_size or set args.bs")
#             batch_size = int(args.bs)
#         self.bs = int(batch_size)

#         self.m = int(args.multirate_m)  #CHANGED

#         # --- randomize initial parameters (now per batch entry) ---
#         # use torch.rand to stay on-device
#         def randu(shape, low, high):
#             return low + (high - low) * torch.rand(shape, device=device)

#         # ranges mirror your original scalar init
#         if self.m == 1:
#             init_log_learning_rate = (
#                 randu((self.bs, self.m), -11.0, -5.)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, self.m), -6.0, device=device)  #CHANGED
#             )
#         elif self.m == 2: #concatenate two inits for slow and fast subpopulations
#             init_log_learning_rate_slow = (
#                 randu((self.bs, 1), -11.0, -7.)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, 1), -6.0, device=device)  #CHANGED
#             )
#             init_log_learning_rate_fast = (
#                 randu((self.bs, 1), -8.0, -5.)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, 1), -6.0, device=device)  #CHANGED
#             )
            
#             init_log_learning_rate = torch.cat([init_log_learning_rate_slow, init_log_learning_rate_fast], dim=1)  # [bs, m]
#         else:
#             init_log_learning_rate = (
#                 randu((self.bs, self.m), -11.0, -5.)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, self.m), -6.0, device=device)  #CHANGED
#             )

#         #CHANGED: always define init_log_learning_rate_decay so args.apply_lr_decay=False is safe
#         if args.apply_lr_decay:
#             init_log_learning_rate_decay = (
#                 randu((self.bs, self.m), -1.0, 1.0)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, self.m), 0.0, device=device)  #CHANGED
#             )
#         else:
#             init_log_learning_rate_decay = torch.zeros((self.bs, self.m), device=device)  #CHANGED

#         init_sigma_b = randu((self.bs,), 0.05, 0.55) if not args.zzz_legacy_init else torch.full((self.bs,), 0.1, device=device)
#         init_output_scale = randu((self.bs,), 0.8, 1.0) if not args.zzz_legacy_init else torch.full((self.bs,), 1.0, device=device)

#         #CHANGED: weight decay is now per subpopulation [bs, m]
#         if args.model_tie_lr_weight_decay:
#             init_sp_weight_decay = (
#                 randu((self.bs, self.m), -5.0, 5.0)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, self.m), -4.0, device=device)  #CHANGED
#             )
#         else:
#             # init_sp_weight_decay = randu((self.bs,), -8.0, -6.0) if not args.zzz_legacy_init else torch.full((self.bs,), -7.0, device=device)

#             init_sp_weight_decay = (
#                 randu((self.bs, self.m), -6.0, 2.0)  #CHANGED
#                 if not args.zzz_legacy_init
#                 else torch.full((self.bs, self.m), -7.0, device=device)  #CHANGED
#             )

#         init_sigma_a = randu((self.bs,), 0.02, 0.12) if not args.zzz_legacy_init else torch.full((self.bs,), 0.1, device=device)
#         init_sigma_x = randu((self.bs,), 0.02, 0.12) if not args.zzz_legacy_init else torch.full((self.bs,), 0.1, device=device)

#         optimize_noises = (not args.model == "toy") or getattr(args, "optimize_toy_noises", False)

#         # --- per-batch parameters ---
#         self.log_learning_rate = nn.Parameter(init_log_learning_rate)  #CHANGED: [bs, m]

#         #CHANGED: only make this a Parameter if enabled; otherwise keep fixed zeros
#         if args.apply_lr_decay:
#             self.log_learning_rate_decay = nn.Parameter(init_log_learning_rate_decay)  #CHANGED: [bs, m]
#         else:
#             self.log_learning_rate_decay = torch.zeros((self.bs, self.m), device=device, requires_grad=False)  #CHANGED

#         self.sigma_b = nn.Parameter(init_sigma_b) if args.enable_sigma_b_tuning else torch.full((self.bs,), 0.1, device=device, requires_grad=False)

#         if args.enable_output_scale_tuning:
#             self.output_scale = nn.Parameter(init_output_scale)
#         else:
#             self.output_scale = torch.ones(self.bs, device=device, requires_grad=False)

#         if args.enable_u_feedback_scale_tuning:
#             self.u_feedback_scale = nn.Parameter(torch.ones(self.bs, device=device))
#         else:
#             self.u_feedback_scale = torch.ones(self.bs, device=device, requires_grad=False)

#         self.sp_weight_decay = nn.Parameter(init_sp_weight_decay)  #CHANGED: [bs, m]

#         if args.enable_weight_decay_exp:
#             self.weight_decay_exp = nn.Parameter(torch.full((self.bs, self.m), 1.0, device=device))
#         else:
#             self.weight_decay_exp = torch.ones((self.bs, self.m), device=device, requires_grad=False)

#         if args.enable_weight_learning_exp:
#             self.weight_learning_exp = nn.Parameter(torch.full((self.bs, self.m), 1.0, device=device))
#         else:
#             self.weight_learning_exp = torch.ones((self.bs, self.m), device=device, requires_grad=False)

#         if args.enable_q_scale_tuning:
#             self.q_scale = nn.Parameter(randu((self.bs,), 0.3, 1.5))
#         else:
#             self.q_scale = torch.ones(self.bs, device=device, requires_grad=False)

#         if not args.assume_opt_output_noise:
#             if optimize_noises:
#                 self.sigma_a = nn.Parameter(init_sigma_a)
#             else:
#                 self.sigma_a = torch.full((self.bs,), float(args.toymodel_OUsigma_obs), device=device, requires_grad=False)

#         if optimize_noises:
#             self.sigma_x = nn.Parameter(init_sigma_x)
#         else:
#             self.sigma_x = torch.full((self.bs,), float(args.toymodel_OUsigma_process), device=device, requires_grad=False)

#         if args.enable_w_in_plasticity:
#             self.w_in_lr = nn.Parameter(randu((self.bs,), 0.001, 0.1))  # [bs]
#             self.w_in_decay = nn.Parameter(randu((self.bs,), 0.0, 0.99))  # [bs]


#         self.bias_lr = nn.Parameter(torch.full((self.bs,), 0.0, device=device)) if args.enable_bias_update else torch.zeros(self.bs, device=device, requires_grad=False)

#         # filters / injection (per batch)
#         self.tauqlpf_m1 = nn.Parameter(torch.full((self.bs,), -1.0, device=device)) if args.enable_qlpf else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
#         self.tauylpf_m1 = nn.Parameter(torch.full((self.bs,), -1.0, device=device)) if args.enable_ylpf else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
#         self.tauelpf_m1 = nn.Parameter(torch.full((self.bs,),  1.0, device=device)) if args.enable_elpf else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)

#         self.direct_injection_scale = nn.Parameter(randu((self.bs,), 0.05, 0.4)) if args.enable_direct_injection else torch.zeros(self.bs, device=device, requires_grad=False)

#         # --- random features (shared across batch, as before) ---
#         self.register_buffer("_z_biases", torch.empty(0))  # base N(0,1) for biases, shape [n]
#         self.register_buffer("_w_in", torch.empty(0))      # random input features, shape [n]
#         self.register_buffer("_w_inq", torch.empty(0))     # random input features for q, shape [n]

#         if args.x_update_mode == 'two_lpfs':
#             self.x_fast_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))  # [bs]
#             self.x_slow_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))  # [bs]
#         elif args.x_update_mode == 'consolidate_to_slow':
#             self.x_slow_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))  # [bs]
#             self.x_fast_gain = nn.Parameter(torch.full((self.bs,), 0.5, device=device))  # [bs]
#         elif args.x_update_mode == 'u_only_lpf':
#             self.x_slow_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))  # [bs]

#         self.args = args
#         self.fudge = fudge

#     # @staticmethod
#     def phi(self,x, nl_activation=None):
#         if self.mult_activation_mode and nl_activation is None:
#             # create a [B,n,li] where li is the number of activations in the list, then combine across that dimension with the specified weights
#             activations = []
#             for act in self.args.nl_activation:
#                 #call recursively to get each activation
#                 act_fn = BatchedElboGenerativeModelTopMulti.phi.__get__(self)  # get the method bound to self
#                 activations.append(act_fn(x, nl_activation=act))  # [B,n]
#             activations = torch.stack(activations, dim=-1)  # [B,n,li]
#             return activations
#         nl_activation = self.args.nl_activation if nl_activation is None else nl_activation
#         if nl_activation == 'relu':
#             return F.relu(x)
#         elif nl_activation == 'rescaled_sigmoid':
#             return torch.sigmoid(4*x - 2) 
#         elif nl_activation == 'const':
#             return torch.ones_like(x)
#         else:
#             raise ValueError(f"Unknown nl_activation {nl_activation}")

#     @staticmethod
#     def softplus(x: torch.Tensor) -> torch.Tensor:
#         # keep your original definition
#         return torch.log1p(torch.exp(x))

#     def _ensure_random_features(self, n: int, device):
#         if self._w_in.numel() != n:
#             self.register_buffer("_z_biases", torch.randn(n, device=device))
#             self.register_buffer("_w_in", torch.randn(n, device=device))
#             self.register_buffer("_w_inq", torch.randn(n, device=device))

#     def get_biases_and_w_in(self, n: int, device):
#         self._ensure_random_features(n, device)
#         # per-batch sigma_b -> per-batch biases vector
#         # biases: [bs, n]
#         biases = self.sigma_b.unsqueeze(1) * self._z_biases.unsqueeze(0)
#         return biases, self._w_in  # w_in: [n]

#     def get_winq(self, n: int, device):
#         self._ensure_random_features(n, device)
#         return self._w_inq  # [n]

#     def _expand_time_input(self, tval: torch.Tensor, bs: int) -> torch.Tensor:
#         """
#         Make sure per-time tensors are shape [bs].
#         Accepts scalar [], [1], or [bs].
#         """
#         if tval is None:
#             return torch.zeros(bs, device=self.log_learning_rate.device)
#         if tval.ndim == 0:
#             return tval.expand(bs)
#         if tval.numel() == 1 and tval.shape[0] == 1:
#             return tval.expand(bs)
#         if tval.shape[0] != bs:
#             raise ValueError(f"Expected time-slice with shape [bs]={bs}, got {tuple(tval.shape)}")
#         return tval

#     def _assign_wd(self, lr0, n):
#         if self.args.weight_decay_mode == 'softplus':
#             wd = self.softplus(self.sp_weight_decay)  #CHANGED: [bs,m]
#         elif self.args.weight_decay_mode == 'sigmoid':
#             wd = self.args.lr_min_mult * lr0 * n *torch.nn.functional.sigmoid(self.sp_weight_decay)  #CHANGED: [bs,m]  #CHANGED: scale by lr*n as in original code
#         elif self.args.weight_decay_mode == 'clipped_sigmoid':
#             wd = self.args.weight_decay_max*torch.nn.functional.sigmoid(self.sp_weight_decay)  #CHANGED: [bs,m]  #CHANGED: scale by lr*n as in original code
#         else:
#             raise ValueError(f"Unknown weight_decay_mode {self.args.weight_decay_mode}")
        
#         if self.args.bound_weight_decay:
#             wd_bound = 1.0 - lr0*self.args.n  # [bs,m]
#             #softly bound wd to be less than wd_bound
#             wd = wd_bound * torch.tanh(wd / wd_bound)  # [bs,m]
#         return wd
    
#     def _init_lr(self,bs):
#         lr_mult = torch.ones(bs, self.m, device=self.device)  #CHANGED: [bs,m]
#         lr0 = torch.exp(self.log_learning_rate)  #CHANGED: [bs,m]
#         if self.args.lr_bound is not None:
#             total_lr0 = torch.einsum("km->k", lr0)  # [bs]
#             target_total_lr0 = self.args.lr_bound*torch.tanh(total_lr0 / self.args.lr_bound)  # [bs], smoothly approaches lr_bound for large total_lr0
#             lr0 = lr0 * (target_total_lr0 / total_lr0).unsqueeze(1)  
#             self.debug_lr0 = lr0  # for debugging: track the pre-multiplier learning rates after bounding
#         lr = lr0 * lr_mult  #CHANGED: [bs,m]
#         return lr, lr0, lr_mult
    
#     def _prep_inputs(self, y, noise_x, q, bs):
#             # y handling
#             if y is None:
#                 y = torch.full((bs,), np.double("nan"), device=self.device)
#             else:
#                 y = self._expand_time_input(y, bs)

#             q = self._expand_time_input(q if q is not None else torch.zeros((1,), device=self.device), bs)
#             noise_x = self._expand_time_input(noise_x, bs)
#             return y, noise_x, q
    
#     def _lowpass_filter(self, new_value, prev_lp, tau, enable):
#         if enable:
#             return (1.0 - 1.0 / tau) * prev_lp + (1.0 / tau) * new_value
#         else:
#             return new_value
    
#     def _calculate_h_bar(self, h, biases_, x, w_in_, u, e, scaled_q_in, inj_, mode=None, x_state=None):
#             # injection_opt == 2: leak perturbation into hidden
#             if mode == 2:
#                 if self.args.noise_injection_node == "x":
#                     raise ValueError("injection_opt2 incompatible with noise_injection_node 'x' yet")
#                 # x_prime = u * self.u_feedback_scale + e
#                 x_prime, _ = self._update_x(x, u, e, x_state)
#                 h_prime = self.phi(
#                     biases_
#                     + (x_prime.unsqueeze(1) * w_in_)
#                     + scaled_q_in
#                 )
#                 inj_col = inj_.unsqueeze(1)  # [bs,1]
#                 h_bar = (1.0 - inj_col) * h + inj_col * h_prime
#             elif mode == 3:
#                 if self.args.noise_injection_node == "x":
#                     raise ValueError("injection_opt3 incompatible with noise_injection_node 'x' yet")
#                 x_prime, _ = self._update_x(x, u, e, x_state)
#                 inj_col = inj_
#                 x_bar = (1.0 - inj_col) * x + inj_col * x_prime
#                 h_bar = self.phi(
#                     biases_
#                     + (x_bar.unsqueeze(1) * w_in_)
#                     + scaled_q_in
#                 )
#             elif mode == 0:
#                 h_bar = h
#             else:
#                 raise ValueError(f"Unknown injection_opt mode {mode}")
#             return h_bar
    
#     def _compute_scaled_q_in(self, prescaled_w_inq, qlp):
#             #scaled q input: [bs,n]
#             if prescaled_w_inq is not None:
#                 q_gain = (self.q_scale * qlp).unsqueeze(1)  # [bs,1]
#                 scaled_q_in = prescaled_w_inq.unsqueeze(0) * q_gain  # [bs,n]
#             else:
#                 scaled_q_in = 0.0
#             return scaled_q_in
    
#     def _update_x(self, x, u, elp, x_state):
#             this_fbk_signal = u * self.u_feedback_scale + elp
#             if self.args.x_update_mode == 'vanilla':
#                 return this_fbk_signal, x_state
#             elif self.args.x_update_mode == 'two_lpfs':
#                 if x_state is None:
#                     raise ValueError("x_state must be provided for dual-rate x update")
#                 x_slow, x_fast = x_state
#                 x_slow_new = self.x_slow_alpha * x_slow + (1 - self.x_slow_alpha) * this_fbk_signal
#                 x_fast_new = self.x_fast_alpha * x_fast + (1 - self.x_fast_alpha) * this_fbk_signal
#                 if self.args.x_update_combine_mode == 'equal_mix':
#                     x_bar = 0.5 * x_slow_new + 0.5 * x_fast_new
#                 elif self.args.x_update_combine_mode == 'learned_mix':
#                     x_bar = self.x_slow_weight * x_slow_new + self.x_fast_weight * x_fast_new
#                 else:                    
#                     raise ValueError(f"Unknown x_update_combine_mode {self.args.x_update_combine_mode}")
#                 return x_bar, (x_slow_new, x_fast_new)
#             elif self.args.x_update_mode == 'consolidate_to_slow':
#                 if x_state is None:
#                     raise ValueError("x_state must be provided for consolidate_to_slow x update")
#                 x_slow, x_fast = x_state
#                 x_slow_new = self.x_slow_alpha * x_slow + (1 - self.x_slow_alpha) * this_fbk_signal
#                 x_fast_new = self.x_fast_gain * (this_fbk_signal - x_slow)
#                 x_bar = x_slow_new + x_fast_new
#                 return x_bar, (x_slow_new, x_fast_new)
#             elif self.args.x_update_mode == 'u_only_lpf':
#                 x_slow, x_fast = x_state

#                 x_slow_new = self.x_slow_alpha * x_slow + (1 - self.x_slow_alpha) * u * self.u_feedback_scale
#                 x_bar = elp + x_slow_new
#                 return  x_bar, (x_slow_new, None)
#             else:
#                 raise ValueError(f"Unknown x_update_mode {self.args.x_update_mode}")
            
#     def _lr_update(self, lr_mult, norms, lr0):
#         nonneg_decay_coeff = torch.exp(self.log_learning_rate_decay)  # [bs,m]
#         lr_mult = self.args.lr_min_mult + (lr_mult - self.args.lr_min_mult) * torch.exp(-(nonneg_decay_coeff * norms))
#         lr = lr0 * lr_mult
#         return lr_mult, lr
    
#     def _w_in_update(self, w_in_tuning, h_bar):
#         if self.mult_activation_mode:
#             raise ValueError("w_in plasticity not yet implemented for mult_activation_mode yet")
#         w_in_tuning = (1 - self.w_in_decay.unsqueeze(1)) * w_in_tuning + self.w_in_lr.unsqueeze(1) * h_bar.squeeze(2)
#         return w_in_tuning

#     def _compute_w_in(self, w_in, w_in_tuning):
#         if self.args.enable_w_in_plasticity:
#             return w_in.unsqueeze(0) + w_in_tuning * torch.sign(w_in.unsqueeze(0))
#         return w_in.unsqueeze(0)

#     def _bias_update(self, biases, bias_tuning, x, w_in_, scaled_q_in):
#         if self.args.enable_bias_update:
#             biases_ = biases + bias_tuning
#             preact = (x.unsqueeze(1) * w_in_) + scaled_q_in
#             bias_tuning = bias_tuning + (self.args.develop_b_tgt + preact - biases_) * self.bias_lr
#             return biases_, bias_tuning
#         return biases, bias_tuning

#     def f(
#         self,
#         n: int,
#         noises: List[torch.Tensor],                # list of [bs]
#         ys: List[Optional[torch.Tensor]],          # list of [bs] or None
#         model_setting: str,
#         qs: Optional[List[Optional[torch.Tensor]]] = None,  # list of [bs] or [1] or None
#     ) -> List[torch.Tensor]:
        
#         #assertions
#         assert len(noises) == len(ys), "noises and ys must have same length"
#         assert model_setting in ["default"], "model_setting must be 'default' for this model class"
#         bs = noises[0].shape[0]
#         assert bs == self.bs, f"Model was initialized with bs={self.bs}, but got input bs={bs}"
        
#         #initialisations
#         device = self.device
#         biases, w_in = self.get_biases_and_w_in(n=n, device=device)  # biases [bs,n], w_in [n]
#         prescaled_w_inq = self.get_winq(n=n, device=device) if qs is not None else None  # [n] or None

#         # state init (per batch)
#         w_out = torch.zeros(bs, n, self.m, device=device)  #CHANGED: [bs,n,m]
#         bias_tuning = torch.zeros(bs, n, device=device)  
#         u = torch.zeros(bs, device=device)
#         x = torch.zeros(bs, device=device)
#         e = torch.zeros(bs, device=device)

#         lr, lr0, lr_mult = self._init_lr(bs)
#         wd = self._assign_wd(lr0, n)

#         if qs is None:# default qs if none
#             qs = [torch.zeros((1,), device=device)] * len(ys)

#         # if not self.args.disable_lpfs:
#         qlp = torch.zeros(bs, device=device)
#         ylp = torch.zeros(bs, device=device)
#         elp = torch.zeros(bs, device=device)
#         tauqlpf = 1.0 + self.softplus(self.tauqlpf_m1)  # [bs]
#         tauylpf = 1.0 + self.softplus(self.tauylpf_m1)  # [bs]
#         tauelpf = 1.0 + self.softplus(self.tauelpf_m1)  # [bs]

#         if self.args.x_update_mode in ['two_lpfs', 'consolidate_to_slow', 'u_only_lpf']:
#             x_slow = torch.zeros(bs, device=device)
#             x_fast = torch.zeros(bs, device=device)
#             x_state = (x_slow, x_fast)
#         else:
#             x_state = None


#         inj_ = torch.nn.functional.sigmoid(self.direct_injection_scale) * self.args.direct_inj_limiter  # [bs]
        
#         if self.args.enable_w_in_plasticity:
#             w_in_tuning = torch.zeros(bs, n, device=device) + 1.0#0.5 # [bs,n]

#         #recording outputs
#         a_means: List[torch.Tensor] = []

#         for y, noise_x, q in zip(ys, noises, qs):

#             y, noise_x, q = self._prep_inputs(y, noise_x, q, bs=bs)
#             qlp = self._lowpass_filter(q, qlp, tauqlpf, enable = not self.args.disable_lpfs) 
#             scaled_q_in = self._compute_scaled_q_in(prescaled_w_inq, qlp)

#             x, x_state = self._update_x(x, u, elp, x_state)

#             w_in_ = self._compute_w_in(w_in, w_in_tuning)
#             biases_, bias_tuning = self._bias_update(biases, bias_tuning, x, w_in_, scaled_q_in)

#             mask = torch.isnan(y)  # [bs]

#             if self.args.injection_opt == 1:
#                 raise ValueError("injection_opt 1 is deprecated, use injection_opt 2 instead")

#             # hidden
#             h = self.phi(
#                 biases_
#                 + (x.unsqueeze(1) * w_in_)
#                 + scaled_q_in
#             )  # [bs,n] or [bs,n,li] if mult_activation_mode

#             #CHANGED: sum subpopulations first: w_out [bs,n,m] -> w_eff [bs,n]
#             w_eff = w_out.sum(dim=2) if not self.mult_activation_mode else w_out  #CHANGED: [bs,n]
#             summing_directive = "kj,kj->k" if not self.mult_activation_mode else "kji,kji->k"
#             # print(f"w_eff shape: {w_eff.shape}, h shape: {h.shape}, summing_directive: {summing_directive}")
#             u = torch.einsum(summing_directive, w_eff, h)  #CHANGED (same einsum, different weight tensor)
#             u = u + self.args.skip_gain * x

#             a_mean = self.output_scale * u

#             # channel trial handling
#             y_ = torch.where(mask, u + self.args.channel_trial_extra_error, y)
#             ylp = self._lowpass_filter(y_, ylp, tauylpf, enable = not self.args.disable_lpfs)

#             # error
#             e = ylp - u
#             elp = self._lowpass_filter(e, elp, tauelpf, enable = not self.args.disable_lpfs)

#             h_bar = self._calculate_h_bar(h, biases_, 
#                                           x, w_in_, u, elp, 
#                                           scaled_q_in, inj_, 
#                                           mode=self.args.injection_opt, 
#                                           x_state=x_state)
#             h_bar = h_bar.unsqueeze(2) if not self.mult_activation_mode else h_bar
            

#             if self.args.enable_weight_learning_exp:
#                 # print(f"elp before learning exp: {elp.shape}, weight_learning_exp: {self.weight_learning_exp.shape}")
#                 elp_ = torch.sign(elp.unsqueeze(1)) * torch.abs(elp.unsqueeze(1)).pow(self.softplus(self.weight_learning_exp))  #CHANGED: [bs,n,m], with separate exponent per subpopulation
#             else:
#                 elp_ = elp.unsqueeze(1)  

#             dw_out = (
#                 lr.unsqueeze(1)                     
#                 * elp_.unsqueeze(2)        
#                 * h_bar                    
#             )  

#             if self.args.apply_lr_decay:
#                 norms = torch.sqrt(self.fudge + torch.sum(dw_out ** 2, dim=1))  #CHANGED: [bs,m]

#             if self.args.model_tie_lr_weight_decay:
#                 decay = (lr * wd).unsqueeze(1)  #CHANGED: [bs,1,m]
#             else:
#                 decay = wd.unsqueeze(1)         #CHANGED: [bs,1,m]

#             if self.args.enable_weight_decay_exp:
#                 w_out_ = torch.sign(w_out) * torch.abs(w_out).pow(self.weight_decay_exp.unsqueeze(1))  #CHANGED: [bs,n,m], with separate exponent per subpopulation
#             else:
#                 w_out_ = w_out  #CHANGED: [bs,n,m]
#             dw_out = dw_out - decay * w_out_  #CHANGED
#             w_out = w_out + dw_out           #CHANGED


#             if self.args.enable_w_in_plasticity:
#                 w_in_tuning = self._w_in_update(w_in_tuning, h_bar)
           
#             if self.args.apply_lr_decay:
#                 lr_mult, lr = self._lr_update(lr_mult, norms, lr0)
            
#             # print(f"lr: {lr}, wd: {wd}, inj_: {inj_}, lr_mult: {lr_mult}, lr0: {lr0}, norms: {norms}")  # for debugging
#             a_means.append(a_mean)

#         return a_means
