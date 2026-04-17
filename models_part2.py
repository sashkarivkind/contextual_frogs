import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class BatchedElboGenerativeModelTop(nn.Module):
    def __init__(self,*args, **kwargs):
            raise NotImplementedError("This class is now deprecated in favor of BatchedElboGenerativeModelTopMulti, for old implementations check legacy_stuff.py.")


class BatchedElboGenerativeModelTopMulti(nn.Module):

    def __init__(self, device: torch.device, args=None, fudge=1e-4, batch_size: Optional[int] = None):
        super().__init__()

        default_args = {
            "injection_opt": 1,
            "skip_gain": 0.0,
            "channel_trial_extra_error": 0.0,
            "multirate_m": 1,
            "lr_min_mult": 0.25,
            "weight_decay_mode": "softplus",
            "nl_activation": "relu",
            "disable_lpfs": False,
            "direct_inj_limiter": 1.0,
            "lr_bound": None,
            "enable_sigma_b_tuning": True,
            "bound_weight_decay": False,
            "enable_weight_decay_exp": False,
            "enable_weight_learning_exp": False,
            "enable_bias_update": False,
            "develop_b_tgt": 0.0,
            "enable_w_in_plasticity": False,
            "x_update_mode": "vanilla",
            "x_update_combine_mode": "equal_mix",

            # NEW
            "enable_separate_win_per_rate": False,
            "enable_w_in_scale_tuning": False,
            "manual_w_in_scale": None,
            "enforce_positive_biases": False,
            "initiate_w_in_tuning_with_steady_state_vals": False,
            "debug_flag_win2nd_column_positive_only": False, 
            "apply_scaled_soft_plus_on_w_in_params": False,
            "lr_update_mode": "basic",
            "lr_recovery_rate": 0.0, 
            "lr_update_qty": "dw_out_norm" ,
            "enable_output_scale_tuning": False,
            "enable_input_scale_tuning": False,

                    }

        self.device = device
        self.mult_activation_mode = isinstance(args.nl_activation, list)

        for key, value in default_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)

        if args.disable_lpfs:
            assert not (args.enable_qlpf or args.enable_ylpf or args.enable_elpf), \
                "Cannot enable LPFs if disable_lpfs is set"

        if batch_size is None:
            if not hasattr(args, "bs"):
                raise ValueError("Provide batch_size or set args.bs")
            batch_size = int(args.bs)

        self.bs = int(batch_size)
        self.m = int(args.multirate_m)
        self.args = args
        self.fudge = fudge
        self.win_nl = (lambda x: x) if not args.enable_w_in_plasticity else lambda x: F.tanh(x)
        def randu(shape, low, high):
            return low + (high - low) * torch.rand(shape, device=device)

        # ---------------------------
        # scalar / per-rate parameters
        # ---------------------------

        #leraning rates were set for n=128. For different n the initialisation 
        #should be adjusted
        size_fac = np.log(128./args.n)
        if self.m == 1:
            init_log_learning_rate = (
                randu((self.bs, self.m), -11.0, -5.0) + size_fac
                if not args.zzz_legacy_init
                else torch.full((self.bs, self.m), -6.0, device=device)
            )
        elif self.m == 2:
            init_log_learning_rate_slow = (
                randu((self.bs, 1), -11.0, -7.0) + size_fac
                if not args.zzz_legacy_init
                else torch.full((self.bs, 1), -6.0, device=device)
            )
            init_log_learning_rate_fast = (
                randu((self.bs, 1), -8.0, -5.0) + size_fac
                if not args.zzz_legacy_init
                else torch.full((self.bs, 1), -6.0, device=device)
            )
            init_log_learning_rate = torch.cat(
                [init_log_learning_rate_slow, init_log_learning_rate_fast], dim=1
            )
        else:
            init_log_learning_rate = (
                randu((self.bs, self.m), -11.0, -5.0) + size_fac
                if not args.zzz_legacy_init
                else torch.full((self.bs, self.m), -6.0, device=device)
            )

        if args.apply_lr_decay:
            init_log_learning_rate_decay = (
                randu((self.bs, self.m), -1.0, 1.0)
                if not args.zzz_legacy_init
                else torch.full((self.bs, self.m), 0.0, device=device)
            )
        else:
            init_log_learning_rate_decay = torch.zeros((self.bs, self.m), device=device)

        init_sigma_b = (
            randu((self.bs,), 0.05, 0.55)
            if not args.zzz_legacy_init
            else torch.full((self.bs,), 0.1, device=device)
        )
        init_output_scale = (
            randu((self.bs,), 0.8, 1.0)
            if not args.zzz_legacy_init
            else torch.full((self.bs,), 1.0, device=device)
        )

        init_input_scale = (
            randu((self.bs,), 0.8, 1.0)
            if not args.zzz_legacy_init
            else torch.full((self.bs,), 1.0, device=device)
        )

        if args.model_tie_lr_weight_decay:
            init_sp_weight_decay = (
                randu((self.bs, self.m), -5.0, 5.0)
                if not args.zzz_legacy_init
                else torch.full((self.bs, self.m), -4.0, device=device)
            )
        else:
            init_sp_weight_decay = (
                randu((self.bs, self.m), -6.0, 2.0)
                if not args.zzz_legacy_init
                else torch.full((self.bs, self.m), -7.0, device=device)
            )

        init_sigma_a = (
            randu((self.bs,), 0.02, 0.12)
            if not args.zzz_legacy_init
            else torch.full((self.bs,), 0.1, device=device)
        )
        init_sigma_x = (
            randu((self.bs,), 0.02, 0.12)
            if not args.zzz_legacy_init
            else torch.full((self.bs,), 0.1, device=device)
        )

        optimize_noises = (not args.model == "toy") or getattr(args, "optimize_toy_noises", False)

        self.log_learning_rate = nn.Parameter(init_log_learning_rate)

        if args.apply_lr_decay:
            self.log_learning_rate_decay = nn.Parameter(init_log_learning_rate_decay)
        else:
            self.log_learning_rate_decay = torch.zeros(
                (self.bs, self.m), device=device, requires_grad=False
            )

        self.sigma_b = (
            nn.Parameter(init_sigma_b)
            if args.enable_sigma_b_tuning
            else torch.full((self.bs,), 0.1, device=device, requires_grad=False)
        )

        # scale of input weights added to explore input weight plastisity
        w_in_scale_shape = (self.m,) if self._separate_win_mode() else (1,)
        self.w_in_scale = (
            nn.Parameter(randu(w_in_scale_shape, 0.5, 2.0))    
            if args.enable_w_in_scale_tuning
            else torch.ones(w_in_scale_shape, device=device, requires_grad=False) * (args.manual_w_in_scale if args.manual_w_in_scale is not None else 1.0)
        )

        if args.enable_output_scale_tuning:
            self.output_scale = nn.Parameter(init_output_scale)
        else:
            self.output_scale = torch.ones(self.bs, device=device, requires_grad=False)

        if args.enable_input_scale_tuning:
            self.input_scale = nn.Parameter(init_input_scale)
        else:
            self.input_scale = torch.ones(self.bs, device=device, requires_grad=False)

        if args.enable_u_feedback_scale_tuning:
            self.u_feedback_scale = nn.Parameter(torch.ones(self.bs, device=device))
        else:
            self.u_feedback_scale = torch.ones(self.bs, device=device, requires_grad=False)

        self.sp_weight_decay = nn.Parameter(init_sp_weight_decay)

        if args.enable_weight_decay_exp:
            self.weight_decay_exp = nn.Parameter(torch.full((self.bs, self.m), 1.0, device=device))
        else:
            self.weight_decay_exp = torch.ones((self.bs, self.m), device=device, requires_grad=False)

        if args.enable_weight_learning_exp:
            self.weight_learning_exp = nn.Parameter(torch.full((self.bs, self.m), 1.0, device=device))
        else:
            self.weight_learning_exp = torch.ones((self.bs, self.m), device=device, requires_grad=False)

        if args.enable_q_scale_tuning:
            self.q_scale = nn.Parameter(randu((self.bs,), 0.3, 1.5))
        else:
            self.q_scale = torch.ones(self.bs, device=device, requires_grad=False)

        if not args.assume_opt_output_noise:
            if optimize_noises:
                self.sigma_a = nn.Parameter(init_sigma_a)
            else:
                self.sigma_a = torch.full(
                    (self.bs,),
                    float(args.toymodel_OUsigma_obs),
                    device=device,
                    requires_grad=False,
                )

        if optimize_noises:
            self.sigma_x = nn.Parameter(init_sigma_x)
        else:
            self.sigma_x = torch.full(
                (self.bs,),
                float(args.toymodel_OUsigma_process),
                device=device,
                requires_grad=False,
            )

        self._init_w_in_plasticity_params(randu, device)

        self.bias_lr = (
            nn.Parameter(torch.full((self.bs,), 0.0, device=device))
            if args.enable_bias_update
            else torch.zeros(self.bs, device=device, requires_grad=False)
        )

        self.tauqlpf_m1 = (
            nn.Parameter(torch.full((self.bs,), -1.0, device=device))
            if args.enable_qlpf
            else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
        )
        self.tauylpf_m1 = (
            nn.Parameter(torch.full((self.bs,), -1.0, device=device))
            if args.enable_ylpf
            else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
        )
        self.tauelpf_m1 = (
            nn.Parameter(torch.full((self.bs,), 1.0, device=device))
            if args.enable_elpf
            else torch.full((self.bs,), -1000.0, device=device, requires_grad=False)
        )

        self.direct_injection_scale = (
            nn.Parameter(randu((self.bs,), 0.05, 0.4))
            if args.enable_direct_injection
            else torch.zeros(self.bs, device=device, requires_grad=False)
        )

        # random features
        self.register_buffer("_z_biases", torch.empty(0))
        self.register_buffer("_w_in", torch.empty(0))
        self.register_buffer("_w_inq", torch.empty(0))

        if args.x_update_mode == "two_lpfs":
            self.x_fast_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))
            self.x_slow_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))
        elif args.x_update_mode == "consolidate_to_slow":
            self.x_slow_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))
            self.x_fast_gain = nn.Parameter(torch.full((self.bs,), 0.5, device=device))
        elif args.x_update_mode == "u_only_lpf":
            self.x_slow_alpha = nn.Parameter(torch.full((self.bs,), 0.5, device=device))

    # ---------------------------
    # mode helpers
    # ---------------------------

    def _separate_win_mode(self) -> bool:
        return bool(getattr(self.args, "enable_separate_win_per_rate", False))

    def _hidden_has_rate_axis(self) -> bool:
        return self._separate_win_mode()

    def _broadcast_x_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, None, None] if self._hidden_has_rate_axis() else x[:, None]

    def _broadcast_inj_to_hidden(self, inj: torch.Tensor) -> torch.Tensor:
        return inj[:, None, None] if self._hidden_has_rate_axis() else inj[:, None]

    # ---------------------------
    # activation / utilities
    # ---------------------------

    def phi(self, x, nl_activation=None):
        if self.mult_activation_mode and nl_activation is None:
            activations = []
            for act in self.args.nl_activation:
                act_fn = BatchedElboGenerativeModelTopMulti.phi.__get__(self)
                activations.append(act_fn(x, nl_activation=act))
            return torch.stack(activations, dim=-1)

        nl_activation = self.args.nl_activation if nl_activation is None else nl_activation
        if nl_activation == "relu":
            return F.relu(x)
        elif nl_activation == "rescaled_sigmoid":
            return torch.sigmoid(4 * x - 2)
        elif nl_activation == "const":
            return torch.ones_like(x)
        else:
            raise ValueError(f"Unknown nl_activation {nl_activation}")

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x))

    # ---------------------------
    # feature / parameter init helpers
    # ---------------------------

    def _init_w_in_plasticity_params(self, randu, device):
        if self._separate_win_mode():
            shape = (self.bs, self.m)
        else:
            # store as [bs,1] rather than [bs] for easier broadcasting
            shape = (self.bs, 1)

        if self.args.enable_w_in_plasticity:
            if not self.args.apply_scaled_soft_plus_on_w_in_params:
                self.w_in_lr_ = nn.Parameter(randu(shape, 0.001, 0.05))
                self.w_in_decay_ = nn.Parameter(randu(shape, 0.0, 0.02))
            else:
                self.scsp_w_in_lr_ = nn.Parameter(randu(shape, 0, 5.))
                self.scsp_w_in_decay_ = nn.Parameter(randu(shape, 0, 2.))

        # else:
        #     self.w_in_lr_ = torch.zeros(shape, device=device, requires_grad=False)
        #     self.w_in_decay_ = torch.zeros(shape, device=device, requires_grad=False)

    def _ensure_random_features(self, n: int, device):
        if self._separate_win_mode():
            need = n * self.m
            if self._w_in.numel() != need:
                self.register_buffer("_z_biases", torch.randn(n, self.m, device=device))
                self.register_buffer("_w_in", torch.randn(n, self.m, device=device))
                self.register_buffer("_w_inq", torch.randn(n, self.m, device=device))
        else:
            if self._w_in.numel() != n:
                self.register_buffer("_z_biases", torch.randn(n, device=device))
                self.register_buffer("_w_in", torch.randn(n, device=device))
                self.register_buffer("_w_inq", torch.randn(n, device=device))

    def get_biases_and_w_in(self, n: int, device):
        self._ensure_random_features(n, device)
        if self._separate_win_mode():
            biases = self.sigma_b[:, None, None] * self._z_biases[None, :, :]
        else:
            biases = self.sigma_b[:, None] * self._z_biases[None, :]
        if self.args.enforce_positive_biases:
            biases = torch.abs(biases)
        
        w_in = self.w_in_scale *self._w_in
        if self.args.debug_flag_win2nd_column_positive_only:
            if self._separate_win_mode() and self.m >= 2:
                w_in = w_in.clone()
                w_in[:, 1] = torch.clamp(w_in[:, 1], min=0.0)
            else:
                raise ValueError("debug_flag_win2nd_column_positive_only only compatible with separate_win_per_rate and m>=2")
        return biases, w_in

    def get_winq(self, n: int, device):
        self._ensure_random_features(n, device)
        return self._w_inq

    def _init_runtime_state(self, bs: int, n: int, device):
        state = {}
        state["w_out"] = torch.zeros(bs, n, self.m, device=device)

        if self._separate_win_mode():
            state["bias_tuning"] = torch.zeros(bs, n, self.m, device=device)
        else:
            state["bias_tuning"] = torch.zeros(bs, n, device=device)

        state["u"] = torch.zeros(bs, device=device)
        state["x"] = torch.zeros(bs, device=device)
        state["e"] = torch.zeros(bs, device=device)

        state["qlp"] = torch.zeros(bs, device=device)
        state["ylp"] = torch.zeros(bs, device=device)
        state["elp"] = torch.zeros(bs, device=device)

        if self.args.enable_w_in_plasticity:
            if self._separate_win_mode():
                state["w_in_tuning"] = torch.zeros(bs, n, self.m, device=device)                                       
            else:
                state["w_in_tuning"] = torch.zeros(bs, n, device=device)
        else:
            state["w_in_tuning"] = None

        if self.args.x_update_mode in ["two_lpfs", "consolidate_to_slow", "u_only_lpf"]:
            state["x_state"] = (torch.zeros(bs, device=device), torch.zeros(bs, device=device))
        else:
            state["x_state"] = None

        return state

    # ---------------------------
    # unchanged utility methods
    # ---------------------------

    def _expand_time_input(self, tval: torch.Tensor, bs: int) -> torch.Tensor:
        if tval is None:
            return torch.zeros(bs, device=self.log_learning_rate.device)
        if tval.ndim == 0:
            return tval.expand(bs)
        if tval.numel() == 1 and tval.shape[0] == 1:
            return tval.expand(bs)
        if tval.shape[0] != bs:
            raise ValueError(f"Expected time-slice with shape [bs]={bs}, got {tuple(tval.shape)}")
        return tval

    def _assign_wd(self, lr0, n):
        if self.args.weight_decay_mode == "softplus":
            wd = self.softplus(self.sp_weight_decay)
        elif self.args.weight_decay_mode == "sigmoid":
            wd = self.args.lr_min_mult * lr0 * n * torch.sigmoid(self.sp_weight_decay)
        elif self.args.weight_decay_mode == "clipped_sigmoid":
            wd = self.args.weight_decay_max * torch.sigmoid(self.sp_weight_decay)
        else:
            raise ValueError(f"Unknown weight_decay_mode {self.args.weight_decay_mode}")

        if self.args.bound_weight_decay:
            wd_bound = 1.0 - lr0 * self.args.n
            wd = wd_bound * torch.tanh(wd / wd_bound)
        return wd

    def _init_lr(self, bs):
        lr_mult = torch.ones(bs, self.m, device=self.device)
        lr0 = torch.exp(self.log_learning_rate)
        if self.args.lr_bound is not None:
            total_lr0 = torch.einsum("km->k", lr0)
            target_total_lr0 = self.args.lr_bound * torch.tanh(total_lr0 / self.args.lr_bound)
            lr0 = lr0 * (target_total_lr0 / total_lr0).unsqueeze(1)
            self.debug_lr0 = lr0
        lr = lr0 * lr_mult
        return lr, lr0, lr_mult

    def _prep_inputs(self, y, noise_x, q, bs):
        if y is None:
            y = torch.full((bs,), np.double("nan"), device=self.device)
        else:
            y = self._expand_time_input(y, bs)

        q = self._expand_time_input(
            q if q is not None else torch.zeros((1,), device=self.device), bs
        )
        noise_x = self._expand_time_input(noise_x, bs)
        return y, noise_x, q

    def _lowpass_filter(self, new_value, prev_lp, tau, enable):
        if enable:
            return (1.0 - 1.0 / tau) * prev_lp + (1.0 / tau) * new_value
        else:
            return new_value

    def _update_x(self, x, u, elp, x_state):
        this_fbk_signal = u * self.u_feedback_scale + elp
        if self.args.x_update_mode == "vanilla":
            return this_fbk_signal, x_state
        elif self.args.x_update_mode == "two_lpfs":
            if x_state is None:
                raise ValueError("x_state must be provided for dual-rate x update")
            x_slow, x_fast = x_state
            x_slow_new = self.x_slow_alpha * x_slow + (1 - self.x_slow_alpha) * this_fbk_signal
            x_fast_new = self.x_fast_alpha * x_fast + (1 - self.x_fast_alpha) * this_fbk_signal
            if self.args.x_update_combine_mode == "equal_mix":
                x_bar = 0.5 * x_slow_new + 0.5 * x_fast_new
            elif self.args.x_update_combine_mode == "learned_mix":
                x_bar = self.x_slow_weight * x_slow_new + self.x_fast_weight * x_fast_new
            else:
                raise ValueError(f"Unknown x_update_combine_mode {self.args.x_update_combine_mode}")
            return x_bar, (x_slow_new, x_fast_new)
        elif self.args.x_update_mode == "consolidate_to_slow":
            if x_state is None:
                raise ValueError("x_state must be provided for consolidate_to_slow x update")
            x_slow, x_fast = x_state
            x_slow_new = self.x_slow_alpha * x_slow + (1 - self.x_slow_alpha) * this_fbk_signal
            x_fast_new = self.x_fast_gain * (this_fbk_signal - x_slow)
            x_bar = x_slow_new + x_fast_new
            return x_bar, (x_slow_new, x_fast_new)
        elif self.args.x_update_mode == "u_only_lpf":
            x_slow, x_fast = x_state
            x_slow_new = self.x_slow_alpha * x_slow + (1 - self.x_slow_alpha) * u * self.u_feedback_scale
            x_bar = elp + x_slow_new
            return x_bar, (x_slow_new, None)
        else:
            raise ValueError(f"Unknown x_update_mode {self.args.x_update_mode}")

    def _lr_update(self, lr_mult, lr0, weight_info=None):

        if self.args.lr_update_qty == "dw_out_norm":
            weights_to_consider = weight_info['dw_out']
        elif self.args.lr_update_qty == "dw_out1_norm":
            weights_to_consider = weight_info['dw_out1']
        elif self.args.lr_update_qty == "wout_norm":
            weights_to_consider = weight_info['w_out']
        else:
            raise ValueError(f"Unknown lr_update_qty {self.args.lr_update_qty}")
        norms = torch.sqrt(self.fudge + torch.sum(weights_to_consider ** 2, dim=1))


        if self.args.lr_update_mode == "basic":
            nonneg_decay_coeff = torch.exp(self.log_learning_rate_decay)
            lr_mult = self.args.lr_min_mult + (lr_mult - self.args.lr_min_mult) * \
                torch.exp(-(nonneg_decay_coeff * norms))
            lr = lr0 * lr_mult
        if self.args.lr_update_mode == "recoverable":
            #same as basic but with a recovery toward lr_mult=1
            nonneg_decay_coeff = torch.exp(self.log_learning_rate_decay)
            lr_plastic = lr_mult - self.args.lr_min_mult
            lr_plastic = (1-self.args.lr_recovery_rate) * lr_plastic + self.args.lr_recovery_rate * 1.0
            lr_plastic = lr_plastic * torch.exp(-(nonneg_decay_coeff * norms))
            lr_mult = self.args.lr_min_mult + lr_plastic
            lr = lr0 * lr_mult
        if self.args.lr_update_mode == "recoverable_opt2":
            #same as recoverable but with the order of updates swapped
            nonneg_decay_coeff = torch.exp(self.log_learning_rate_decay)
            lr_plastic = lr_mult - self.args.lr_min_mult
            lr_plastic = (1-self.args.lr_recovery_rate) * lr_plastic + self.args.lr_recovery_rate * 1.0
            lr_plastic = lr_plastic * torch.exp(-(nonneg_decay_coeff * norms))
            lr_mult = self.args.lr_min_mult + lr_plastic
            lr = lr0 * lr_mult        
        return lr_mult, lr

    # ---------------------------
    # hidden / input helpers
    # ---------------------------

    def _compute_scaled_q_in(self, prescaled_w_inq, qlp):
        if prescaled_w_inq is None:
            return 0.0

        if self._separate_win_mode():
            q_gain = (self.q_scale * qlp)[:, None, None]
            return prescaled_w_inq[None, :, :] * q_gain
        else:
            q_gain = (self.q_scale * qlp)[:, None]
            return prescaled_w_inq[None, :] * q_gain

    def _compute_w_in(self, w_in, w_in_tuning):
        base = w_in.unsqueeze(0)
        if not self.args.enable_w_in_plasticity:
            return base
        return self.win_nl(base + w_in_tuning * torch.sign(base))

    def _hidden_preact(self, biases_, x, w_in_, scaled_q_in):
        return biases_ + self._broadcast_x_to_hidden(x) * w_in_ + scaled_q_in

    def _compute_hidden(self, biases_, x, w_in_, scaled_q_in):
        return self.phi(self._hidden_preact(biases_, x, w_in_, scaled_q_in))

    def _compute_u(self, w_out, h, x):
        if self._separate_win_mode():
            u = torch.einsum("bnm,bnm->b", w_out, h)
        else:
            w_eff = w_out.sum(dim=2)
            if not self.mult_activation_mode:
                u = torch.einsum("bn,bn->b", w_eff, h)
            else:
                u = torch.einsum("bni,bni->b", w_eff, h)
        return u + self.args.skip_gain * x

    def _compute_h_bar(self, h, biases_, x, w_in_, u, e, scaled_q_in, inj_, mode=None, x_state=None):
        if mode == 2:
            if self.args.noise_injection_node == "x":
                raise ValueError("injection_opt2 incompatible with noise_injection_node 'x' yet")
            x_prime, _ = self._update_x(x, u, e, x_state)
            h_prime = self._compute_hidden(biases_, x_prime, w_in_, scaled_q_in)
            inj = self._broadcast_inj_to_hidden(inj_)
            return (1.0 - inj) * h + inj * h_prime

        elif mode == 3:
            if self.args.noise_injection_node == "x":
                raise ValueError("injection_opt3 incompatible with noise_injection_node 'x' yet")
            x_prime, _ = self._update_x(x, u, e, x_state)
            x_bar = (1.0 - inj_) * x + inj_ * x_prime
            return self._compute_hidden(biases_, x_bar, w_in_, scaled_q_in)

        elif mode == 0:
            return h

        else:
            raise ValueError(f"Unknown injection_opt mode {mode}")

    def _w_in_update(self, w_in_tuning, h_bar):
        if self.mult_activation_mode:
            raise ValueError("w_in plasticity not yet implemented for mult_activation_mode yet")

        if self._separate_win_mode():
            decay = self.w_in_decay().unsqueeze(1)   # [bs,1,m]
            lr = self.w_in_lr().unsqueeze(1)         # [bs,1,m]
        else:
            decay = self.w_in_decay()                # [bs,1]
            lr = self.w_in_lr()                      # [bs,1]
        return (1 - decay) * w_in_tuning + lr * h_bar

    def _bias_update(self, biases, bias_tuning, x, w_in_, scaled_q_in):
        if self.args.enable_bias_update:
            biases_ = biases + bias_tuning
            preact = self._broadcast_x_to_hidden(x) * w_in_ + scaled_q_in
            if self._separate_win_mode():
                bias_tuning = bias_tuning + (self.args.develop_b_tgt + preact - biases_) * self.bias_lr[:, None, None]
            else:
                bias_tuning = bias_tuning + (self.args.develop_b_tgt + preact - biases_) * self.bias_lr[:, None]
            return biases_, bias_tuning
        return biases, bias_tuning

    def _compute_elp_for_learning(self, elp):
        if self.args.enable_weight_learning_exp:
            return torch.sign(elp[:, None]) * torch.abs(elp[:, None]).pow(
                self.softplus(self.weight_learning_exp)
            )
        else:
            return elp[:, None]

    def _compute_dw_out(self, lr, elp, h_bar, w_out, wd):
        # elp_ is [bs,m]
        elp_ = self._compute_elp_for_learning(elp)

        if self._separate_win_mode():
            dw_out = lr.unsqueeze(1) * elp_.unsqueeze(1) * h_bar
        else:
            h_bar_ = h_bar.unsqueeze(2) if not self.mult_activation_mode else h_bar
            dw_out = lr.unsqueeze(1) * elp_.unsqueeze(1) * h_bar_

        if self.args.model_tie_lr_weight_decay:
            decay = (lr * wd).unsqueeze(1)
        else:
            decay = wd.unsqueeze(1)

        if self.args.enable_weight_decay_exp:
            w_out_ = torch.sign(w_out) * torch.abs(w_out).pow(self.weight_decay_exp.unsqueeze(1))
        else:
            w_out_ = w_out

        return dw_out, - decay * w_out_

    def _compute_steady_state_w_in_tuning(self, biases):
        if self._separate_win_mode():
            raise NotImplementedError("Steady state w_in tuning not implemented for separate_win_mode yet")
        else:
            #biases in [b,n], w_in_tuning in [b,n], w_in_decay and w_in_lr in [b,1]
            #we need to compute phi(biases) * self.w_in_lr / self.w_in_decay
            return self.win_nl(biases) * (self.w_in_lr() / self.w_in_decay())

    def w_in_lr(self):
        if not self.args.apply_scaled_soft_plus_on_w_in_params:
            return self.w_in_lr_
        else:
            return 0.01*self.softplus(self.scsp_w_in_lr_)
    
    def w_in_decay(self):
        if not self.args.apply_scaled_soft_plus_on_w_in_params:
            return self.w_in_decay_
        else:
            return 0.01*self.softplus(self.scsp_w_in_decay_)

    # ---------------------------
    # forward rollout
    # ---------------------------

    def f(
        self,
        n: int,
        noises: List[torch.Tensor],
        ys: List[Optional[torch.Tensor]],
        model_setting: str,
        qs: Optional[List[Optional[torch.Tensor]]] = None,
        record_internals: bool = False,
        record_vectors: bool = False,
        record_inoutmaps: bool = False,
        inoutmaps_probing_vec: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:

        assert len(noises) == len(ys), "noises and ys must have same length"
        assert model_setting in ["default"], "model_setting must be 'default' for this model class"

        bs = noises[0].shape[0]
        assert bs == self.bs, f"Model was initialized with bs={self.bs}, but got input bs={bs}"

        device = self.device

        biases, w_in = self.get_biases_and_w_in(n=n, device=device)
        prescaled_w_inq = self.get_winq(n=n, device=device) if qs is not None else None

        state = self._init_runtime_state(bs, n, device)
        if self.args.initiate_w_in_tuning_with_steady_state_vals:
            # print(f'win_tuning shape: {state["w_in_tuning"].shape}, biases shape: {biases.shape}, w_in_lr shape: {self.w_in_lr.shape}, w_in_decay shape: {self.w_in_decay.shape}')
            state["w_in_tuning"] = self._compute_steady_state_w_in_tuning(biases)
            # print(f'win_tuning shape after initialisation: {state["w_in_tuning"].shape} ')

        w_out = state["w_out"]
        bias_tuning = state["bias_tuning"]
        u = state["u"]
        x = state["x"]
        e = state["e"]
        qlp = state["qlp"]
        ylp = state["ylp"]
        elp = state["elp"]
        w_in_tuning = state["w_in_tuning"]
        x_state = state["x_state"]

        lr, lr0, lr_mult = self._init_lr(bs)
        wd = self._assign_wd(lr0, n)

        if qs is None:
            qs = [torch.zeros((1,), device=device)] * len(ys)

        tauqlpf = 1.0 + self.softplus(self.tauqlpf_m1)
        tauylpf = 1.0 + self.softplus(self.tauylpf_m1)
        tauelpf = 1.0 + self.softplus(self.tauelpf_m1)

        inj_ = torch.sigmoid(self.direct_injection_scale) * self.args.direct_inj_limiter

        a_means: List[torch.Tensor] = []

        if record_internals:
            self.internals = {
                "u": [],
                "e": [],
                "qlp": [],
                "ylp": [],
                "elp": [],
                "lr": [],
                "wd": [],
                "x": [],
            }
        if record_vectors:
            raise NotImplementedError("record_vectors not yet implemented for this model class")
            # self.vectors = {
            #     "w_out": [],
            #     "h": [],
            #     "biases_": [],
            #     "x": [],
            #     "w_in_": [],
            #     "scaled_q_in": [],
            #     }
        if record_inoutmaps:
            self.inoutmaps = []
            if inoutmaps_probing_vec is None:
                raise ValueError("inoutmaps_probing_vec must be provided if record_inoutmaps is True")


        for y, noise_x, q in zip(ys, noises, qs):
            y, noise_x, q = self._prep_inputs(y, noise_x, q, bs=bs)

            y = self.input_scale * y 

            qlp = self._lowpass_filter(q, qlp, tauqlpf, enable=not self.args.disable_lpfs)
            scaled_q_in = self._compute_scaled_q_in(prescaled_w_inq, qlp)

            x, x_state = self._update_x(x, u, elp, x_state)

            w_in_ = self._compute_w_in(w_in, w_in_tuning)
            biases_, bias_tuning = self._bias_update(biases, bias_tuning, x, w_in_, scaled_q_in)

            mask = torch.isnan(y)

            if self.args.injection_opt == 1:
                raise ValueError("injection_opt 1 is deprecated, use injection_opt 2 instead")

            h = self._compute_hidden(biases_, x, w_in_, scaled_q_in)
            u = self._compute_u(w_out, h, x)

            a_mean = self.output_scale * u

            if record_inoutmaps:
                inoutmap_out = self._compute_hidden(biases_, inoutmaps_probing_vec, w_in_, scaled_q_in)
                self.inoutmaps.append((inoutmap_out, inoutmaps_probing_vec))

            y_ = torch.where(mask, u + self.args.channel_trial_extra_error, y)
            ylp = self._lowpass_filter(y_, ylp, tauylpf, enable=not self.args.disable_lpfs)

            e = ylp - u
            elp = self._lowpass_filter(e, elp, tauelpf, enable=not self.args.disable_lpfs)

            h_bar = self._compute_h_bar(
                h=h,
                biases_=biases_,
                x=x,
                w_in_=w_in_,
                u=u,
                e=elp,
                scaled_q_in=scaled_q_in,
                inj_=inj_,
                mode=self.args.injection_opt,
                x_state=x_state,
            )

            dw_out1, dw_out2 = self._compute_dw_out(lr, elp, h_bar, w_out, wd)

            dw_out = dw_out1 + dw_out2

            # if self.args.apply_lr_decay:
                #norms = torch.sqrt(self.fudge + torch.sum(dw_out ** 2, dim=1))     

            w_out = w_out + dw_out 

            if self.args.enable_w_in_plasticity:
                w_in_tuning = self._w_in_update(w_in_tuning, h_bar)

            if self.args.apply_lr_decay:
                lr_mult, lr = self._lr_update(lr_mult, lr0, weight_info={'w_out': w_out, 'dw_out': dw_out, 'dw_out1': dw_out1, 'dw_out2': dw_out2})

            a_means.append(a_mean)

            if record_internals:
                self.internals["u"].append(u)
                self.internals["e"].append(e)
                self.internals["qlp"].append(qlp)
                self.internals["ylp"].append(ylp)
                self.internals["elp"].append(elp)
                self.internals["lr"].append(lr)
                self.internals["wd"].append(wd)
                self.internals["x"].append(x)

        if record_internals:
            for k in self.internals:
                self.internals[k] = torch.stack(self.internals[k], dim=0)
            return a_means, self.internals

        return a_means