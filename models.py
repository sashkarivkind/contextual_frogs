import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


def nl_selector(nl):

    if nl is None:
        nl = nn.Identity

    if isinstance(nl,str):
        if nl == 'relu':
            nl = nn.ReLU
        elif nl == 'symrelu':
            nl = SymReLU
        elif nl == 'tanh':
            nl = nn.Tanh
        elif nl == 'sigmoid':
            nl = nn.Sigmoid
        # elif nl is None:
        #     nl = lambda x: x
        else:
            raise ValueError('unkonow nonlinearity')
    return nl


class ElboGenerativeModelTop(nn.Module):

    def __init__(self, device: torch.device, args=None, fudge=1e-4):
        super().__init__()
        # randomize initial parameters within reasonable ranges (conditional on not zzz_legacy_init for backwards compatibility):
        init_log_learning_rate = -7.0 + 2.5 * np.random.rand() if not args.zzz_legacy_init else -6.0  # [-7,-4.5]
        init_log_learning_rate_decay = -1 + 2 * np.random.rand() if not args.zzz_legacy_init else 0.0  # [-1, 1]
        init_sigma_b = 0.05 + 0.5 * np.random.rand() if not args.zzz_legacy_init else 0.1  # [0.05, 0.55]
        init_output_scale = 0.8 + 0.2 * np.random.rand()  if not args.zzz_legacy_init else 1.0  # [0.8, 1.0]
        init_sp_weight_decay = -5 + 2 * np.random.rand() if not args.zzz_legacy_init else -4  # [-5, -3]
        init_sigma_a = 0.02 + 0.1 * np.random.rand()  if not args.zzz_legacy_init else 0.1  # [0.02, 0.12]
        init_sigma_x = 0.02 + 0.1 * np.random.rand()  if not args.zzz_legacy_init else 0.1  # [0.02, 0.12]

        # match OCaml init scales
        optimize_noises = not args.model == "toy" or args.optimize_toy_noises
        self.log_learning_rate = nn.Parameter(torch.full((1,), init_log_learning_rate, device=device)) 
        self.log_learning_rate_decay = nn.Parameter(torch.full((1,), init_log_learning_rate_decay, device=device))
        self.sigma_b = nn.Parameter(torch.full((1,), init_sigma_b, device=device))

        self.output_scale = nn.Parameter(torch.full((1,), init_output_scale, device=device)) if args.enable_output_scale_tuning else torch.tensor(1.0, device=device, requires_grad=False)
        self.u_feedback_scale = nn.Parameter(torch.full((1,), 1.0, device=device)) if args.enable_u_feedback_scale_tuning else torch.tensor(1.0, device=device, requires_grad=False)    
        self.sp_weight_decay = nn.Parameter(torch.full((1,), init_sp_weight_decay, device=device))
        self.q_scale = nn.Parameter(torch.full((1,), 0.3, device=device)) if args.enable_q_scale_tuning else torch.tensor(1.0, device=device, requires_grad=False)
        if not args.assume_opt_output_noise:
            self.sigma_a = nn.Parameter(torch.full((1,), init_sigma_a, device=device)) if optimize_noises else torch.tensor(args.toymodel_OUsigma_obs, 
                                                                                                                       device=device, 
                                                                                                                       requires_grad=False)
        self.sigma_x = nn.Parameter(torch.full((1,), init_sigma_x, device=device)) if optimize_noises else torch.tensor(args.toymodel_OUsigma_process, 
                                                                                                                       device=device, 
                                                                                                                       requires_grad=False)

        
        self.tauqlpf_m1 = nn.Parameter(torch.full((1,), -1., device=device)) if args.enable_qlpf else torch.tensor(-1000, device=device, requires_grad=False)
        self.tauylpf_m1 = nn.Parameter(torch.full((1,), -1., device=device)) if args.enable_ylpf else torch.tensor(-1000, device=device, requires_grad=False)
        self.tauelpf_m1 = nn.Parameter(torch.full((1,), 1., device=device)) if args.enable_elpf else torch.tensor(-1000, device=device, requires_grad=False)


        self.register_buffer("_z_biases", torch.empty(0))  # base N(0,1) for biases
        self.register_buffer("_w_in", torch.empty(0))      # random input features
        self.register_buffer("_w_inq", torch.empty(0))  # random input features for q
        self.args = args
        self.fudge = fudge


    @staticmethod
    def better_relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)
    
    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x))

    # def _sample_biases_and_w_in(self, n, bs, device):
    #     noise = torch.randn(n, device=device)          # no grad needed
    #     biases = self.sigma_b.reshape(1) * noise       # grad flows to sigma_b
    #     w_in = torch.randn(n, device=device)
    #     return biases, w_in

    def _ensure_random_features(self, n: int, device):
        # Lazily initialize once, tied to n
        if self._w_in.numel() != n:
            # (Re)create fixed random features/bias base draw
            self.register_buffer("_z_biases", torch.randn(n, device=device))
            self.register_buffer("_w_in", torch.randn(n, device=device))
            self.register_buffer("_w_inq", torch.randn(n, device=device))

    def get_biases_and_w_in(self, n: int, device):
        self._ensure_random_features(n, device)
        biases = self.sigma_b.reshape(1) * self._z_biases  # scale base draw; grads flow to sigma_b
        return biases, self._w_in
    
    def get_winq(self, n: int, device):
        self._ensure_random_features(n, device)
        return self._w_inq

    def f(self,
          n: int,
          noises: List[torch.Tensor],   # list of [bs] tensors (one per time step)
          ys: List[Optional[torch.Tensor]],  # list of optional target tensors (shape [1] per OCaml)
          model_setting: str,
          qs: Optional[List[Optional[torch.Tensor]]] = None, #   model_offset_as_wtr_ys: bool = True,
          ) -> List[torch.Tensor]:
        """
        Port of the OCaml `Generative_model.f`.
        Returns a list of a_means (each shape [bs]) for each time step.
        """
        assert len(noises) == len(ys), "noises and ys must have same length"
        bs = noises[0].shape[0]
        device = self.log_learning_rate.device

        # biases, w_in = self._sample_biases_and_w_in(n=n, bs=bs, dvice=device)
        biases, w_in = self.get_biases_and_w_in(n=n, device=device)
        prescaled_w_inq = self.get_winq(n=n, device=device) if qs is not None else None

        # state init
        w_out = torch.zeros(bs, n, device=device)
        u = torch.zeros(bs, device=device)
        x = torch.zeros(bs, device=device)
        e = torch.zeros(bs, device=device)
        lr_mult = torch.ones(bs, device=device)
        qlp = torch.zeros(bs, device=device)
        ylp = torch.zeros(bs, device=device)
        elp = torch.zeros(bs, device=device)
        lr0 = torch.exp(self.log_learning_rate).expand(bs)
        lr = lr0 * lr_mult

        tauqlpf = 1.0 + self.softplus(self.tauqlpf_m1)
        tauylpf = 1.0 + self.softplus(self.tauylpf_m1)
        tauelpf = 1.0 + self.softplus(self.tauelpf_m1)

        a_means: List[torch.Tensor] = []
        for y, noise_x, q in zip(ys, noises,
                                 qs if qs is not None else [torch.zeros((1,), device=device)]*len(ys)):
            #if y is np.nan convert to None (y might be a torch tensor or a numpy scalar)
            # if y is not None and (isinstance(y, float) and np.isnan(y)):
            #     y = None
            if y is None:
                y = torch.full((bs,), float('nan'), device=device)
        
            if model_setting == "toy":
                x = self.args.toymodel_OUphi * x + noise_x
                a_means.append(1.0 * x)

            elif model_setting == "default":
                qlp = (1.0 - 1./tauqlpf) * qlp + 1./tauqlpf * q
                scaled_q_in  = (prescaled_w_inq.unsqueeze(0) * self.q_scale * qlp.unsqueeze(1))  if prescaled_w_inq is not None else 0 #TODO: refactor

                x = u * self.u_feedback_scale + \
                    + e + (noise_x if self.args.noise_injection_node == 'x' else 0) #TODO: rename noise_x to generalise
                h = self.better_relu(biases + (x.unsqueeze(1) * w_in.unsqueeze(0)) + scaled_q_in)
                u = torch.einsum("kj,kj->k", w_out, h) + (noise_x if self.args.noise_injection_node == 'u' else 0)
                a_mean = (self.output_scale * u).squeeze() + (noise_x if self.args.noise_injection_node == 'a' else 0)

                # if y is not None: 
                #     ylp = (1.0 - 1./tauylpf) * ylp + 1./tauylpf * y
                #     e = ylp.to(device).expand_as(u) - u
                # else:
                #     e = torch.zeros_like(u, device=device)
                
                # elements of y that are not specified (NaN) are replaced with corresponding elements of u
                mask = torch.isnan(y)
                y_ = torch.where(mask, u, y)
                ylp = (1.0 - 1./tauylpf) * ylp + 1./tauylpf * y_
                e = ylp.expand_as(u) - u

                elp = (1.0 - 1./tauelpf) * elp + 1./tauelpf * e
                dw_out = lr.unsqueeze(1) * elp.unsqueeze(1) * h
                norms = torch.sqrt(self.fudge + torch.einsum("ki->k", dw_out ** 2))
                dw_out = dw_out - lr.unsqueeze(1) * self.softplus(self.sp_weight_decay) * w_out
                w_out = w_out + dw_out
                #learning rate update
                lr_mult = lr_mult * torch.exp(-(torch.exp(self.log_learning_rate_decay) * norms))
                lr = lr0 * lr_mult
                # keeping records
                a_means.append(a_mean)
            else:
                raise ValueError("unknown model setting")

        # if model_offset_as_wtr_ys:
        #     a_means = [torch.zeros((bs, 1), device=device)] + a_means[:-1] 
        return a_means


class ElboGenerativeModelDualRate(nn.Module):

    def __init__(self, device: torch.device, args=None, fudge=1e-4):
        super().__init__()

        # --- sample within recommended ranges ---
        # Aslow in [0.97, 0.999]
        As = 0.97 + (0.999 - 0.97) * torch.rand(1, device=device)

        # draw a gap so Af < As, then clamp Af to [0.55, 0.90] for realism
        gap = 0.07 + (0.40 - 0.07) * torch.rand(1, device=device)   # gap in [0.07, 0.40]
        Af_raw = As - gap
        Af = torch.clamp(Af_raw, min=torch.tensor(0.55, device=device),
                                   max=torch.tensor(0.90, device=device))
        # ensure strict Af < As (tiny epsilon in case clamp hit As)
        Af = torch.minimum(Af, As - torch.tensor(1e-4, device=device))

        # Bslow in [0.005, 0.05]
        Bs = 0.005 + (0.05 - 0.005) * torch.rand(1, device=device)
        # Bfast in [0.15, 0.35]
        Bf = 0.15 + (0.35 - 0.15) * torch.rand(1, device=device)

        # --- register learnable parameters ---
        self.aslow = nn.Parameter(As)   # ~0.97–0.999
        self.afast = nn.Parameter(Af)   # < aslow, ~0.55–0.90
        self.bslow = nn.Parameter(Bs)   # ~0.005–0.05
        self.bfast = nn.Parameter(Bf)   # ~0.15–0.35

        # self.aslow = nn.Parameter(torch.full((1,), 0.95, device=device))  
        # self.afast = nn.Parameter(torch.full((1,), 0.2, device=device))
        # self.bslow = nn.Parameter(torch.full((1,), 0.05, device=device))
        # self.bfast = nn.Parameter(torch.full((1,), 0.5, device=device))

        self.sigma_x = nn.Parameter(torch.full((1,), 0.1, device=device))
        self.fudge = fudge
        if args.noise_injection_node not in ['x', 'a']:
            raise ValueError("noise_injection_node for Dual Rate model must be one of 'x', 'a'")
        self.args = args


    @staticmethod
    def better_relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)
    
    def f(self,
          n: int,
          noises: List[torch.Tensor],   # list of [bs] tensors (one per time step)
          ys: List[Optional[torch.Tensor]],  # list of optional target tensors (shape [1] per OCaml)
          model_setting: str,
          qs: Optional[List[Optional[torch.Tensor]]] = None,         
          ) -> List[torch.Tensor]:
        """
        Port of the OCaml `Generative_model.f`.
        Returns a list of a_means (each shape [bs]) for each time step.
        """
        assert len(noises) == len(ys), "noises and ys must have same length"
        bs = noises[0].shape[0]
        device = self.aslow.device

        # biases, w_in = self._sample_biases_and_w_in(n=n, bs=bs, dvice=device)


        # state init
        xs = torch.zeros(bs, device=device)
        xf = torch.zeros(bs, device=device)
        e = torch.zeros(bs, device=device)

        a_means: List[torch.Tensor] = []
        for y, noise_x in zip(ys, noises):

                xs = self.aslow * xs + self.bslow * e + (noise_x if self.args.noise_injection_node == 'x' else 0) #TODO: rename noise_x to generalise
                xf = self.afast * xf + self.bfast * e 
                u = xs + xf 
                if y is not None: 
                    e = y - u
                else:
                    e = torch.zeros_like(u, device=device)
                # keeping records
                a_mean = u + (noise_x if self.args.noise_injection_node == 'a' else 0)
                a_means.append(a_mean)
        return a_means

class LegacyElboGenerativeModelTop(nn.Module):
    """
    OCaml generative parameters:
      log_learning_rate, log_learning_rate_decay, sigma_b, output_scale,
      log_weight_decay, sigma_a, sigma_x  (all scalars in the OCaml init)
    """
    def __init__(self, device: torch.device, args=None, fudge=1e-4):
        super().__init__()
        # randomize initial parameters within reasonable ranges (conditional on not zzz_legacy_init for backwards compatibility):
        init_log_learning_rate = -7.0 + 2.5 * np.random.rand() if not args.zzz_legacy_init else -6.0  # [-7,-4.5]
        init_log_learning_rate_decay = -1 + 2 * np.random.rand() if not args.zzz_legacy_init else 0.0  # [-1, 1]
        init_sigma_b = 0.05 + 0.5 * np.random.rand() if not args.zzz_legacy_init else 0.1  # [0.05, 0.55]
        init_output_scale = 0.8 + 0.2 * np.random.rand()  if not args.zzz_legacy_init else 1.0  # [0.8, 1.0]
        init_log_weight_decay = -0.00 - 0.05 * np.random.rand() if not args.zzz_legacy_init else -0.001  # [-0.01, 0.01]
        init_sigma_a = 0.02 + 0.1 * np.random.rand()  if not args.zzz_legacy_init else 0.1  # [0.02, 0.12]
        init_sigma_x = 0.02 + 0.1 * np.random.rand()  if not args.zzz_legacy_init else 0.1  # [0.02, 0.12]

        # match OCaml init scales
        optimize_noises = not args.model == "toy" or args.optimize_toy_noises
        self.log_learning_rate = nn.Parameter(torch.full((1,), init_log_learning_rate, device=device)) 
        self.log_learning_rate_decay = nn.Parameter(torch.full((1,), init_log_learning_rate_decay, device=device))
        self.sigma_b = nn.Parameter(torch.full((1,), init_sigma_b, device=device))

        self.output_scale = nn.Parameter(torch.full((1,), init_output_scale, device=device)) if args.enable_output_scale_tuning else torch.tensor(1.0, device=device, requires_grad=False)
        self.u_feedback_scale = nn.Parameter(torch.full((1,), 1.0, device=device)) if args.enable_u_feedback_scale_tuning else torch.tensor(1.0, device=device, requires_grad=False)    
        self.log_weight_decay = nn.Parameter(torch.full((1,), init_log_weight_decay, device=device))
        self.q_scale = nn.Parameter(torch.full((1,), 1.0, device=device)) if args.enable_q_scale_tuning else torch.tensor(1.0, device=device, requires_grad=False)
        if not args.assume_opt_output_noise:
            self.sigma_a = nn.Parameter(torch.full((1,), init_sigma_a, device=device)) if optimize_noises else torch.tensor(args.toymodel_OUsigma_obs, 
                                                                                                                       device=device, 
                                                                                                                       requires_grad=False)
        self.sigma_x = nn.Parameter(torch.full((1,), init_sigma_x, device=device)) if optimize_noises else torch.tensor(args.toymodel_OUsigma_process, 
                                                                                                                       device=device, 
                                                                                                                       requires_grad=False)

        
        self.tauqlpf_m1 = nn.Parameter(torch.full((1,), -1., device=device)) if args.enable_qlpf else torch.tensor(-1000, device=device, requires_grad=False)
        self.tauylpf_m1 = nn.Parameter(torch.full((1,), -1., device=device)) if args.enable_ylpf else torch.tensor(-1000, device=device, requires_grad=False)
        self.tauelpf_m1 = nn.Parameter(torch.full((1,), -1., device=device)) if args.enable_ylpf else torch.tensor(-1000, device=device, requires_grad=False)


        self.register_buffer("_z_biases", torch.empty(0))  # base N(0,1) for biases
        self.register_buffer("_w_in", torch.empty(0))      # random input features
        self.register_buffer("_w_inq", torch.empty(0))  # random input features for q
        self.args = args
        self.fudge = fudge


    @staticmethod
    def better_relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)
    
    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x))

    # def _sample_biases_and_w_in(self, n, bs, device):
    #     noise = torch.randn(n, device=device)          # no grad needed
    #     biases = self.sigma_b.reshape(1) * noise       # grad flows to sigma_b
    #     w_in = torch.randn(n, device=device)
    #     return biases, w_in

    def _ensure_random_features(self, n: int, device):
        # Lazily initialize once, tied to n
        if self._w_in.numel() != n:
            # (Re)create fixed random features/bias base draw
            self.register_buffer("_z_biases", torch.randn(n, device=device))
            self.register_buffer("_w_in", torch.randn(n, device=device))
            self.register_buffer("_w_inq", torch.randn(n, device=device))

    def get_biases_and_w_in(self, n: int, device):
        self._ensure_random_features(n, device)
        biases = self.sigma_b.reshape(1) * self._z_biases  # scale base draw; grads flow to sigma_b
        return biases, self._w_in
    
    def get_winq(self, n: int, device):
        self._ensure_random_features(n, device)
        return self._w_inq

    def f(self,
          n: int,
          noises: List[torch.Tensor],   # list of [bs] tensors (one per time step)
          ys: List[Optional[torch.Tensor]],  # list of optional target tensors (shape [1] per OCaml)
          model_setting: str,
          qs: Optional[List[Optional[torch.Tensor]]] = None,         
          ) -> List[torch.Tensor]:
        """
        Port of the OCaml `Generative_model.f`.
        Returns a list of a_means (each shape [bs]) for each time step.
        """
        assert len(noises) == len(ys), "noises and ys must have same length"
        bs = noises[0].shape[0]
        device = self.log_learning_rate.device

        # biases, w_in = self._sample_biases_and_w_in(n=n, bs=bs, dvice=device)
        biases, w_in = self.get_biases_and_w_in(n=n, device=device)
        prescaled_w_inq = self.get_winq(n=n, device=device) if qs is not None else None

        # state init
        w_out = torch.zeros(bs, n, device=device)
        u = torch.zeros(bs, device=device)
        x = torch.zeros(bs, device=device)
        e = torch.zeros(bs, device=device)
        lr_mult = torch.ones(bs, device=device)
        qlp = torch.zeros(1, device=device)
        ylp = torch.zeros(1, device=device)
        lr0 = torch.exp(self.log_learning_rate).expand(bs)
        lr = lr0 * lr_mult

        tauqlpf = 1.0 + self.softplus(self.tauqlpf_m1)
        tauylpf = 1.0 + self.softplus(self.tauylpf_m1)

        a_means: List[torch.Tensor] = []
        for y, noise_x, q in zip(ys, noises,
                                 qs if qs is not None else [torch.zeros((1,), device=device)]*len(ys)):
            #if y is np.nan convert to None (y might be a torch tensor or a numpy scalar)
            if y is not None and (isinstance(y, float) and np.isnan(y)):
                y = None
            if model_setting == "toy":
                x = self.args.toymodel_OUphi * x + noise_x
                a_means.append(1.0 * x)

            elif model_setting == "default":
                qlp = (1.0 - tauqlpf) * qlp + tauqlpf * q
                scaled_q_in  = (prescaled_w_inq * self.q_scale * qlp).unsqueeze(0)  if prescaled_w_inq is not None else 0 #TODO: refactor

                x = u * self.u_feedback_scale + \
                    + e + (noise_x if self.args.noise_injection_node == 'x' else 0) #TODO: rename noise_x to generalise
                h = self.better_relu(biases + (x.unsqueeze(1) * w_in.unsqueeze(0)) + scaled_q_in)
                u = torch.einsum("kj,kj->k", w_out, h) + (noise_x if self.args.noise_injection_node == 'u' else 0)
                a_mean = (self.output_scale * u).squeeze() + (noise_x if self.args.noise_injection_node == 'a' else 0)
                if y is not None: 
                    ylp = (1.0 - tauylpf) * ylp + tauylpf * y
                    e = ylp.to(device).expand_as(u) - u
                    dw_out = (e.unsqueeze(1) * h) * lr.unsqueeze(1)
                    w_out = w_out + dw_out
                    norms = torch.sqrt(self.fudge + torch.einsum("ki->k", dw_out ** 2))
                    lr_mult = lr_mult * torch.exp(-(torch.exp(self.log_learning_rate_decay) * norms))
                    lr = lr0 * lr_mult
                else:
                    e = torch.zeros_like(u, device=device)
                w_out = (torch.exp(self.log_weight_decay * (lr_mult if self.args.model_tie_lr_weight_decay else 1.0)).unsqueeze(1) * w_out)
                # keeping records
                a_means.append(a_mean)
            else:
                raise ValueError("unknown model setting")

        return a_means

class ModelForRunner():
    #abstract class with reset_state method
    def reset_state(self):
        raise NotImplementedError

class PlainRNN(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, g=1.0, nl='relu', trainable=False):
        super(PlainRNN, self).__init__()
        self.nl = nl_selector(nl)()
        self.hidden_size = hidden_size
        
        # Initialize fixed weights (non-trainable)
        self.W = nn.Parameter(torch.normal(mean=0, std=g/np.sqrt(hidden_size), size=(hidden_size, hidden_size)), requires_grad=trainable)
        self.U = nn.Parameter(torch.normal(mean=0, std=1., size=(input_size, hidden_size)), requires_grad=trainable)
    
    def forward(self, x, h):
        """
        x: Input tensor of shape (batch_size, input_size)
        h: Hidden state tensor of shape (batch_size, hidden_size)
        """
        h = self.nl(
            torch.matmul(h, self.W) + torch.matmul(x, self.U))
                              
        return h

class ParallelMLP(nn.Module, ModelForRunner):
    # parallel MLPs receive list of parameter dicts, each is used to initialize a separate MLP
    # raises error in n_outs is different for different MLPs
    # or if n_inputs is different for different MLPs when shared_input=True
    # if shared_input is false then the input is sliced according to n_inputs in each parameter set, and each MLP receives a different slice 
    # the outputs are summed
    def __init__(self, parameter_sets=None, n_inputs=None, shared_input=True, info=None):
        super(ParallelMLP, self).__init__()
        self.models = nn.ModuleList([MLP(**params) for params in parameter_sets])
        self.n_outs = self.models[0].output_layer.out_features
        self.shared_input = shared_input
        for model in self.models:
            if model.output_layer.out_features != self.n_outs:
                raise ValueError('inconsistent n_outs in parallel MLPs')
        
        if shared_input:
            self.n_inputs = self.models[0].input_layer.in_features
            for model in self.models:
                if model.input_layer.in_features != self.n_inputs:
                    raise ValueError('inconsistent n_inputs in parallel MLPs')
        else:
            self.n_inputs = [model.input_layer.in_features for model in self.models]
        
    def forward(self, x):
        if self.shared_input:
            return torch.stack([model(x) for model in self.models], dim=0).sum(dim=0)
        else:
            #prepare input slices for each model
            counter = 0
            slices = []
            for n_inputs in self.n_inputs:
                slices.append(x[:, counter:counter+n_inputs])
                counter += n_inputs
            return torch.stack([model(slice) for model, slice in zip(self.models, slices)], dim=0).sum(dim=0)

    def reset_state(self):
        for model in self.models:
            model.reset_state()


class MLP(nn.Module, ModelForRunner):
    def __init__(
        self, n_inputs, n_hidden, n_outs,
        n_layers=1, nl='tanh', en_bias=True,
        prescaling=None, main_gain=None, skip_gain=None,
        b_low=None, b_high=None, manual_bias_requires_grad=False,
        first_layer_init='default', first_layer_weights_trainable=False,
        out_layer_init='default', post_activation_bias=None,
        post_activation_bias_scale=1, info=None, return_post_acts=False,
    ):
        super().__init__()
        # prescaling buffer
        prescale = torch.ones(n_inputs) if prescaling is None else torch.tensor(prescaling, dtype=torch.float32)
        self.register_buffer('prescaling', prescale)
        # main_gain buffer
        mg = torch.ones(n_outs) if main_gain is None else torch.tensor(main_gain, dtype=torch.float32)
        self.register_buffer('main_gain', mg)

        # skip_gain buffer: default scalar 0 if None
        sg_val = 0 if skip_gain is None else skip_gain
        if isinstance(sg_val, (int, float)):
            sg_list = [sg_val] * n_inputs
        elif isinstance(sg_val, (np.ndarray, list, torch.Tensor)):
            # convert to Python list
            sg_list = list(sg_val) if not isinstance(sg_val, torch.Tensor) else sg_val.tolist()
            if len(sg_list) == 1:
                sg_list = sg_list * n_inputs
            if len(sg_list) != n_inputs:
                raise ValueError(f"skip_gain must be scalar, length-1, or length-{n_inputs}")
        else:
            raise TypeError("skip_gain must be scalar, list, np.ndarray, or torch.Tensor")
        self.register_buffer('skip_gain', torch.tensor(sg_list, dtype=torch.float32))

        # activation
        self.activation = nl_selector(nl)()
        self.return_post_acts = return_post_acts
        # build layers
        self.layers = nn.ModuleList()
        if b_low is None and b_high is not None:
            b_low = -b_high
        in_bias = en_bias or (b_low is not None and b_high is not None)
        inp = nn.Linear(n_inputs, n_hidden, bias=in_bias)
        _init_linear(inp, first_layer_init)
        inp.weight.requires_grad = first_layer_weights_trainable

        # if custom_first_bias:
        #     bias_values = torch.linspace(b_low, b_high, n_hidden)
        #     with torch.no_grad():
        #         self.input_layer.bias.copy_(bias_values)
        #     self.input_layer.bias.requires_grad = False


        if b_low is not None and b_high is not None:
            bias_vals = torch.linspace(b_low, b_high, n_hidden)
            with torch.no_grad(): inp.bias.copy_(bias_vals)
            inp.bias.requires_grad = manual_bias_requires_grad
        self.layers.append(inp)
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=en_bias))

        # optional post-activation bias list
        if post_activation_bias is not None:
            size = n_hidden if post_activation_bias == 'per_neuron' else 1
            self.post_activation_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(size)) for _ in range(n_layers)
            ])
        else:
            self.post_activation_bias = None
        self.register_buffer('post_activation_bias_scale', torch.tensor(post_activation_bias_scale, dtype=torch.float32))

        # output layer
        self.output_layer = nn.Linear(n_hidden, n_outs, bias=en_bias)
        _init_linear(self.output_layer, out_layer_init)

        self.full_layer_list = self.layers + nn.ModuleList([self.output_layer])
        self.input_layer = inp # useid in upper modules


    def _pre_activation(self, idx, x):
        return self.layers[idx](x)

    def forward(self, x):
        # apply prescaling
        x = x * self.prescaling
        # skip connection residual
        x_res = torch.sum(self.skip_gain * x, dim=-1, keepdim=True)
        post_acts = []
        for idx in range(len(self.layers)):
            lin = self._pre_activation(idx, x)
            x = self.activation(lin)
            if self.post_activation_bias is not None:
                x = x + self.post_activation_bias[idx] * self.post_activation_bias_scale
            post_acts.append(x)
        out = self.main_gain * self.output_layer(x) + x_res
        if self.return_post_acts:
            return out, post_acts
        else:
            return out
        
    def reset_state(self):
        pass


# Utility for linear layer initialization

def _init_linear(layer: nn.Linear, mode: str):
    if mode == 'ones':
        nn.init.constant_(layer.weight, 1.0)
    elif mode == 'uniform_unity':
        nn.init.uniform_(layer.weight, -1, 1)
    elif mode == 'zeros':
        nn.init.constant_(layer.weight, 0.0)
    # default: leave


# SingleStepRNN overrides _pre_activation to add recurrence optionally per layer
class SingleStepRNN(MLP):
    def __init__(
        self,
        *args,
        recurrence_mask=None,
        recurrence_init=None,
        **kwargs
    ):
        """
        recurrence_mask: list or tensor of booleans of length n_layers.
        If True at idx, adds recurrence for that layer; else skip recurrence.
        By default, recurrence on all layers.
        """
        # verify that return_post_acts is True or unset
        if 'return_post_acts' in kwargs:
            if kwargs['return_post_acts'] is False:
                raise ValueError("SingleStepRNN requires return_post_acts=True")
        else:
            kwargs['return_post_acts'] = True
        # call parent constructor
        super().__init__(*args, **kwargs)
        L = len(self.layers)
        # Default mask: all True
        if recurrence_mask is None:
            recurrence_mask = [True] * L
        if len(recurrence_mask) != L:
            raise ValueError(f"recurrence_mask must be length {L}")
        # store mask
        self.recurrence_mask = torch.tensor(recurrence_mask, dtype=torch.bool)
        self.recurrence_init = recurrence_init
        hidden_size = self.layers[0].out_features
        # one U matrix per layer
        self.U_mats = nn.ParameterList([
            nn.Parameter(self._init_recurrence((hidden_size,hidden_size))) if self.recurrence_mask[i] else None
            for i in range(L)
        ])
        self.reset_state()

    def reset_state(self):
        # clear previous post-activations
        self.prev_postacts = [None] * len(self.layers)

    def _init_recurrence(self, size):
        s = size[0] 
        if self.recurrence_init is None:
            return torch.zeros(size)
        elif isinstance(self.recurrence_init, torch.Tensor):
            if self.recurrence_init.shape != size:
                raise ValueError(f"recurrence_init tensor must be same shape as layer size {size}")
            return self.recurrence_init
        elif isinstance(self.recurrence_init, np.ndarray):
            if self.recurrence_init.shape != size:
                raise ValueError(f"recurrence_init tensor must be same shape as layer size {size}")
            return torch.tensor(self.recurrence_init, dtype=torch.float32)
        elif self.recurrence_init == 'uniform':
            return (2*torch.rand(size)-1)/np.sqrt(s)
        elif self.recurrence_init == 'normal':
            return torch.randn(size)/np.sqrt(s)
        elif self.recurrence_init == 'zeros':
            return torch.zeros(size)
        else:
            raise ValueError(f"Unknown recurrence_init: {self.recurrence_init}")

    def _pre_activation(self, idx, x):
        # base linear
        lin = super()._pre_activation(idx, x)
        # add recurrence only if mask True
        if self.recurrence_mask[idx] and self.prev_postacts[idx] is not None:
            lin = lin + F.linear(self.prev_postacts[idx], self.U_mats[idx])
        return lin

    def forward(self, x):
        out, post_acts = super().forward(x)
        # store for next step (detach to prevent BPTT beyond single-step)
        self.prev_postacts = [h.detach() for h in post_acts]
        return out, post_acts


class LegacyMLP(nn.Module, ModelForRunner):
    def __init__(self, n_inputs=None, n_hidden=None, n_outs=None, n_layers=1, nl='tanh', en_bias=True, prescaling=None, main_gain=None,
                 b_low=None, b_high=None, first_layer_init='default', skip_gain=None, first_layer_weights_trainable=False, out_layer_init='default',
                 post_activation_bias=None, post_activation_bias_scale=1,
                 info=None):
        super(LegacyMLP, self).__init__()
        

        if prescaling is not None:
            #ensure prescaling is same size as n_inputs
            if len(prescaling) != n_inputs:
                raise ValueError('prescaling must be same size as n_inputs')
            prescaling = torch.tensor(prescaling, dtype=torch.float32, requires_grad=False)
        else:
            prescaling = torch.ones(n_inputs, dtype=torch.float32, requires_grad=False)
        #prescaling is not a layer and therefre should be registered as buffer
        self.register_buffer('prescaling', prescaling)

        if main_gain  is not None:
            #ensure outscaling is same size as n_outs
            if n_outs == 1 and not isinstance(main_gain, list) and not isinstance(main_gain, np.ndarray):
                main_gain = [main_gain]
            if len(main_gain) != n_outs:
                raise ValueError('outscaling must be same size as n_outs')
            main_gain  = torch.tensor(main_gain, dtype=torch.float32, requires_grad=False)
        else:
            main_gain  = torch.ones(n_outs, dtype=torch.float32, requires_grad=False)

        skip_gain = skip_gain if skip_gain is not None else 0  # Default gain is 0

        if isinstance(skip_gain, list) or isinstance(skip_gain, np.ndarray):
            skip_gain = torch.tensor(skip_gain, dtype=torch.float32, requires_grad=False)
        else:
            skip_gain = torch.tensor([skip_gain]*n_outs, dtype=torch.float32, requires_grad=False)

        #prescaling, main_gain, skip_gain are not layers and therefre should be registered as buffer
        self.register_buffer('prescaling', prescaling)
        self.register_buffer('main_gain', main_gain)
        self.register_buffer('skip_gain', skip_gain)

        nl = nl_selector(nl)
        self.activation = nl()

        # Input layer
        if b_low is None and b_high is not None:
            b_low = -b_high

        custom_first_bias = b_low is not None and b_high is not None

        self.n_layers = n_layers

        self.input_layer = nn.Linear(n_inputs, n_hidden, bias=en_bias or custom_first_bias)
        
        if first_layer_init == 'ones':
            nn.init.constant_(self.input_layer.weight, 1.0)          
        elif first_layer_init == 'uniform_unity':
            nn.init.uniform_(self.input_layer.weight, -1, 1)
        elif first_layer_init == 'default':
            pass
        else:
            raise ValueError('Unknown first_layer_init')

        self.input_layer.weight.requires_grad = first_layer_weights_trainable

        if custom_first_bias:
            bias_values = torch.linspace(b_low, b_high, n_hidden)
            with torch.no_grad():
                self.input_layer.bias.copy_(bias_values)
            self.input_layer.bias.requires_grad = False

        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_hidden, n_hidden, bias=en_bias) for _ in range(n_layers - 1)
        ])

        
        if post_activation_bias is not None:
            if post_activation_bias == 'per_neuron':
                self.post_activation_bias = nn.ParameterList([nn.Parameter(torch.zeros(n_hidden)) for _ in range(n_layers)])
            elif post_activation_bias == 'per_layer':
                self.post_activation_bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_layers)])
            else:
                raise ValueError('Unknown post_activation_bias')
        else:
            self.post_activation_bias = None

        post_activation_bias_scale = torch.tensor(post_activation_bias_scale, dtype=torch.float32, requires_grad=False)
        self.register_buffer('post_activation_bias_scale', post_activation_bias_scale)


        self.output_layer = nn.Linear(n_hidden, n_outs, bias=en_bias)
        self.full_layer_list  = nn.ModuleList([self.input_layer]) + self.hidden_layers + nn.ModuleList([self.output_layer]) #for compatibility with tester

        if out_layer_init == 'ones':
            nn.init.constant_(self.output_layer.weight, 1.0)
        elif out_layer_init == 'uniform_unity':
            nn.init.uniform_(self.output_layer.weight, -1, 1)
        elif out_layer_init == 'zeros':
            nn.init.constant_(self.output_layer.weight, 0.0)
        elif out_layer_init == 'default':
            pass

    def forward(self, x):
        
        if self.prescaling is not None:
            x = x * self.prescaling

        x_res = torch.sum(self.skip_gain *x, dim=-1, keepdim=True)

        x = self.input_layer(x)
        x = self.activation(x)

        if self.post_activation_bias is not None:
            x = x + self.post_activation_bias[0] * self.post_activation_bias_scale

        for l, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)
            if self.post_activation_bias is not None:
                x = x + self.post_activation_bias[l+1]* self.post_activation_bias_scale
        
        x = self.main_gain  * self.output_layer(x) +  x_res  
        return x
    
    def reset_state(self):
        pass
    
    


class SymReLU(torch.nn.Module):
    def __init__(self, threshold=1):
        super(SymReLU, self).__init__()
        self.threshold = threshold
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.threshold - torch.abs(x))


class OneOverSqr(torch.nn.Module):
    def __init__(self, w=1, c=0, normalize=False):
        super(OneOverSqr, self).__init__()
        self.w = w
        self.c = c
        self.normalize = normalize
        self.norm_factor = 1.0 / (2 * w) if normalize else 1.0

    def forward(self, x):
        return self.norm_factor*(1 + (x / self.w).pow(2)).reciprocal() + self.c


class GaussianKernel(torch.nn.Module):
    def __init__(self, w=1, c=0, normalize=False):
        super(GaussianKernel, self).__init__()
        self.w = w
        self.c = c
        self.normalize = normalize
        self.norm_factor = 1.0 / (2 * w) if normalize else 1.0

    def forward(self, x):
        return self.norm_factor*torch.exp(-0.5 * (x / self.w).pow(2)) + self.c
    
class RectangularKernel(torch.nn.Module):
    def __init__(self, w=1, c=0, normalize=False):
        super(RectangularKernel, self).__init__()
        self.w = w
        self.c = c
        self.normalize = normalize
        self.norm_factor = 1.0 / (2 * w) if normalize else 1.0

    def forward(self, x):
        return self.norm_factor*(torch.abs(x) <= self.w).float() + self.c


class DualRateModel(ModelForRunner):
    def __init__(self, a_s: float, b_s: float, a_f: float, b_f: float, input_proj_vec=None, info=None):
        """
        Initialize the dual-rate model parameters.
        :param a_s: Retention factor for the slow process (close to 1)
        :param b_s: Learning rate for the slow process (small value)
        :param a_f: Retention factor for the fast process (close to 0)
        :param b_f: Learning rate for the fast process (larger than b_s)
        """
        self.a_s = a_s
        self.b_s = b_s
        self.a_f = a_f
        self.b_f = b_f
        self.input_proj_vec = np.array(input_proj_vec) if input_proj_vec is not None else None
        self.reset_state()

    def reset_state(self):
        """Reset the model's adaptation states."""
        self.x_s = 0.0  # Slow process state
        self.x_f = 0.0  # Fast process state

    def step(self, e: float) -> float:
        """
        Update the model's states given an error signal e.
        :param e: The error signal at the current step.
        :return: The total adaptation response.
        """
        if self.input_proj_vec is not None:
            e = np.dot(self.input_proj_vec, e)
        self.x_s = self.a_s * self.x_s + self.b_s * e
        self.x_f = self.a_f * self.x_f + self.b_f * e
        return self.x_s + self.x_f
    
    def current_state(self) -> float:
        """ Returns the current state of the model. """
        return self.x_s + self.x_f

    def __call__(self, e: float) -> float:
        """
        Allows the model to be called as model(e), equivalent to model.step(e).
        :param e: The error signal.
        :return: The total adaptation response.
        """
        return self.step(e)


class SingleRateFlexModel(ModelForRunner):
    def __init__(self, a: float, b: float, c=1, d=1, decay_floor=None, input_proj_vec=None, info=None):
        """
        Initialize the dual-rate model parameters.
        :param a: Retention factor
        :param b: Learning rate 
        :param c: Retention factor for the fast process (close to 0)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d  
        self.decay_floor = decay_floor
        self.input_proj_vec = np.array(input_proj_vec) if input_proj_vec is not None else None
        self.reset_state()

    def reset_state(self):
        self.x = 0.0  

    def step(self, e: float) -> float:

        if self.input_proj_vec is not None:
            e = np.dot(self.input_proj_vec, e)
        x_sign = np.sign(self.x)
        e_sign = np.sign(e)
        self.x = self.x + e_sign * self.b * np.abs(e)**self.c -  x_sign * (1-self.a) *np.abs(self.x) ** self.d
        
        if self.decay_floor is not None:
            if x_sign > 0:
                self.x = max(self.x, self.decay_floor)
            elif x_sign < 0:
                self.x = min(self.x, -self.decay_floor)

        return self.x
    
    def current_state(self) -> float:
        """ Returns the current state of the model. """
        return self.x

    def __call__(self, e: float) -> float:
        """
        Allows the model to be called as model(e), equivalent to model.step(e).
        :param e: The error signal.
        :return: The total adaptation response.
        """
        return self.step(e)

class Herzfeld14Model(ModelForRunner):
    def __init__(self, sigma=0.25, beta=0.01, alpha=1, basis_interval=[-3,3], n_basis_el=20, eta_ini=0.2, input_proj_vec=None, info=None):
        """
        Initialize the Herzfeld14 model parameters.
        """
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.basis_interval = basis_interval
        self.n_basis_el = n_basis_el
        self.eta_ini = eta_ini

        self.input_proj_vec = np.array(input_proj_vec) if input_proj_vec is not None else None
        self.reset_state()

    def reset_state(self):
        """Reset the model's adaptation states."""
        self.e_base = np.linspace(self.basis_interval[0], self.basis_interval[1], self.n_basis_el)
        self.w = np.ones(self.n_basis_el)
        
        g_at_zero = self.kernel_(0)
        eta_ = np.sum(self.w * g_at_zero)
        self.w = self.w / eta_ * self.eta_ini
        self.x = 0.0
        self.e_tm1 = 0.0

    def step(self, e: float) -> float:
        """
        Update the model's states given an error signal e.
        :param e: The error signal at the current step.
        :return: The total adaptation response.
        """
        if self.input_proj_vec is not None:
            e = np.dot(self.input_proj_vec, e)

        g = self.kernel_(e)
        eta = np.sum(self.w * g)
        self.w = self.w + self.beta * np.sign(e*self.e_tm1) * g / np.sum(g**2)
        self.e_tm1 = e
        return self.alpha * self.x + eta * e

    def kernel_(self, e):
        return np.exp(-(e-self.e_base)**2/(2*self.sigma**2))

    def __call__(self, e: float) -> float:
        """
        Allows the model to be called as model(e), equivalent to model.step(e).
        :param e: The error signal.
        :return: The total adaptation response.
        """
        return self.step(e)



class MLP_minimal(nn.Module, ModelForRunner):
    def __init__(self, n_inputs=None, n_hidden=None, n_outs=None, n_layers=1, nl='tanh', en_bias=True, prescaling=None, main_gain=None,
                 b_low=None, b_high=None, first_layer_init='default', skip_gain=None, first_layer_weights_trainable=False, out_layer_init='default',
                 info=None):
        super(MLP_minimal, self).__init__()
        

        nl = nl_selector(nl)
        self.activation = nl()

        # Input layer
        if b_low is None and b_high is not None:
            b_low = -b_high

        custom_first_bias = b_low is not None and b_high is not None

        self.n_layers = n_layers

        self.input_layer = nn.Linear(n_inputs, n_hidden, bias=en_bias or custom_first_bias)
        
        if first_layer_init == 'ones':
            nn.init.constant_(self.input_layer.weight, 1.0)          
        elif first_layer_init == 'uniform_unity':
            nn.init.uniform_(self.input_layer.weight, -1, 1)
        elif first_layer_init == 'default':
            pass
        else:
            raise ValueError('Unknown first_layer_init')

        self.input_layer.weight.requires_grad = first_layer_weights_trainable

        if custom_first_bias:
            bias_values = torch.linspace(b_low, b_high, n_hidden)
            with torch.no_grad():
                self.input_layer.bias.copy_(bias_values)
            self.input_layer.bias.requires_grad = False

        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_hidden, n_hidden, bias=en_bias) for _ in range(n_layers - 1)
        ])

        self.output_layer = nn.Linear(n_hidden, n_outs, bias=en_bias)

        if out_layer_init == 'ones':
            nn.init.constant_(self.output_layer.weight, 1.0)
        elif out_layer_init == 'uniform_unity':
            nn.init.uniform_(self.output_layer.weight, -1, 1)
        elif out_layer_init == 'zeros':
            nn.init.constant_(self.output_layer.weight, 0.0)
        elif out_layer_init == 'default':
            pass

    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x)

        return x
    
    def reset_state(self):
        pass

if __name__ == "__main__":

    torch.manual_seed(42)
    batch_size = 4
    input_dim = 3
    hidden_dim = 8
    output_dim = 1
    x = torch.randn(batch_size, input_dim)

    # 1) Consistency tests for LegacyMLP vs MLP
    test_cases = []
    test_cases.append(((input_dim, hidden_dim, output_dim), {}))
    test_cases.append(((input_dim, hidden_dim, output_dim, 2, 'relu', False), {}))
    test_cases.append(((input_dim, hidden_dim, output_dim),
                       {'prescaling':[1,0.5,2], 'main_gain':[2], 'skip_gain':[0.1,0.2,0.3]}))
    test_cases.append(((input_dim, hidden_dim, output_dim),
                       {'b_low':-1.0, 'b_high':1.0}))
    test_cases.append(((input_dim, hidden_dim, output_dim),
                       {'first_layer_init':'uniform_unity', 'first_layer_weights_trainable':True,
                        'out_layer_init':'zeros', 'post_activation_bias':'per_neuron'}))
    for pos_args, kwargs in test_cases:
        print(f"Testing with args: {pos_args}, kwargs: {kwargs}")
        legacy = LegacyMLP(*pos_args, **kwargs)
        new = MLP(*pos_args, **kwargs)
        out1 = legacy(x)
        out2 = new(x)
        # print(f'LegacyMLP output: {out1}')
        # print(f'Modern MLP output: {out2}')
        assert out1.shape == out2.shape == (batch_size, output_dim), "Output shape mismatch"
    print("Consistency tests passed.")

    # 2) Skip connection functional tests
    # 2a) per-input skip_gain
    net_skip_in = MLP(3,5,1, skip_gain=[1,2,3])
    net_skip_in.full_layer_list[0].weight.data.zero_()
    net_skip_in.output_layer.bias.data.zero_()
    net_skip_in.output_layer.weight.data.zero_()
    x3 = torch.tensor([[1.,2.,3.],[0.1,0.1,0.1]])
    # out_in, _ = net_skip_in(x3)
    out_in = net_skip_in(x3)
    expected_in = torch.tensor([[1*1+2*2+3*3],[1*0.1+2*0.1+3*0.1]]).sum(dim=-1, keepdim=True)
    # print(f"Expected output (per-input skip): {expected_in}")
    # print(f"Actual output (per-input skip): {out_in}")
    assert torch.allclose(out_in, expected_in), "Per-input skip failed"
    # # 2b) per-output skip_gain
    # net_skip_out = MLP(3,5,2, skip_gain=[1,2])
    # net_skip_out.layers[0].weight.data.zero_()
    # net_skip_out.output_layer.weight.data.zero_()
    # x4 = torch.tensor([[1.,1.,1.]])
    # out_out, _ = net_skip_out(x4)
    # sum_in = torch.tensor([[3.]])
    # expected_out = torch.cat([sum_in*1, sum_in*2], dim=-1)
    # assert torch.allclose(out_out, expected_out), "Per-output skip failed"

    # 2c) scalar skip_gain
    for construct in [LegacyMLP, MLP]:
        net_skip_sc = construct(4,5,1, skip_gain=0.5)
        net_skip_sc.full_layer_list[0].weight.data.zero_()
        net_skip_sc.output_layer.bias.data.zero_()
        net_skip_sc.output_layer.weight.data.zero_()
        x5 = torch.ones(2,4)
        out_sc = net_skip_sc(x5)
        expected_sc = torch.full((2,1), 0.5 * 4)
        assert torch.allclose(out_sc, expected_sc), f"Scalar skip failed, for {construct.__name__}."
        print(f"Scalar skip tests passed, for {construct.__name__}.")

    # 2d) nonlinearity test. Loop over relu and tanh. Ensure that both models produce the same output.
    for nl in ['relu', 'tanh']:
        outputs = []
        for construct in [LegacyMLP, MLP]:
            net = construct(2,5,1, nl=nl)
            net.full_layer_list[0].weight.data.fill_(1.0)
            net.full_layer_list[0].bias.data.zero_()
            net.output_layer.weight.data.fill_(1.0)
            net.output_layer.bias.data.zero_()
            x6 = torch.tensor([[1.,2.]])
            out = net(x6)
            outputs.append(out)
        assert torch.allclose(outputs[0], outputs[1]), f"Nonlinearity test failed, for {construct.__name__} with nl={nl}."
        print(f"Nonlinearity tests passed, for {construct.__name__} with nl={nl}.")

    # 3) Prescaling + main_gain + activation
    for construct in [LegacyMLP, MLP]:
        net_pm = construct(2,11,1, prescaling=[2,3], main_gain=[2], skip_gain=0, nl='relu')
        net_pm.full_layer_list[0].weight.data.fill_(1.0)
        net_pm.full_layer_list[0].bias.data.zero_()
        net_pm.output_layer.weight.data.fill_(1.0)
        net_pm.output_layer.bias.data.zero_()
        x2 = torch.tensor([[1.,1.],[2.,0.]])
        # out_pm, _ = net_pm(x2)
        out_pm = net_pm(x2)

        # prescale -> [[2,3],[4,0]] sum-> [5,4] -> ReLU([5,4]) -> *2
        manual = torch.tensor([[110.],[88.]])

        # print(f"Expected output (prescale + main_gain): {manual}")
        # print(f"Actual output (prescale + main_gain): {out_pm}")
        assert torch.allclose(out_pm, manual), f"Prescale/main_gain test failed, for {construct.__name__}."
        print(f"Prescaling/main_gain tests passed, for {construct.__name__}.")

    # 4) Recurrence in SingleStepRNN
    rnn = SingleStepRNN(1,1,1, n_layers=1, recurrence_mask=[True], skip_gain=0)
    rnn.layers[0].weight.data.fill_(1.0); rnn.layers[0].bias.data.zero_()
    rnn.output_layer.weight.data.fill_(1.0); rnn.output_layer.bias.data.zero_()
    rnn.U_mats[0].data.fill_(1.0)
    x1 = torch.ones(2,1)
    out1, _ = rnn(x1)
    out2, _ = rnn(x1)
    assert (out2 > out1).all(), "Recurrence did not accumulate"
    print("Recurrence tests passed.")

    #testing some more complex configurations
    w = 1
    skip = 0.5
    model_construct_args =  dict(n_inputs = 4,
                          n_hidden = 5*4*512,
                          n_outs = 1,
                          en_bias = False,
                         b_high=3, first_layer_init='ones',
                        first_layer_weights_trainable = True,
                        out_layer_init='zeros',
                          nl = lambda : OneOverSqr(w=w), skip_gain= skip)
    
    outs = []
    for construct in [LegacyMLP, MLP]:
        print(f"Testing {construct.__name__} with OneOverSqr")
        model = construct(**model_construct_args)
        model.full_layer_list[0].weight.data.fill_(1.0)
        model.full_layer_list[0].bias.data.zero_()
        model.full_layer_list[1].weight.data.fill_(1.0)
        # model.layers[-1].bias.data.zero_()
        x = torch.ones(2,4)
        out = model(x)
        # print(f'output for {construct.__name__}: {out}')
        outs.append(out)
    assert torch.allclose(outs[0], outs[1]), f"OneOverSqr test failed."
    print("OneOverSqr tests passed.")
            

    print("All tests passed successfully.")


