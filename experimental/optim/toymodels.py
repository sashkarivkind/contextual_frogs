import torch

class ToyObs:
    def __init__(
        self,
        n_steps: int = 340,
        phi: float = 0.9,
        sigma_process: float = 0.1,
        sigma_obs: float = 0.05,
        device = None,  # default to CPU
        dtype: torch.dtype = torch.float32,
        stationary_init: bool = False,
        mask = None,
    ):
        self.T = int(n_steps)
        self.phi = float(phi)
        self.sigma_p = float(sigma_process)
        self.sigma_o = float(sigma_obs)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.stationary_init = bool(stationary_init)
        self.mask = torch.tensor(mask, dtype=torch.bool, device=self.device) if mask is not None else None

        self._obs = None
        self._latent = None

        if self.mask is not None:
            assert len(self.mask) == self.T, "mask length must match n_steps"

    # ---- public API ----
    @torch.no_grad()
    def get_obs(self, seed = None, return_latents: bool = False):
        """
        Simulate observations y_t from the AR(1)/OU-like model:
          x_0 ~ N(0, sigma_p^2) if stationary_init=False (default),
              ~ N(0, sigma_p^2/(1-phi^2)) if True
          x_t = phi x_{t-1} + eta_t,   eta_t ~ N(0, sigma_p^2)
          y_t = x_t + eps_t,           eps_t ~ N(0, sigma_o^2)
        Saves to self._obs (and self._latent) and returns y (and optionally x).
        """
        if seed is not None:
            torch.manual_seed(seed)

        T = self.T
        phi = torch.tensor(self.phi, device=self.device, dtype=self.dtype)
        sigma_p = torch.tensor(self.sigma_p, device=self.device, dtype=self.dtype)
        sigma_o = torch.tensor(self.sigma_o, device=self.device, dtype=self.dtype)

        x = torch.empty(T, device=self.device, dtype=self.dtype)
        # init x_0
        if self.stationary_init:
            var0 = sigma_p**2 / (1 - phi**2)
        else:
            var0 = sigma_p**2
        x[0] = torch.randn((), device=self.device, dtype=self.dtype) * var0.sqrt()

        # evolve
        for t in range(1, T):
            eta = torch.randn((), device=self.device, dtype=self.dtype) * sigma_p
            x[t] = phi * x[t - 1] + eta

        # observations
        eps = torch.randn(T, device=self.device, dtype=self.dtype) * sigma_o
        y = x + eps
        #replace masked values with NaN
        if self.mask is not None:
            y = y.masked_fill(~self.mask, float('nan'))
        self._latent = x
        self._obs = y

        return (y, x) if return_latents else y

    def get_density(self, obs = None, jitter: float = 1e-6) -> torch.Tensor:
        """
        Return log p(obs) under the Gaussian model with the *analytic* covariance.
        If obs is None, uses the last sample from get_obs().
        """
        y = obs if obs is not None else self._obs
        if y is None:
            raise ValueError("No observations available. Call get_obs() or pass obs=...")

        y = y.to(device=self.device, dtype=self.dtype).reshape(-1)
        if y.numel() != self.T:
            raise ValueError(f"obs length {y.numel()} != n_steps {self.T}")

        Sigma = self._obs_covariance()
        # small jitter for numerical stability
        Sigma = Sigma + torch.eye(self.T, device=self.device, dtype=self.dtype) * float(jitter)

        if self.mask is not None:
            # select only the unmasked entries
            Sigma_ = Sigma[self.mask][:, self.mask]
            y = y[self.mask]
            T_ = y.numel()
        else:
            Sigma_ = Sigma
            T_ = self.T

        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(T_, device=self.device, dtype=self.dtype),
            covariance_matrix=Sigma_,
        )
        return mvn.log_prob(y)

    # ---- internals ----
    def _obs_covariance(self) -> torch.Tensor:
        """
        Covariance of y_t = x_t + eps_t:
          Cov(y_t, y_s) = Cov(x_t, x_s) + (sigma_o^2 if t==s else 0)

        If stationary_init:
          Cov(x_t, x_s) = (sigma_p^2 / (1 - phi^2)) * phi^{|t-s|}
        Else (matches your original simulation with x_0 ~ N(0, sigma_p^2)):
          Cov(x_t, x_s) = sigma_p^2 * [ phi^{t+s}
                                        + phi^{|t-s|} * (1 - phi^{2*min(t,s)})/(1 - phi^2) ]
        """
        T = self.T
        device, dtype = self.device, self.dtype
        phi = torch.tensor(self.phi, device=device, dtype=dtype)
        sp2 = torch.tensor(self.sigma_p**2, device=device, dtype=dtype)
        so2 = torch.tensor(self.sigma_o**2, device=device, dtype=dtype)

        idx = torch.arange(T, device=device, dtype=dtype)
        I = idx.unsqueeze(0)  # row indices (1 x T)
        J = idx.unsqueeze(1)  # col indices (T x 1)
        abs_diff = (I - J).abs()
        min_ij = torch.minimum(I, J)

        if self.stationary_init:
            var_x = sp2 / (1 - phi**2)
            Sigma_x = var_x * (phi**abs_diff)
        else:
            # non-stationary covariance (matches x_0 ~ N(0, sigma_p^2))
            term1 = sp2 * (phi ** (I + J))
            term2 = sp2 * (phi ** abs_diff) * (1 - (phi ** (2 * min_ij))) / (1 - phi**2)
            Sigma_x = term1 + term2

        Sigma_y = Sigma_x + torch.eye(T, device=device, dtype=dtype) * so2
        return Sigma_y
