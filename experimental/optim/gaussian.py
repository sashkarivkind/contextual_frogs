import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

class GaussianParams(nn.Module):
    """
    OCaml:
    type 'a p = { mu : 'a ; sigma12 : 'a }
    """
    def __init__(self, mu: torch.Tensor, sigma12: torch.Tensor):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.sigma12 = nn.Parameter(sigma12)  # unconstrained (masked to tril at use-sites)

    @staticmethod
    def tril_from_sigma12(sigma12: torch.Tensor) -> torch.Tensor:
        # OCaml uses: mask * sigma12 where mask is lower-triangular ones.
        # We directly take the lower-triangular part.
        return torch.tril(sigma12)

class Gaussian:
    @staticmethod
    def log_density(theta: GaussianParams, x: torch.Tensor) -> torch.Tensor:
        """
        Log N(x | mu, Sigma), where we parameterize via an arbitrary sigma12 and
        only use its lower-triangular part 'ell' as a Cholesky-like factor.
        OCaml builds e by solving (e * ell^T) = (x - mu).
        """
        # ell is lower triangular
        ell = GaussianParams.tril_from_sigma12(theta.sigma12)
        dim = ell.shape[-1]

        # batch size inferred from x
        bs = x.shape[0]
        d = x - theta.mu  # [bs, dim] (mu broadcasts)
        # Solve for e in e * ell^T = d  =>  left-solve with ell on transposed system
        e = Gaussian.linsolve_triangular_right(ell.transpose(-1, -2), d, upper=True)

        quadratic_term = (-0.5 / float(bs)) * torch.sum(e ** 2)
        # log-det term: -0.5 * log det(Sigma) = - sum(log(diag(ell)))
        diag_ell = torch.diagonal(ell, offset=0, dim1=-2, dim2=-1)
        # avoid log(0)
        diag_ell_safe = torch.clamp(diag_ell.abs(), min=1e-12)
        log_det_term = -torch.sum(torch.log(diag_ell_safe)) 
        const_term = -0.5 * dim * math.log(2.0 * math.pi)
        return const_term + quadratic_term + log_det_term

    @staticmethod
    def gaussian_kl_full_vs_diag(mu_p: torch.Tensor,
                                 sigma12_p: torch.Tensor,
                                 diag_sigma2_q: torch.Tensor) -> torch.Tensor:
        """
        KL( N(mu_p, Sigma_p) || N(0, D) ), with D = diag(diag_sigma2_q):
        0.5 * [ tr(D^{-1} Sigma_p) + mu_p^T D^{-1} mu_p - k + log det D - log det Sigma_p ]
        Using Sigma_p = ell @ ell^T, ell = tril(sigma12_p).

        NOTE(IMPORTANT): the original OCaml code has:
           trace_term = sum (sigma_p * inv_d)
        which *looks* like it multiplies a matrix by a vector elementwise, which would not
        equal tr(D^{-1} Sigma_p). Here I implement the mathematically correct form:
           tr(D^{-1} Sigma_p) = sum_i inv_d[i] * Sigma_p[ii]
        If you intended a different broadcasting behavior, please clarify.
        """
        k = mu_p.shape[-1]
        ell = torch.tril(sigma12_p)
        sigma_p = ell @ ell.transpose(-1, -2)

        eps = 1e-7
        d_vec = diag_sigma2_q + eps
        inv_d = 1.0 / d_vec

        # trace term: sum_i inv_d[i] * Sigma_p[ii]
        trace_term = torch.sum(inv_d * torch.diagonal(sigma_p, offset=0, dim1=-2, dim2=-1))
        # quadratic term: mu^T D^{-1} mu
        quad_term = torch.sum(inv_d * (mu_p ** 2))
        # log det terms
        log_det_q = torch.sum(torch.log(d_vec))

        diag_ell = torch.diagonal(ell, offset=0, dim1=-2, dim2=-1)
        diag_ell_safe = torch.clamp(diag_ell.abs(), min=1e-12)
        log_det_p = torch.sum(torch.log(diag_ell_safe ** 2))  # = 2 * sum log diag(ell)

        kl = 0.5 * (trace_term + quad_term - float(k) + log_det_q - log_det_p)
        return kl

    @staticmethod
    def sample(theta: GaussianParams, batch_size: int) -> torch.Tensor:
        """
        Sample x ~ N(mu, ell ell^T), using z @ ell^T + mu.
        """
        ell = torch.tril(theta.sigma12)
        n = theta.sigma12.shape[-1]
        z = torch.randn(batch_size, n, device=theta.sigma12.device, dtype=theta.sigma12.dtype)
        return theta.mu + z @ ell.transpose(-1, -2)
    
    @staticmethod
    def linsolve_triangular_right(U: torch.Tensor, B: torch.Tensor, upper=True) -> torch.Tensor:
        """
        Solve for X in X * U = B (right-side triangular solve).
        PyTorch only provides left-side solves; so we transpose: (U^T) * X^T = B^T.
        """
        # Solve (U^T) * X^T = B^T
        Xt = torch.linalg.solve_triangular(U.transpose(-1, -2), B.transpose(-1, -2), upper=not upper)
        return Xt.transpose(-1, -2)

if __name__ == "__main__":
    
    torch.manual_seed(0)

    # Problem dimensions
    dim = 5
    bs = 4096
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float64  # use double for more stringent tests

    # Construct a valid lower-triangular 'ell' with positive diagonal
    raw = torch.randn(dim, dim, device=device, dtype=dtype)
    ell = torch.tril(raw)
    diag = F.softplus(torch.diagonal(ell, 0)) + 0.25  # ensure strictly positive
    ell = ell - torch.diag(torch.diagonal(ell, 0)) + torch.diag(diag)

    mu = torch.randn(dim, device=device, dtype=dtype)
    theta = GaussianParams(mu.clone(), ell.clone())

    # --- Log-density test vs PyTorch's MultivariateNormal ---
    print(f'theta = (mu={theta.mu}, sigma12={theta.sigma12})')
    x = Gaussian.sample(theta, bs)
    mvn = torch.distributions.MultivariateNormal(loc=mu, scale_tril=ell)
    ref_lp = mvn.log_prob(x).mean()                # [bs]
    our_lp = Gaussian.log_density(theta, x)        # [scalar] TODO: doublecheck why mean

    max_abs_err = (our_lp - ref_lp).abs().max().item()
    print(f"ref logprob = {ref_lp}")
    print(f"our logprob = {our_lp}")
    print(f"max |logprob error| = {max_abs_err:.3e}")
    assert max_abs_err < 5e-10, "log_density mismatch vs PyTorch MvNormal"

    # --- KL test vs PyTorch (full vs diagonal) ---
    diag_var = F.softplus(torch.randn(dim, device=device, dtype=dtype)) + 0.1
    mvn_p = torch.distributions.MultivariateNormal(loc=mu, scale_tril=ell)
    mvn_q = torch.distributions.MultivariateNormal(loc=torch.zeros(dim, device=device, dtype=dtype),
                                                   covariance_matrix=torch.diag(diag_var))
    ref_kl = torch.distributions.kl.kl_divergence(mvn_p, mvn_q)
    our_kl = Gaussian.gaussian_kl_full_vs_diag(mu, ell, diag_var)
    kl_err = abs((our_kl - ref_kl).item())
    print(f"ref KL(full||diag) = {ref_kl}")
    print(f"our KL(full||diag) = {our_kl}")
    print(f"|KL error| = {kl_err:.3e}")
    assert kl_err < 1e-5, "KL(full||diag) mismatch vs PyTorch"

    # --- Mean/cov sanity from samples ---
    xs = Gaussian.sample(theta, 100_000)
    emp_mean = xs.mean(0)
    emp_cov = torch.cov(xs.T)
    true_cov = ell @ ell.T
    mean_err = (emp_mean - mu).norm().item()
    cov_err = (emp_cov - true_cov).norm().item()
    print(f"empirical mean = {emp_mean}")
    print(f"true mean = {mu}")
    print(f"empirical cov = {emp_cov}")
    print(f"true cov = {true_cov}")
    print(f"mean err={mean_err:.3e}, cov err={cov_err:.3e}")