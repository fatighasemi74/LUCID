"""
models/likelihood.py
====================
Gaussian likelihood  L_φ(Y | X, Z)  for the forward model

    Y = H_φ(X, Z) + Σ^{1/2} ε,    ε ~ N(0, I_m)

so

    L_φ(Y | X, Z) = N(Y ; H_φ(X, Z), Σ)

Critical design note — NO θ dependence
---------------------------------------
The likelihood L_φ(Y | X, Z) does NOT depend on θ (the diffusion prior
parameters). Once X is given, θ is already accounted for through the prior
p(X | θ). Mixing θ into the likelihood is a modelling error. This module
enforces that boundary: it takes (x, z, phi) and returns a log-likelihood;
it never sees θ.

Forward operator H_φ
---------------------
    H_φ(X, Z) = a · S_ω[ h_η * r_ψ(X) ]_{· − τ} + b

Components and their status across parts:

    r_ψ(X)   : vascular response applied to ECG.
                Part 1 — fixed (identity or fixed IR); ψ not estimated.
                Part 2 — learned; ψ ∈ φ.

    h_η * ·  : convolution with Gaussian lowpass kernel of bandwidth η.
                Part 1 — fixed known Gaussian; η fixed.
                Part 2 — η ∈ φ (estimated by VI).

    S_ω[·]   : residual sensor correction.
                Part 1 — identity (S_ω = id); ω not estimated.
                Part 2 — ω ∈ φ.

    τ        : time delay (shift).
                Part 1 — estimated (in φ̂; first identifiable parameter).
                Part 2 — estimated.

    a, b     : amplitude and offset scalars.
                Part 1 — estimated (in φ̂).
                Part 2 — estimated.

    Σ        : noise covariance (NoiseCov instance).
                Part 1 — isotropic, single σ² estimated by VI.
                Part 2 — diagonal, per-timestep σ_i².

φ in Part 1 = {τ, a, b, σ²}  (minimal identifiable subset first,
then progressively η, ψ, ω per the controlled-difficulty schedule).

Ownership
---------
This module is owned by Fatemeh (Part 1 / VI likelihood).
The NoiseCov instance lives here (likelihood owns Σ).
VI (inference/vi.py) reads and updates the variational parameters λ
that approximate the posterior over φ.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.noise_cov import NoiseCov, CovMode


# ======================================================================
# Forward operator components
# ======================================================================

def gaussian_kernel_1d(sigma: float, kernel_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns a normalized 1-D Gaussian convolution kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian in samples.
    kernel_size : int
        Length of the kernel (should be odd; truncation at ±3σ is typical).
    device : torch.device

    Returns
    -------
    torch.Tensor, shape (kernel_size,)
        Normalized so entries sum to 1.
    """
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return k / k.sum()


def apply_lowpass(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Apply 1-D convolution (lowpass) to signal x.

    Parameters
    ----------
    x : torch.Tensor, shape (B, 1, L)
        Input signal (ECG or intermediate).
    kernel : torch.Tensor, shape (K,)
        1-D convolution kernel (from gaussian_kernel_1d).

    Returns
    -------
    torch.Tensor, shape (B, 1, L)
        Filtered signal, same length (same-padding).
    """
    K = kernel.shape[0]
    pad = K // 2
    k = kernel.view(1, 1, K)
    return F.conv1d(x, k, padding=pad)


def apply_delay(x: torch.Tensor, tau: float, fs: float) -> torch.Tensor:
    """
    Apply a fractional-sample time delay τ (in seconds) by circular shift.
    For Part 1, τ is treated as an integer number of samples (floor).
    Sub-sample interpolation is a Part 2 refinement.

    Parameters
    ----------
    x : torch.Tensor, shape (B, 1, L)
    tau : float
        Delay in seconds. Converted to samples as round(tau * fs).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    torch.Tensor, shape (B, 1, L)
    """
    shift = int(round(tau * fs))
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-1)


# ======================================================================
# Likelihood module
# ======================================================================

class GaussianLikelihood(nn.Module):
    """
    Gaussian likelihood  L_φ(Y | X, Z) = N(Y ; H_φ(X, Z), Σ).

    This class owns:
    *   the forward operator H_φ (fixed components in Part 1)
    *   the noise covariance Σ (NoiseCov instance)
    *   the differentiable parameters φ = {log_a, b, τ, ...}

    It does NOT own or reference θ (diffusion prior parameters).

    Parameters
    ----------
    obs_dim : int
        Observation (PPG window) length m.
    fs : float
        Sampling frequency in Hz (for delay conversion).
    cov_mode : CovMode
        Noise covariance mode, passed to NoiseCov.
    kernel_sigma : float
        Gaussian kernel bandwidth η in samples (fixed in Part 1).
    kernel_size : int
        Gaussian kernel length (fixed in Part 1, should be odd).

    Learnable φ parameters (Part 1 minimal set)
    --------------------------------------------
    log_a : nn.Parameter, scalar
        Log of amplitude scaling a > 0 (log ensures positivity).
    b : nn.Parameter, scalar
        Additive offset.
    tau : nn.Parameter, scalar
        Time delay in seconds.
        (Gradient through tau requires sub-sample interpolation in Part 2;
        in Part 1 we treat tau as a discrete shift and estimate it via VI,
        not through autograd directly — see inference/vi.py stub.)
    noise_cov : NoiseCov
        Owns log_diag (σ² or σ_i²).

    Notes
    -----
    *   In Part 1 the kernel (h_η), sensor correction (S_ω = id), and
        vascular response (r_ψ = id) are all fixed. Only {a, b, τ, σ²}
        are in the variational approximation q_λ(φ).
    *   In Part 2, η, ψ, ω enter φ and this class is extended.
    """

    def __init__(
        self,
        obs_dim: int,
        fs: float,
        cov_mode: CovMode | str = CovMode.ISOTROPIC,
        kernel_sigma: float = 10.0,
        kernel_size: int = 61,
    ) -> None:
        super().__init__()

        self.obs_dim     = obs_dim
        self.fs          = fs
        self.kernel_size = kernel_size

        # Fixed kernel (Part 1): registered as buffer, not a parameter
        k = gaussian_kernel_1d(kernel_sigma, kernel_size, device=torch.device("cpu"))
        self.register_buffer("kernel", k)

        # Learnable φ — Part 1 minimal set
        self.log_a = nn.Parameter(torch.tensor(0.0))   # a = exp(log_a), init a=1
        self.b     = nn.Parameter(torch.tensor(0.0))   # offset, init b=0
        self.tau   = nn.Parameter(torch.tensor(0.0))   # delay in seconds, init τ=0

        # Noise covariance Σ — owned here, updated by VI
        self.noise_cov = NoiseCov(dim=obs_dim, mode=cov_mode)

    # ------------------------------------------------------------------
    # Forward operator H_φ
    # ------------------------------------------------------------------

    def forward_operator(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply H_φ(X, Z) to produce the predicted PPG mean.

        H_φ(X, Z) = a · S_ω[ h_η * r_ψ(X) ]_{· − τ} + b

        In Part 1:  r_ψ = identity,  S_ω = identity,
                    h_η = fixed Gaussian kernel,
                    τ, a, b are the learnable parameters.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, L)
            ECG signal (or posterior sample of X).
        z : torch.Tensor or None
            Nuisance parameter Z. In Part 1 this carries τ if τ is treated
            as a random variable with prior ρ rather than a point estimate.
            Pass None in Part 1 fixed-τ mode.

        Returns
        -------
        torch.Tensor, shape (B, 1, L)
            Predicted PPG H_φ(X, Z), same length as x.

        Notes
        -----
        The returned tensor is differentiable w.r.t. log_a, b, and the
        kernel buffer (for Part 2 when η becomes learnable). Gradient
        through tau (discrete shift) is zero almost everywhere; see vi.py
        for the variational treatment of τ.
        """
        # Step 1: vascular response r_ψ — identity in Part 1
        h = x                                       # (B, 1, L)

        # Step 2: lowpass convolution h_η *
        h = apply_lowpass(h, self.kernel)           # (B, 1, L)

        # Step 3: sensor correction S_ω — identity in Part 1
        # (no-op)

        # Step 4: time delay τ
        tau_val = self.tau.item() if z is None else self._tau_from_z(z)

        h = apply_delay(h, tau_val, self.fs)        # (B, 1, L)

        # Step 5: amplitude and offset
        a = self.log_a.exp()
        h = a * h + self.b                          # (B, 1, L)

        return h

    def _tau_from_z(self, z: torch.Tensor) -> float:
        """
        Extract τ from Z = [τ, a, b].
        Z[0] = tau, Z[1] = a, Z[2] = b.
        """
        return float(z.squeeze()[0])
    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute log L_φ(Y | X, Z) = log N(Y ; H_φ(X,Z), Σ).

        = -0.5 * [m log(2π) + log|Σ| + (y - H_φ(x,z))^T Σ^{-1} (y - H_φ(x,z))]

        Parameters
        ----------
        y : torch.Tensor, shape (B, obs_dim) or (B, 1, L)
            Observed PPG.
        x : torch.Tensor, shape (B, 1, L)
            ECG sample (current MALA state or posterior sample).
        z : torch.Tensor or None
            Nuisance parameters Z. None in fixed-operator Part 1 mode.

        Returns
        -------
        torch.Tensor, shape (B,)
            Per-window log-likelihood. Mean over B for the ELBO.

        Notes
        -----
        *   Does NOT depend on θ. θ enters only through p(X|θ) in the prior.
        *   Gradient w.r.t. x is used by MALA for the likelihood score.
        *   Gradient w.r.t. log_a, b, noise_cov.log_diag is used by VI.
        """
        y_flat = y.view(y.shape[0], -1)                        # (B, m)
        mu     = self.forward_operator(x, z).view(y.shape[0], -1)  # (B, m)
        resid  = y_flat - mu                                   # (B, m)

        log_norm = -0.5 * (
            self.obs_dim * math.log(2 * math.pi)
            + self.noise_cov.log_det()
        )
        log_exp = -0.5 * self.noise_cov.mahal(resid)           # (B,)
        return log_norm + log_exp                              # (B,)

    def score_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Likelihood score w.r.t. X: ∇_X log L_φ(Y | X, Z).

        Used by the MALA sampler (inference/mala.py) as the likelihood
        gradient term in the Langevin drift.

        Parameters
        ----------
        y : torch.Tensor, shape (B, 1, L)
        x : torch.Tensor, shape (B, 1, L), requires_grad=True
        z : torch.Tensor or None

        Returns
        -------
        torch.Tensor, shape (B, 1, L)
            Gradient of log-likelihood w.r.t. x. Same shape as x.
        """
        x_ = x.detach().requires_grad_(True)
        ll  = self.log_likelihood(y, x_, z).sum()
        return torch.autograd.grad(ll, x_)[0]

    # ------------------------------------------------------------------
    # Simulation (Part 1 forward pass)
    # ------------------------------------------------------------------

    def simulate(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate synthetic PPG observations Y = H_φ(X,Z) + Σ^{1/2} ε.

        Used in experiments/part1/generate.py to build the semi-synthetic
        dataset with known ground truth.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, L)
            Ground-truth ECG windows.
        z : torch.Tensor or None
        n_samples : int
            Number of noise realisations per window.

        Returns
        -------
        torch.Tensor, shape (n_samples, B, 1, L)
            Synthetic PPG observations.
        """
        with torch.no_grad():
            mu = self.forward_operator(x, z)        # (B, 1, L)
        ys = []
        for _ in range(n_samples):
            noise = self.noise_cov.sample_noise((x.shape[0],))   # (B, m)
            noise = noise.view_as(mu)
            ys.append(mu + noise)
        return torch.stack(ys, dim=0)               # (n_samples, B, 1, L)
