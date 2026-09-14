"""
models/diffusion_core.py
========================
Gaussian diffusion schedule, forward process utilities, and DDIM sampler.

This module is self-contained — it knows nothing about the U-Net architecture,
the likelihood, or the MALA sampler. It only defines the math of the diffusion
process and the reverse sampler.

Compatible with
---------------
*   Window length : 4000 samples (8s at 500 Hz)
*   Prediction    : x0 (the denoiser predicts the clean signal directly)
*   Schedule      : cosine (Nichol & Dhariwal 2021) — better than linear
                    for physiological signals which have structured low-freq content
*   Sampler       : DDIM (Song et al. 2020) — deterministic, 50 steps at inference

Interface with prior.py
-----------------------
ECGDiffusionPrior calls:
    diff = GaussianDiffusion(...)
    x_t  = diff.q_sample(x0, t, noise)      # forward process (training)
    s    = diff.score_from_x0hat(x_t, t, x0hat)  # score for MALA
    x    = ddim_sample(diff, denoise_fn, shape, device)  # prior sampling
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


# ======================================================================
# Schedule
# ======================================================================

def _cosine_schedule(T: int) -> torch.Tensor:
    """
    Cosine noise schedule ᾱ_t, shape (T,).
    Nichol & Dhariwal (2021): ᾱ_t = cos²(((t/T + s)/(1+s)) · π/2) / ᾱ_0
    where s=0.008 prevents ᾱ_t from being too small near t=0.
    """
    s     = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64)
    f     = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    abar  = f / f[0]
    return abar[1:].float()   # shape (T,)


def _linear_schedule(T: int,
                     beta_start: float = 1e-4,
                     beta_end: float   = 0.02) -> torch.Tensor:
    """Linear beta schedule → ᾱ_t, shape (T,)."""
    betas  = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


# ======================================================================
# GaussianDiffusion
# ======================================================================

class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process with configurable schedule.

    Stores all schedule buffers as non-parameter tensors (registered
    with register_buffer so they move with .to(device)).

    Parameters
    ----------
    T : int
        Total diffusion timesteps. Default 1000.
    schedule : str
        "cosine" (recommended) or "linear".
    pred_type : str
        "x0" — denoiser predicts clean signal x̂_0 directly.
               Tweedie formula: x̂_0 = f_θ(x_t, t).
               Score: s_θ = (x̂_0 - x_t) / (1 - ᾱ_t).
        "eps" — denoiser predicts noise ε.
               Tweedie formula: x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t.
               Score: s_θ = -ε̂ / √(1-ᾱ_t).
        Use "x0" for this project — more stable for peaked ECG signals.

    Key buffers (all shape (T,))
    ----------------------------
    alphas_bar          : ᾱ_t
    sqrt_alphas_bar     : √ᾱ_t
    sqrt_one_minus_abar : √(1-ᾱ_t)
    alphas_bar_prev     : ᾱ_{t-1}  (with ᾱ_0 = 1)
    """

    def __init__(
        self,
        T        : int = 1000,
        schedule : str = "cosine",
        pred_type: str = "x0",
    ) -> None:
        super().__init__()
        self.T         = T
        self.pred_type = pred_type

        if schedule == "cosine":
            abar = _cosine_schedule(T)
        elif schedule == "linear":
            abar = _linear_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        abar_prev = torch.cat([torch.ones(1), abar[:-1]])

        self.register_buffer("alphas_bar",          abar)
        self.register_buffer("sqrt_alphas_bar",     abar.sqrt())
        self.register_buffer("sqrt_one_minus_abar", (1.0 - abar).sqrt())
        self.register_buffer("alphas_bar_prev",     abar_prev)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        Gather scalar a[t_i] for each i in the batch and broadcast to `shape`.
        a : (T,)
        t : (B,) long
        returns: (B, 1, 1, ...) broadcastable to shape (B, C, L)
        """
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0    : torch.Tensor,
        t     : torch.Tensor,
        noise : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) I).

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

        Parameters
        ----------
        x0    : (B, 1, L)  clean ECG window
        t     : (B,) long  timestep indices in [0, T)
        noise : (B, 1, L) or None — if None, sampled fresh

        Returns
        -------
        x_t : (B, 1, L)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self._extract(self.sqrt_alphas_bar,     t, x0.shape)
        sma = self._extract(self.sqrt_one_minus_abar, t, x0.shape)
        return sa * x0 + sma * noise

    # ------------------------------------------------------------------
    # Prediction conversions
    # ------------------------------------------------------------------

    def x0_from_eps(
        self,
        x_t : torch.Tensor,
        t   : torch.Tensor,
        eps : torch.Tensor,
    ) -> torch.Tensor:
        """x̂_0 from predicted noise ε̂: x̂_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t"""
        sa  = self._extract(self.sqrt_alphas_bar,     t, x_t.shape)
        sma = self._extract(self.sqrt_one_minus_abar, t, x_t.shape)
        return (x_t - sma * eps) / sa.clamp(min=1e-8)

    def eps_from_x0(
        self,
        x_t : torch.Tensor,
        t   : torch.Tensor,
        x0  : torch.Tensor,
    ) -> torch.Tensor:
        """ε̂ from predicted x̂_0: ε̂ = (x_t - √ᾱ_t · x0) / √(1-ᾱ_t)"""
        sa  = self._extract(self.sqrt_alphas_bar,     t, x_t.shape)
        sma = self._extract(self.sqrt_one_minus_abar, t, x_t.shape)
        return (x_t - sa * x0) / sma.clamp(min=1e-8)

    def to_x0_and_eps(
        self,
        x_t     : torch.Tensor,
        t       : torch.Tensor,
        raw_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw denoiser output to (x̂_0, ε̂) pair regardless of pred_type.
        Both tensors have shape (B, 1, L).
        """
        if self.pred_type == "x0":
            x0  = raw_pred
            eps = self.eps_from_x0(x_t, t, x0)
        elif self.pred_type == "eps":
            eps = raw_pred
            x0  = self.x0_from_eps(x_t, t, eps)
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type!r}")
        return x0, eps

    # ------------------------------------------------------------------
    # Training target
    # ------------------------------------------------------------------

    def target(
        self,
        x0   : torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        What the denoiser should regress against.
        pred_type="x0"  → target is x0
        pred_type="eps" → target is noise
        """
        return x0 if self.pred_type == "x0" else noise

    # ------------------------------------------------------------------
    # Score function
    # ------------------------------------------------------------------

    def score_from_x0hat(
        self,
        x_t   : torch.Tensor,
        t     : torch.Tensor,
        x0hat : torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate score ∇_x log p_t(x) from the Tweedie estimate x̂_0.

            s_θ(x_t, t) = (x̂_0 - x_t) / (1 - ᾱ_t)

        This is the score function used by the MALA sampler in
        inference/mala.py (via ECGDiffusionPrior.score_x).

        Parameters
        ----------
        x_t   : (B, 1, L)
        t     : (B,) long
        x0hat : (B, 1, L)  Tweedie posterior mean

        Returns
        -------
        score : (B, 1, L)
        """
        one_minus_abar = 1.0 - self._extract(self.alphas_bar, t, x_t.shape)
        return (x0hat - x_t) / one_minus_abar.clamp(min=1e-6)


# ======================================================================
# DDIM sampler
# ======================================================================

def _make_ddim_timesteps(T: int, steps: int, device: torch.device) -> torch.Tensor:
    """
    Uniformly spaced DDIM timestep indices in [0, T-1], shape (steps,).
    Reversed (from T-1 down to 0) for the reverse process.
    """
    ts = torch.linspace(0, T - 1, steps, device=device).round().long()
    ts[0]  = 0
    ts[-1] = T - 1
    return torch.unique_consecutive(ts).flip(0)   # high → low


@torch.no_grad()
def ddim_sample(
    diffusion    : GaussianDiffusion,
    denoise_fn   : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape        : tuple[int, ...],
    device       : torch.device,
    steps        : int = 50,
    eta          : float = 0.0,
    guidance_fn  : Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    DDIM reverse process. Generates samples from the prior p(x).

    When guidance_fn is provided (DPS-style posterior sampling), gradients
    are enabled inside the loop so the measurement gradient can flow through
    x̂_0. Otherwise the full function runs under no_grad.

    Parameters
    ----------
    diffusion : GaussianDiffusion
    denoise_fn : Callable (x_t, t_batch) → raw_pred
        The U-Net denoiser. Returns x̂_0 or ε̂ depending on pred_type.
    shape : tuple
        Output shape, e.g. (B, 1, L).
    device : torch.device
    steps : int
        Number of DDIM reverse steps. Default 50.
    eta : float
        DDIM stochasticity. 0.0 = deterministic DDIM. 1.0 = DDPM.
    guidance_fn : Callable (x0_hat, t_int) → grad, or None
        DPS guidance: returns ∇_{x̂_0} log p(y | x̂_0) (normalized).
        If None, samples from the prior unconditionally.

    Returns
    -------
    torch.Tensor, shape `shape`
        Denoised sample x̂_0.

    Notes
    -----
    *   With guidance_fn=None this is pure prior sampling (used in
        train_prior.py validation and sample_prior.py).
    *   With guidance_fn set this is DPS posterior sampling (used in
        experiments/part1/run.py).
    """
    diff = diffusion
    x    = torch.randn(shape, device=device)
    ts   = _make_ddim_timesteps(diff.T, steps, device)   # high → low

    for i, t_val in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else torch.tensor(0, device=device)
        t_batch = t_val.expand(shape[0])

        abar_t    = diff._extract(diff.alphas_bar,      t_batch, x.shape)
        abar_prev = diff._extract(diff.alphas_bar_prev,
                                  t_prev.expand(shape[0]), x.shape)

        if guidance_fn is not None:
            # Need gradients for DPS correction
            x = x.detach().requires_grad_(True)
            raw   = denoise_fn(x, t_batch)
            x0hat, eps = diff.to_x0_and_eps(x, t_batch, raw)
            grad  = guidance_fn(x0hat, int(t_val))
            x0hat = (x0hat + grad).detach()
            eps   = diff.eps_from_x0(x, t_batch, x0hat)
        else:
            raw        = denoise_fn(x, t_batch)
            x0hat, eps = diff.to_x0_and_eps(x, t_batch, raw)

        # DDIM update
        if eta > 0.0:
            sigma = eta * ((1 - abar_prev) / (1 - abar_t)
                           * (1 - abar_t / abar_prev.clamp(min=1e-8))).sqrt()
        else:
            sigma = torch.zeros_like(abar_t)

        dir_xt = (1 - abar_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
        noise  = sigma * torch.randn_like(x) if eta > 0 else 0.0
        x      = abar_prev.sqrt() * x0hat + dir_xt + noise

        if guidance_fn is not None:
            x = x.detach()

    return x
