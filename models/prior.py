"""
models/prior.py
===============
ECG diffusion prior  p(X | θ)  and its bootstrap ensemble wrapper.

Two classes
-----------
ECGDiffusionPrior
    Single diffusion prior trained on PTB-XL. Wraps the U-Net denoiser
    and DiffusionConfig. Exposes score_x() for MALA and sample() for
    prior draws. Identical in structure to the plain prior from the
    CapnoBase work; here retrained on PTB-XL Lead II at 500 Hz.

BootstrapEnsemble
    Wraps B instances of ECGDiffusionPrior and provides an ensemble
    score and ensemble sample. Used by the MALA sampler in place of a
    single prior, reducing bias and variance in the score estimate.

    *** STUB — bootstrap interpretation (independent retraining vs.
    different seeds from the same checkpoint) is PENDING advisor
    confirmation. Interface is final; internals are placeholders. ***

Why bootstrap rather than a single prior
-----------------------------------------
Plugging in a single θ (the trained diffusion prior) gives a biased,
high-variance score estimate because θ is itself uncertain. A bootstrap
ensemble over θ approximates the marginal score

    ∇_X log p(X) = ∇_X log ∫ p(X|θ) p(θ) dθ
                 ≈ (1/B) Σ_b ∇_X log p(X | θ_b)

which has lower bias and variance than the plug-in estimate.

Ownership
---------
Fatemeh owns this module. Alex's code receives a BootstrapEnsemble
instance (or ECGDiffusionPrior for the single-prior baseline) through
the sampler interface — Alex never imports the internals of this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


# ======================================================================
# Diffusion schedule utilities (minimal, self-contained)
# ======================================================================

def _make_alphas_bar(T: int, beta_start: float = 1e-4,
                     beta_end: float = 0.02) -> torch.Tensor:
    """Cosine schedule ᾱ_t, shape (T,)."""
    import math
    steps = torch.arange(T + 1, dtype=torch.float32)
    s = 0.008
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return abar[1:]


def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(shape) - 1)))


# ======================================================================
# Single ECG diffusion prior
# ======================================================================

class ECGDiffusionPrior(nn.Module):
    """
    Unconditional ECG diffusion prior  p(X | θ).

    Wraps a trained 1-D denoising U-Net (models/unet1d.py) and the
    diffusion schedule. Provides:

    *   score_x()  — ∇_X log p(X | θ), used by MALA
    *   sample()   — unconditional ECG samples from the prior
    *   log_prob() — approximate log p(X | θ) via the ELBO lower bound
                     (used for Metropolis accept/reject in MALA)

    Parameters
    ----------
    denoiser : nn.Module
        Trained U-Net denoiser f_θ(x_t, t) → predicted x_0 or ε.
        Loaded from a PTB-XL checkpoint.
    T : int
        Number of diffusion timesteps.
    pred_type : str
        "x0" or "eps" — what the denoiser predicts.
    device : torch.device

    Notes
    -----
    *   This prior is UNCONDITIONAL — it was trained on ECG only,
        with no PPG conditioning. The likelihood L_φ provides the
        PPG measurement signal; the prior provides the ECG manifold.
    *   The denoiser does NOT take φ or Z as inputs.
    *   score_x() is the key method called by MALA (inference/mala.py).
    """

    def __init__(
        self,
        denoiser: nn.Module,
        T: int = 1000,
        pred_type: str = "x0",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.denoiser  = denoiser
        self.T         = T
        self.pred_type = pred_type
        _device = device or torch.device("cpu")

        abar = _make_alphas_bar(T).to(_device)
        self.register_buffer("alphas_bar", abar)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        denoiser_cls: type,
        denoiser_kwargs: dict,
        device: torch.device | None = None,
    ) -> "ECGDiffusionPrior":
        """
        Load a trained prior from a checkpoint file.

        Parameters
        ----------
        ckpt_path : str or Path
            Path to the .pt checkpoint saved by train_prior.py.
        denoiser_cls : type
            The U-Net class (e.g. UNet1D from models/unet1d.py).
        denoiser_kwargs : dict
            Constructor kwargs for denoiser_cls.
        device : torch.device or None

        Returns
        -------
        ECGDiffusionPrior, in eval mode.
        """
        _device = device or torch.device("cpu")
        ckpt = torch.load(str(ckpt_path), map_location=_device, weights_only=False)

        denoiser = denoiser_cls(**denoiser_kwargs).to(_device)
        denoiser.load_state_dict(ckpt["model"])
        denoiser.eval()

        return cls(
            denoiser=denoiser,
            T=ckpt.get("T", 1000),
            pred_type=ckpt.get("pred_type", "x0"),
            device=_device,
        ).to(_device)

    # ------------------------------------------------------------------

    def _x0_hat(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Tweedie posterior mean x̂_0 = E[X_0 | X_t], shape (B, 1, L).
        Derived from denoiser output regardless of pred_type.
        """
        raw = self.denoiser(x_t, t)
        abar = _extract(self.alphas_bar, t, x_t.shape)
        if self.pred_type == "x0":
            return raw
        elif self.pred_type == "eps":
            return (x_t - (1 - abar).sqrt() * raw) / abar.sqrt()
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

    def score_x(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Approximate score ∇_X log p(X | θ) at noise level t.

        Uses the denoiser-based score estimator:
            s_θ(x_t, t) = (x̂_0 - x_t) / (1 - ᾱ_t)

        This is the standard diffusion score function used in
        DPS-style posterior sampling and MALA with a diffusion prior.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, L)
            Current state (noisy ECG at level t, or clean X at t=0).
        t : torch.Tensor, shape (B,), dtype long
            Diffusion timestep index for each batch element.

        Returns
        -------
        torch.Tensor, shape (B, 1, L)
            Score estimate ∇_X log p(X | θ).

        Notes
        -----
        At t=0 (MCMC over clean X), pass t = torch.zeros(B, dtype=long).
        The score is still meaningful as the gradient of the prior log-density
        approximated by the denoiser at the lowest noise level.
        """
        with torch.no_grad():
            x0_hat = self._x0_hat(x, t)
        abar = _extract(self.alphas_bar, t, x.shape)
        return (x0_hat - x) / (1.0 - abar).clamp(min=1e-6)

    def log_prob_approx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Approximate log p(X | θ) via the denoising ELBO at level t.

        Used for the Metropolis acceptance step in MALA.
        This is an approximation — the true log p(X | θ) is intractable.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, L)
        t : torch.Tensor, shape (B,), dtype long

        Returns
        -------
        torch.Tensor, shape (B,)
            Per-sample approximate log prior.
        """
        x0_hat = self._x0_hat(x, t)
        # Gaussian approximation: -||x - x0_hat||² / 2(1 - ᾱ_t)
        abar = _extract(self.alphas_bar, t, x.shape)
        var  = (1.0 - abar).clamp(min=1e-6)
        return -0.5 * ((x - x0_hat).pow(2) / var).view(x.shape[0], -1).sum(dim=1)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        ddim_steps: int = 50,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Draw unconditional ECG samples from the prior via DDIM.

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the output tensor, e.g. (B, 1, L).
        ddim_steps : int
            Number of DDIM reverse steps.
        device : torch.device or None

        Returns
        -------
        torch.Tensor, shape (*shape)
            Clean ECG samples drawn from p(X | θ).
        """
        _device = device or next(self.parameters()).device
        # Import here to avoid circular; ddim_sample lives in diffusion_core.py
        from models.diffusion_core import ddim_sample, GaussianDiffusion

        diff = GaussianDiffusion(
            T=self.T,
            schedule="cosine",
            pred_type=self.pred_type,
        ).to(_device)
        denoise_fn = lambda x_t, t_batch: self.denoiser(x_t, t_batch)
        return ddim_sample(diff, denoise_fn, shape, _device, steps=ddim_steps)


# ======================================================================
# Bootstrap ensemble wrapper  — INTERFACE FINAL, INTERNALS STUB
# ======================================================================

class BootstrapEnsemble(nn.Module):
    """
    Ensemble of B ECGDiffusionPrior instances for reduced-bias score
    and log-probability estimation.

    The ensemble approximates the marginal score:

        ∇_X log p(X) ≈ (1/B) Σ_b ∇_X log p(X | θ_b)

    and is used by the MALA sampler in place of a single prior.

    *** INTERNALS ARE A STUB ***
    The interpretation of "bootstrap" — whether θ_b are drawn by
    retraining on bootstrap resamples of PTB-XL, or by drawing
    different noise seeds from a single checkpoint — is PENDING
    advisor confirmation. The interface below is final.

    Parameters
    ----------
    priors : list[ECGDiffusionPrior]
        List of B trained prior instances.

    Usage (by MALA sampler)
    -----------------------
    ensemble = BootstrapEnsemble(priors)
    score = ensemble.score_x(x, t)       # averaged score
    lp    = ensemble.log_prob_approx(x, t)  # averaged log-prob
    """

    def __init__(self, priors: list[ECGDiffusionPrior]) -> None:
        super().__init__()
        self.priors = nn.ModuleList(priors)
        self.B = len(priors)
        if self.B == 0:
            raise ValueError("BootstrapEnsemble requires at least one prior.")

    @classmethod
    def from_checkpoint_list(
        cls,
        ckpt_paths   : list[str | Path],
        denoiser_cls : type,
        denoiser_kwargs : dict,
        device       : torch.device | None = None,
    ) -> "BootstrapEnsemble":
        """
        Load K trained priors from K checkpoint files.

        Each checkpoint was produced by train_bootstrap.py — one full
        training run on a bootstrap resample of PTB-XL patients.

        Parameters
        ----------
        ckpt_paths : list of str or Path
            Paths to the K checkpoint files, e.g.
            ["checkpoints/prior_k00.pt", ..., "checkpoints/prior_k09.pt"]
        denoiser_cls : type
            The U-Net class (UNet1D from models/unet1d.py).
        denoiser_kwargs : dict
            Constructor kwargs for denoiser_cls, e.g.
            {"base_ch": 64, "time_dim": 128, "n_res": 2}
        device : torch.device or None

        Returns
        -------
        BootstrapEnsemble with B = len(ckpt_paths) members, all in eval mode.
        """
        priors = []
        for path in ckpt_paths:
            prior = ECGDiffusionPrior.from_checkpoint(
                path, denoiser_cls, denoiser_kwargs, device
            )
            priors.append(prior)
        print(f"[BootstrapEnsemble] loaded {len(priors)} priors from checkpoints")
        return cls(priors)

    def score_x(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Ensemble-averaged score ∇_X log p(X):

            (1/B) Σ_b ∇_X log p(X | θ_b)

        Parameters
        ----------
        x : torch.Tensor, shape (B_batch, 1, L)
        t : torch.Tensor, shape (B_batch,), dtype long

        Returns
        -------
        torch.Tensor, shape (B_batch, 1, L)
        """
        # STUB: averaging is correct; internals of each prior.score_x
        # are implemented. Only from_single_checkpoint is stubbed.
        scores = torch.stack([p.score_x(x, t) for p in self.priors], dim=0)
        return scores.mean(dim=0)

    def log_prob_approx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Ensemble-averaged approximate log p(X).

        Parameters
        ----------
        x : torch.Tensor, shape (B_batch, 1, L)
        t : torch.Tensor, shape (B_batch,), dtype long

        Returns
        -------
        torch.Tensor, shape (B_batch,)
        """
        lps = torch.stack([p.log_prob_approx(x, t) for p in self.priors], dim=0)
        return lps.mean(dim=0)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        ddim_steps: int = 50,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Draw samples by randomly selecting one prior θ_b and sampling from it.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape, e.g. (N, 1, L).
        ddim_steps : int
        device : torch.device or None

        Returns
        -------
        torch.Tensor, shape (*shape)
        """
        b = torch.randint(0, self.B, (1,)).item()
        return self.priors[b].sample(shape, ddim_steps=ddim_steps, device=device)
