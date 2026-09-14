"""
inference/vi.py
===============
Offline variational inference for likelihood parameters φ (Phase 1).

Per Algorithm 2: VI is run ONCE on D_pair (paired PPG+ECG from VitalDB)
to produce frozen λ* = (μ_{λ*}, log_σ²_{λ*}). After Phase 1, λ* is
frozen and used as the MH proposal kernel in Phase 2.

What this module provides
--------------------------
MeanFieldGaussian — the variational family q_λ(φ), unchanged from before.
    diagonal Gaussian with learnable (μ_λ, log_σ²_λ).

train_vi() — offline training loop.
    Maximises ELBO(λ) = E_{q_λ}[log L_φ(y|x,z)] - KL(q_λ || p(φ))
    on the paired dataset D_pair.
    Returns frozen λ* (a MeanFieldGaussian in eval mode).

save_lambda / load_lambda — persist λ* to disk between runs.

Notes
-----
*   Fisher-identity gradient: standard reparametrisation gradient is used
    (correct for offline VI with a Gaussian family). The reparametrisation
    trick gives low-variance gradients without the Fisher correction needed
    for score-function estimators.
*   The ELBO expectation is estimated by importance sampling over φ ~ q_λ.
*   λ* is FROZEN after train_vi() — never updated in Phase 2.
*   The likelihood does NOT depend on θ. Only (x, z, φ) enter L_φ.

Ownership: Fatemeh
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ======================================================================
# Mean-field Gaussian q_λ(φ)
# ======================================================================

class MeanFieldGaussian(nn.Module):
    """
    Mean-field Gaussian variational distribution q_λ(φ).

        q_λ(φ) = N(φ ; μ_λ, diag(σ²_λ))

    where σ²_λ = exp(log_sigma2_lambda) to enforce positivity.

    Parameters
    ----------
    phi_dim : int
        Dimensionality of φ.
        Part 1 minimal set {τ, a, b, σ²} → phi_dim = 4.
    init_mu : tensor or None
        Initial mean. Defaults to zeros.
    init_log_sigma2 : float
        Initial log variance (all entries). Default -2.0 → σ² ≈ 0.14.
    """

    def __init__(
        self,
        phi_dim         : int,
        init_mu         : Optional[torch.Tensor] = None,
        init_log_sigma2 : float = -2.0,
    ) -> None:
        super().__init__()
        self.phi_dim = phi_dim
        mu_init = init_mu if init_mu is not None else torch.zeros(phi_dim)
        self.mu_lambda         = nn.Parameter(mu_init.clone().float())
        self.log_sigma2_lambda = nn.Parameter(
            torch.full((phi_dim,), init_log_sigma2)
        )

    @property
    def sigma2(self) -> torch.Tensor:
        """Diagonal variance σ²_λ, shape (phi_dim,)."""
        return self.log_sigma2_lambda.exp()

    @property
    def sigma(self) -> torch.Tensor:
        """Diagonal std σ_λ, shape (phi_dim,)."""
        return self.sigma2.sqrt()

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Draw φ_s ~ q_λ via reparametrisation.
        φ_s = μ_λ + σ_λ ⊙ ε,  ε ~ N(0, I)

        Returns (n_samples, phi_dim). Differentiable w.r.t. λ.
        """
        eps = torch.randn(n_samples, self.phi_dim,
                          device=self.mu_lambda.device)
        return self.mu_lambda + self.sigma * eps

    def log_prob(self, phi: torch.Tensor) -> torch.Tensor:
        """
        log q_λ(φ) for a batch of φ values.
        phi : (S, phi_dim) → returns (S,)
        """
        diff    = phi - self.mu_lambda
        log_det = self.log_sigma2_lambda.sum()
        mah     = (diff.pow(2) / self.sigma2).sum(dim=-1)
        return -0.5 * (self.phi_dim * math.log(2 * math.pi) + log_det + mah)

    def kl_to_prior(
        self,
        prior_mu        : Optional[torch.Tensor] = None,
        prior_log_sigma2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Closed-form KL(q_λ || p(φ)) for Gaussian prior p(φ).
        Default prior: p(φ) = N(0, I).
        Returns scalar.
        """
        mu0  = prior_mu if prior_mu is not None \
               else torch.zeros_like(self.mu_lambda)
        ls20 = prior_log_sigma2 if prior_log_sigma2 is not None \
               else torch.zeros_like(self.log_sigma2_lambda)
        s20  = ls20.exp()
        kl   = 0.5 * (
            (self.sigma2 / s20)
            + (self.mu_lambda - mu0).pow(2) / s20
            - 1.0
            + ls20 - self.log_sigma2_lambda
        ).sum()
        return kl


# ======================================================================
# ELBO
# ======================================================================

def elbo(
    q_lambda        : MeanFieldGaussian,
    likelihood      : "GaussianLikelihood",
    y               : torch.Tensor,
    x               : torch.Tensor,
    z               : Optional[torch.Tensor],
    phi_to_params_fn: Callable[[torch.Tensor], dict],
    n_phi_samples   : int = 8,
    beta_kl         : float = 1.0,
    prior_mu        : Optional[torch.Tensor] = None,
    prior_log_sigma2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ELBO(λ) = E_{q_λ}[log L_φ(y|x,z)] - β · KL(q_λ(φ) || p(φ))

    The expectation is estimated by importance sampling:
        E_{q_λ}[log L_φ] ≈ log[(1/S) Σ_s L_{φ_s}(y|x,z)]  φ_s ~ q_λ

    Gradient flows through q_lambda via the reparametrisation trick.

    Parameters
    ----------
    q_lambda : MeanFieldGaussian
        Current variational distribution.
    likelihood : GaussianLikelihood
        Likelihood module. Parameters temporarily set per φ_s sample.
    y : (B, 1, L)   observed PPG
    x : (B, 1, L)   ECG (from paired dataset, DETACHED from any MCMC chain)
    z : (B, d_z) or None
    phi_to_params_fn : maps φ vector → dict of likelihood param updates
    n_phi_samples : int   IS samples S
    beta_kl : float       KL weight (1.0 = standard ELBO)
    prior_mu, prior_log_sigma2 : prior over φ (default N(0,I))

    Returns
    -------
    torch.Tensor, scalar   ELBO value (maximise this)
    """
    # Sample φ_s ~ q_λ via reparametrisation (differentiable)
    phi_samples = q_lambda.sample(n_phi_samples)      # (S, phi_dim)

    log_liks = []
    for s in range(n_phi_samples):
        phi_s  = phi_samples[s]
        params = phi_to_params_fn(phi_s)
        _set_likelihood_params(likelihood, params)
        ll_s   = likelihood.log_likelihood(y, x.detach(), z)  # (B,)
        log_liks.append(ll_s)

    log_liks_stack = torch.stack(log_liks, dim=0)             # (S, B)
    log_lik_mean   = (torch.logsumexp(log_liks_stack, dim=0)
                      - math.log(n_phi_samples)).mean()       # scalar

    kl = q_lambda.kl_to_prior(prior_mu, prior_log_sigma2)
    return log_lik_mean - beta_kl * kl


def _set_likelihood_params(likelihood, params: dict) -> None:
    """Set likelihood parameters in-place. Phi=[log_sigma2] only."""
    if "log_a"    in params:
        likelihood.log_a.data.fill_(float(params["log_a"]))
    if "b"        in params:
        likelihood.b.data.fill_(float(params["b"]))
    if "tau"      in params:
        likelihood.tau.data.fill_(float(params["tau"]))
    if "log_diag" in params:
        likelihood.noise_cov.log_diag.data.copy_(
            params["log_diag"].view_as(likelihood.noise_cov.log_diag))


# ======================================================================
# Offline VI training — Phase 1
# ======================================================================

def train_vi(
    likelihood       : "GaussianLikelihood",
    paired_loader    : DataLoader,
    phi_to_params_fn : Callable[[torch.Tensor], dict],
    phi_dim          : int,
    device           : torch.device,
    n_epochs         : int = 50,
    lr               : float = 1e-3,
    n_phi_samples    : int = 8,
    beta_kl          : float = 1.0,
    init_mu          : Optional[torch.Tensor] = None,
    prior_mu         : Optional[torch.Tensor] = None,
    prior_log_sigma2 : Optional[torch.Tensor] = None,
    verbose          : bool = True,
) -> MeanFieldGaussian:
    """
    Phase 1 VI: learn q_{λ*}(φ) from paired PPG+ECG data D_pair.

    Maximises ELBO(λ) over the paired dataset for n_epochs.
    Returns a FROZEN MeanFieldGaussian (in eval mode, no grad) — λ*.

    The paired dataset provides (y, x) pairs where:
        y = real PPG from VitalDB
        x = aligned real ECG from VitalDB
    VI fits q_λ(φ) so that the likelihood L_φ(y|x,z) is maximised
    in expectation over φ ~ q_λ.

    Parameters
    ----------
    likelihood : GaussianLikelihood
        Likelihood module (parameters updated during ELBO evaluation).
    paired_loader : DataLoader
        Yields batches of (ecg, ppg) tensors, both shape (B, 1, L).
        From data/vitaldb.py (to be built).
    phi_to_params_fn : Callable
        Maps φ vector (phi_dim,) → dict of likelihood param updates.
        Defined in experiments/part1/run.py where the φ layout is known.
    phi_dim : int
        Dimensionality of Φ. Part 1: 1 (log_σ² only).
        Z=(τ,a,b) is not part of VI — handled by MCMC.
    device : torch.device
    n_epochs : int
        Training epochs over D_pair. Default 50 (VI converges fast).
    lr : float
        Adam learning rate for λ.
    n_phi_samples : int
        IS samples per ELBO estimate.
    beta_kl : float
        KL weight. 1.0 = standard ELBO.
    init_mu : tensor or None
        Initial mean for q_λ. None → zeros.
    prior_mu, prior_log_sigma2 : prior over φ (default N(0,I)).
    verbose : bool

    Returns
    -------
    MeanFieldGaussian
        Frozen λ* in eval mode. Use as the MH proposal in Phase 2.

    Usage
    -----
    lambda_star = train_vi(likelihood, paired_loader, phi_to_params_fn,
                           phi_dim=1, device=device)  # Phi=[log_sigma2] only
    # λ* is now frozen — use in Phase 2:
    phi_prop = lambda_star.sample(1).squeeze(0)
    """
    q = MeanFieldGaussian(
        phi_dim=phi_dim, init_mu=init_mu
    ).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    if verbose:
        print(f"[VI] training q_λ(φ) on {len(paired_loader.dataset)} paired windows "
              f"for {n_epochs} epochs …")

    for ep in range(1, n_epochs + 1):
        q.train()
        total, n = 0.0, 0

        for batch in paired_loader:
            # paired_loader yields (ecg, ppg) or (ecg, ppg, meta)
            if len(batch) == 2:
                x_batch, y_batch = batch
            else:
                x_batch, y_batch = batch[0], batch[1]

            x_batch = x_batch.to(device)   # (B, 1, L) ECG
            y_batch = y_batch.to(device)   # (B, 1, L) PPG

            opt.zero_grad(set_to_none=True)
            loss = -elbo(
                q, likelihood, y_batch, x_batch, z=None,
                phi_to_params_fn=phi_to_params_fn,
                n_phi_samples=n_phi_samples,
                beta_kl=beta_kl,
                prior_mu=prior_mu,
                prior_log_sigma2=prior_log_sigma2,
            )
            loss.backward()
            opt.step()
            total += float(-loss.detach())
            n     += 1

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  VI epoch {ep:03d}/{n_epochs}  ELBO={total/max(1,n):.4f}"
                  f"  μ_λ={q.mu_lambda.data.tolist()}"
                  f"  σ_λ={q.sigma.data.tolist()}")

    # Freeze λ*
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)

    if verbose:
        print(f"[VI] done. λ* frozen.")
        print(f"  μ_λ* = {q.mu_lambda.data.tolist()}")
        print(f"  σ_λ* = {q.sigma.data.tolist()}")

    return q


# ======================================================================
# Save / load λ*
# ======================================================================

def save_lambda(q: MeanFieldGaussian, path: str | Path) -> None:
    """Save frozen λ* to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mu_lambda"         : q.mu_lambda.data,
        "log_sigma2_lambda" : q.log_sigma2_lambda.data,
        "phi_dim"           : q.phi_dim,
    }, str(path))
    print(f"[VI] λ* saved → {path}")


def load_lambda(path: str | Path, device: torch.device) -> MeanFieldGaussian:
    """Load frozen λ* from disk."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    q    = MeanFieldGaussian(phi_dim=ckpt["phi_dim"])
    q.mu_lambda.data.copy_(ckpt["mu_lambda"])
    q.log_sigma2_lambda.data.copy_(ckpt["log_sigma2_lambda"])
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)
    print(f"[VI] λ* loaded from {path}")
    return q.to(device)
