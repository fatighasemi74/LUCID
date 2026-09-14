"""
inference/sampler.py
====================
Phase 2 posterior inference: outer loop over k bootstrap atoms.

Implements Algorithm 2 (Phase 2) exactly:

    For k = 0..K-1 (outer loop, fixed θ_k):
        Initialize φ⁽⁰⁾ ~ q_{λ*}, z⁽⁰⁾ ~ ρ, x⁽⁰⁾ ~ prior
        For r = 0..N-1 (inner MCMC loop):
            φ ← MH with proposal q_{λ*}                [phi_mh_step]
            z ← MALA targeting L_φ(y|x,z)·ρ(z)        [z_mala_step]
            x ← annealed guided diffusion MCMC          [x_diffusion_step]
            if r >= B: collect (x, z, φ)
        Ẑ_k(y) = (1/|samples|) Σ L_φ(y|x,z)          [evidence estimate]

    ω̂_k = π_k · Ẑ_k / Σ_j π_j · Ẑ_j               [weight update]
    P̂_{Θ|y} = Σ_k ω̂_k δ_{θ_k}

Output per call to run_inference():
    samples  : list of K SampleSet objects (x, z, phi per k)
    weights  : (K,) tensor of ω̂_k values
    evidence : (K,) tensor of Ẑ_k values

These feed directly into evaluation/uncertainty.py for the four
variance terms U_X, U_Z, U_Φ, U_Θ.

Ownership: shared interface (Fatemeh's prior/likelihood + Alex's Part 2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from models.prior import ECGDiffusionPrior, BootstrapEnsemble
from models.likelihood import GaussianLikelihood
from models.diffusion_core import GaussianDiffusion
from inference.mala import (
    MCMCState, MALAConfig,
    phi_mh_step, tau_xcorr_mh_step, ab_mala_step,
    x_diffusion_step, _set_likelihood_params,
    set_state_params, z_to_params, phi_to_params,
    make_z_prior,
)
from inference.vi import MeanFieldGaussian, load_lambda


# ======================================================================
# Sample container
# ======================================================================

@dataclass
class SampleSet:
    """
    Posterior samples collected from one bootstrap atom θ_k.

    Fields
    ------
    k       : int           bootstrap index
    x_samples   : (N_s, 1, L)  ECG posterior samples (after burn-in)
    z_samples   : (N_s, d_z)   Z posterior samples
    phi_samples : (N_s, phi_dim) φ posterior samples
    evidence    : float         Ẑ_k(y) — estimated model evidence
    accept_rate_phi : float     MH acceptance rate for φ
    accept_rate_z   : float     MALA acceptance rate for z
    """
    k               : int
    x_samples       : torch.Tensor
    z_samples       : torch.Tensor
    phi_samples     : torch.Tensor
    evidence        : float
    accept_rate_phi : float
    accept_rate_z   : float

    @property
    def n_samples(self) -> int:
        return self.x_samples.shape[0]


# ======================================================================
# Evidence estimation via Sequential Monte Carlo
# ======================================================================

def estimate_evidence_smc(
    y                : torch.Tensor,
    prior_k          : "ECGDiffusionPrior",
    q_lambda_star    : "MeanFieldGaussian",
    likelihood       : GaussianLikelihood,
    diffusion        : GaussianDiffusion,
    phi_to_params_fn : Callable,
    cfg              : MALAConfig,
    log_z_prior      : Callable,
    device           : torch.device,
    n_particles      : int = 64,
    n_temps          : int = 8,
    fix_log_sigma2   : float | None = None,
) -> float:
    """
    SMC estimate of Ẑ_k(y) = ∫ L_φ(y|x,z) μ_{θ_k}(dx) ρ(dz) q_{λ*}(φ) dφ

    Uses a geometric tempering ladder from the prior to the posterior:
        p_β(x,z,φ) ∝ L_φ(y|x,z)^β · μ_{θ_k}(x) · ρ(z) · q_{λ*}(φ)
    with β ∈ {0, 1/T, 2/T, ..., 1}.

    Algorithm (Annealed Importance Sampling / SMC):
        1. Draw N particles from the prior at β=0:
               x^i ~ μ_{θ_k},  z^i ~ ρ,  φ^i ~ q_{λ*}
        2. For each temperature step β_{t-1} → β_t:
               w^i = L_φ^i(y|x^i,z^i)^{β_t - β_{t-1}}
               normalize weights → ω^i
               resample if ESS < N/2
               propagate particles (MCMC kernel, a few steps)
        3. Ẑ_k = product of normalizing constants across all temperature steps

    This gives an UNBIASED estimate of Ẑ_k, unlike plain MC which
    averages L over posterior samples (circular and biased).

    Parameters
    ----------
    y : (1, 1, L)
    prior_k : ECGDiffusionPrior — θ_k prior for x sampling
    q_lambda_star : MeanFieldGaussian — frozen φ proposal
    likelihood : GaussianLikelihood
    diffusion : GaussianDiffusion
    phi_to_params_fn : Callable
    cfg : MALAConfig
    log_z_prior : Callable — log ρ(z)
    device : torch.device
    n_particles : int — number of SMC particles (default 64)
    n_temps : int — number of temperature levels (default 8)

    Returns
    -------
    float : log Ẑ_k (log evidence, for numerical stability)
    """
    L      = y.shape[-1]
    betas  = torch.linspace(0.0, 1.0, n_temps + 1)[1:]   # (n_temps,)

    # ── Step 1: Initialize particles from prior at β=0 ────────────────
    with torch.no_grad():
        # x particles from prior
        x_p = prior_k.sample((n_particles, 1, L),
                              ddim_steps=20, device=device)   # (P,1,L)
        # z particles from ρ — Z=[tau,a,b], sample from prior
        tau_p = torch.randn(n_particles, 1, device=device) * 0.05 + 0.20
        a_p   = torch.randn(n_particles, 1, device=device) * 0.20 + 0.80
        b_p   = torch.randn(n_particles, 1, device=device) * 0.10 + 0.05
        z_p   = torch.cat([tau_p, a_p, b_p], dim=1)   # (P, 3)
        # φ=[log_sigma2] — fixed, no sampling needed
        # Use fixed log_sigma2 if provided, else default
        _log_s2 = fix_log_sigma2 if fix_log_sigma2 is not None else -3.0
        phi_p = torch.full((n_particles, 1), _log_s2, device=device)

    log_Z_hat = 0.0   # accumulated log normalizing constant
    log_w     = torch.zeros(n_particles, device=device)

    # ── Step 2: Temperature ladder ────────────────────────────────────
    beta_prev = 0.0
    for t_idx in range(n_temps):
        beta_t = float(betas[t_idx])
        delta_beta = beta_t - beta_prev

        # Compute log L^{delta_beta} for each particle
        log_liks = []
        with torch.no_grad():
            for i in range(n_particles):
                # Set Z=[tau,a,b] and Phi=[log_sigma2]
                _set_likelihood_params_direct(likelihood,
                    z_to_params(z_p[i].unsqueeze(0)))
                _set_likelihood_params_direct(likelihood,
                    phi_to_params_fn(phi_p[i]))
                ll = float(likelihood.log_likelihood(
                    y, x_p[i].unsqueeze(0), z_p[i].unsqueeze(0)
                ).sum())
                log_liks.append(ll)

        log_liks_t  = torch.tensor(log_liks, device=device)
        log_w       = log_w + delta_beta * log_liks_t

        # Accumulate log normalizing constant
        log_Z_t  = torch.logsumexp(log_w, dim=0) - math.log(n_particles)
        log_Z_hat += float(log_Z_t)

        # Normalize weights
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        w_norm = log_w.exp()

        # Resample if ESS < N/2
        ess = float(1.0 / (w_norm ** 2).sum())
        if ess < n_particles / 2 and t_idx < n_temps - 1:
            indices = torch.multinomial(w_norm, n_particles, replacement=True)
            x_p     = x_p[indices]
            z_p     = z_p[indices]
            phi_p   = phi_p[indices]
            log_w   = torch.zeros(n_particles, device=device)

        # MCMC refresh (one MH step for φ to propagate diversity)
        if t_idx < n_temps - 1:
            for i in range(n_particles):
                # Propose new Z=[tau,a,b] via small perturbation
                z_prop = z_p[i] + torch.randn(3, device=device) *                          torch.tensor([0.02, 0.05, 0.02], device=device)
                _set_likelihood_params_direct(likelihood,
                    z_to_params(z_prop.unsqueeze(0)))
                _set_likelihood_params_direct(likelihood,
                    phi_to_params_fn(phi_p[i]))
                with torch.no_grad():
                    ll_prop = float(likelihood.log_likelihood(
                        y, x_p[i].unsqueeze(0), z_prop.unsqueeze(0)
                    ).sum())
                ll_curr = log_liks[i]
                if math.log(torch.rand(1).item() + 1e-10) < beta_t * (ll_prop - ll_curr):
                    z_p[i] = z_prop
                    log_liks[i] = ll_prop

        beta_prev = beta_t

    return log_Z_hat


def _set_likelihood_params_direct(likelihood, params: dict) -> None:
    if "log_a"    in params: likelihood.log_a.data.fill_(params["log_a"].item())
    if "b"        in params: likelihood.b.data.fill_(params["b"].item())
    if "tau"      in params: likelihood.tau.data.fill_(params["tau"].item())
    if "log_diag" in params: likelihood.noise_cov.log_diag.data.copy_(
        params["log_diag"])


# ======================================================================
# Weight update — direct (no MCMC)
# ======================================================================

def compute_weights(
    log_evidences : list[float],
    pi            : Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ω̂_k(y) = π_k · Ẑ_k(y) / Σ_j π_j · Ẑ_j(y)
    Computed entirely in log space to avoid underflow.

    Parameters
    ----------
    log_evidences : list[float]
        log Ẑ_k values from estimate_evidence_smc().
    pi : (K,) tensor or None
        Prior weights π_k. None → uniform 1/K.

    Returns
    -------
    weights : (K,) tensor, sums to 1.
    """
    K      = len(log_evidences)
    pi     = pi if pi is not None else torch.ones(K) / K
    log_pi = pi.double().log()
    log_Z  = torch.tensor(log_evidences, dtype=torch.float64)
    log_w  = log_pi + log_Z
    log_w  = log_w - torch.logsumexp(log_w, dim=0)
    return log_w.exp().float()


# ======================================================================
# Z prior (Part 1: Gaussian prior on delay τ)
# ======================================================================

def make_gaussian_z_prior(
    mu    : float = 0.2,
    sigma : float = 0.1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Gaussian prior on Z = τ (delay in seconds).
    ρ(z) = N(z; mu, sigma²)

    Returns a callable log_z_prior(z) → scalar tensor.
    """
    def log_z_prior(z: torch.Tensor) -> torch.Tensor:
        return (-0.5 * ((z - mu) / sigma).pow(2)
                - math.log(sigma) - 0.5 * math.log(2 * math.pi))
    return log_z_prior


# ======================================================================
# Main inference function
# ======================================================================

def run_inference(
    y                : torch.Tensor,
    ensemble         : BootstrapEnsemble,
    q_lambda_star    : MeanFieldGaussian,
    likelihood       : GaussianLikelihood,
    diffusion        : GaussianDiffusion,
    phi_to_params_fn : Callable,
    phi_dim          : int,
    device           : torch.device,
    cfg              : MALAConfig,
    log_z_prior      : Optional[Callable] = None,
    verbose          : bool = True,
    fix_log_sigma2   : float | None = None,
) -> tuple[list[SampleSet], torch.Tensor, torch.Tensor]:
    """
    Full Phase 2 inference for one PPG observation y.

    Parameters
    ----------
    y : (1, 1, L)
        Observed PPG window (normalized).
    ensemble : BootstrapEnsemble
        K trained ECG diffusion priors {θ_k}.
    q_lambda_star : MeanFieldGaussian
        Frozen variational distribution λ* from Phase 1 VI.
    likelihood : GaussianLikelihood
        Likelihood module L_φ(y|x,z).
    diffusion : GaussianDiffusion
        Shared diffusion schedule.
    phi_to_params_fn : Callable
        Maps φ vector → dict of likelihood parameter updates.
    phi_dim : int
        Dimensionality of φ.
    device : torch.device
    cfg : MALAConfig
        MALA hyperparameters (step sizes, annealing schedule, n_inner, burn_in).
    log_z_prior : Callable or None
        Log prior ρ(z). None → Gaussian with default params.
    verbose : bool

    Returns
    -------
    sample_sets : list[SampleSet]
        One SampleSet per k, containing posterior samples.
    weights : (K,) tensor
        ω̂_k posterior weights over θ.
    evidences : (K,) tensor
        Ẑ_k model evidence values.
    """
    K = ensemble.B
    if log_z_prior is None:
        log_z_prior = make_z_prior()

    sample_sets = []
    evidences   = []

    for k in range(K):
        if verbose:
            print(f"[sampler] k={k:02d}/{K-1}  inner loop N={cfg.n_inner} "
                  f"burn_in={cfg.burn_in} …")

        prior_k = ensemble.priors[k]

        # ── Initialize state ──────────────────────────────────────────
        # x⁽⁰⁾ ~ prior (unconditional ECG sample from θ_k)
        with torch.no_grad():
            x0 = prior_k.sample((1, 1, y.shape[-1]),
                                 ddim_steps=20, device=device)
        # z⁽⁰⁾ = [tau, a, b] initialized at typical values
        z0 = torch.tensor([[0.20, 0.80, 0.05]],
                           dtype=torch.float32, device=device)
        # Φ⁽⁰⁾ = [log_sigma2] — fixed at true value in Part 1
        if fix_log_sigma2 is not None:
            phi0 = torch.tensor([fix_log_sigma2],
                                  dtype=torch.float32, device=device)
        else:
            phi0 = torch.tensor([-3.0], dtype=torch.float32, device=device)

        # Initial log-likelihood
        _set_likelihood_params(likelihood, z_to_params(z0))
        _set_likelihood_params(likelihood, phi_to_params(phi0))
        with torch.no_grad():
            log_lik0 = float(likelihood.log_likelihood(y, x0, z0).sum())

        state = MCMCState(
            x=x0, z=z0, phi=phi0,
            log_lik=log_lik0, step=0,
            n_acc_phi=0, n_acc_z=0, n_acc_tau=0,
        )

        # ── Prior score function for this k ───────────────────────────
        # Returns RAW denoiser output — x_diffusion_step handles
        # the to_x0_and_eps conversion and gradient computation internally.
        def prior_score_fn(x_t, t_batch, prior=prior_k):
            return prior.denoiser(x_t, t_batch)

        # ── Inner MCMC loop ───────────────────────────────────────────
        x_collected   = []
        z_collected   = []
        phi_collected = []

        for r in range(cfg.n_inner):
            # 1. Φ = log_sigma2 — fixed in Part 1
            state, _ = phi_mh_step(
                state, y, likelihood,
                fix_log_sigma2=fix_log_sigma2,
                device=device,
            )

            # 2. Z = [tau, a, b]:
            #    tau gets cross-correlation MH
            #    (a, b) get MALA
            state, _ = tau_xcorr_mh_step(
                state, y, likelihood, cfg, log_z_prior, device
            )
            state, _ = ab_mala_step(
                state, y, likelihood, log_z_prior, cfg, device
            )

            # 3. X — annealed guided DDIM
            state = x_diffusion_step(
                state, y, prior_score_fn, likelihood,
                diffusion, cfg, device
            )
            state = MCMCState(
                x=state.x, z=state.z, phi=state.phi,
                log_lik=state.log_lik,
                step=r + 1,
                n_acc_phi=state.n_acc_phi,
                n_acc_z=state.n_acc_z,
                n_acc_tau=state.n_acc_tau,
            )

            # Collect after burn-in
            if r >= cfg.burn_in:
                x_collected.append(state.x.squeeze(0).cpu())
                z_collected.append(state.z.squeeze(0).cpu())
                phi_collected.append(state.phi.cpu())

        if verbose:
            n_s = len(x_collected)
            print(f"  k={k:02d}: {n_s} samples  "
                  f"acc_phi={state.accept_rate_phi:.2f}  "
                  f"acc_z={state.accept_rate_z:.2f}")

        # Always print sample count for debugging
        print(f"  [sampler] k={k:02d}: collected {len(x_collected)} samples "
              f"(n_inner={cfg.n_inner}, burn_in={cfg.burn_in})")

        # ── Build SampleSet ───────────────────────────────────────────
        ss = SampleSet(
            k               = k,
            x_samples       = torch.stack(x_collected),      # (N_s,1,L)
            z_samples       = torch.stack(z_collected),      # (N_s,d_z)
            phi_samples     = torch.stack(phi_collected),    # (N_s,phi_dim)
            evidence        = 0.0,   # filled below
            accept_rate_phi = state.accept_rate_phi,
            accept_rate_z   = state.accept_rate_z,
        )

        # ── Evidence estimate Ẑ_k via SMC ────────────────────────────
        log_ev = estimate_evidence_smc(
            y=y,
            prior_k=prior_k,
            q_lambda_star=q_lambda_star,
            likelihood=likelihood,
            diffusion=diffusion,
            phi_to_params_fn=phi_to_params_fn,
            cfg=cfg,
            log_z_prior=log_z_prior,
            device=device,
            fix_log_sigma2=fix_log_sigma2,
        )
        # Store log-evidence (avoid exp underflow)
        ss = SampleSet(
            k=ss.k, x_samples=ss.x_samples,
            z_samples=ss.z_samples, phi_samples=ss.phi_samples,
            evidence=log_ev,   # store log-evidence
            accept_rate_phi=ss.accept_rate_phi,
            accept_rate_z=ss.accept_rate_z,
        )

        sample_sets.append(ss)
        evidences.append(log_ev)   # log-evidence list

        if verbose:
            print(f"  k={k:02d}: log Ẑ_k={log_ev:.2f}")

    # ── θ posterior weights ───────────────────────────────────────────
    weights = compute_weights(evidences)

    if verbose:
        print(f"\n[sampler] θ posterior weights ω̂_k:")
        for k, (w, ev) in enumerate(zip(weights.tolist(), evidences)):
            print(f"  k={k:02d}: ω̂_k={w:.4f}  Ẑ_k={ev:.4e}")

    return sample_sets, weights, torch.tensor(evidences)
