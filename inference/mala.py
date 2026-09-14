"""
inference/mala.py
=================
MCMC update kernels for Pulse2Posterior Part 1.

Restructured per professor's Sep 8 direction:

    Z = (τ, a, b)       record-specific per-observation variables  dim=3
    Φ = (log_σ²)        global forward-model noise parameter        dim=1

τ has its own structured proposal (cross-correlation initialization +
local Gaussian MH) rather than being bundled with q_λ*.

a, b are updated via MALA within the Z block.

Φ = log_σ² is fixed at σ²_true for Part 1 controlled experiments.

Update order per inner step:
    1. τ   — cross-correlation MH
    2. a,b — MALA
    3. X   — annealed guided DDIM
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# State containers
# ======================================================================

@dataclass
class MCMCState:
    """
    Current state of the inner MCMC chain for one (k, window) pair.

    Fields
    ------
    x     : (1, 1, L)   current ECG sample
    z     : (1, 3)      current Z = [τ, a, b]  (record-specific)
    phi   : (1,)        current Φ = [log_σ²]   (global noise)
    log_lik : float     log L_φ(y | x, z) at current state
    step  : int         inner loop iteration index r
    n_acc_phi : int     accepted φ proposals so far
    n_acc_z   : int     accepted z proposals so far
    n_acc_tau : int     accepted τ proposals so far
    """
    x         : torch.Tensor
    z         : torch.Tensor          # (1, 3): [tau, a, b]
    phi       : torch.Tensor          # (1,):   [log_sigma2]
    log_lik   : float
    step      : int = 0
    n_acc_phi : int = 0
    n_acc_z   : int = 0
    n_acc_tau : int = 0

    @property
    def accept_rate_phi(self) -> float:
        return self.n_acc_phi / max(1, self.step)

    @property
    def accept_rate_z(self) -> float:
        return self.n_acc_z / max(1, self.step)

    @property
    def accept_rate_tau(self) -> float:
        return self.n_acc_tau / max(1, self.step)

    # Convenience accessors
    @property
    def tau(self) -> float:
        return float(self.z[0, 0])

    @property
    def a(self) -> float:
        return float(self.z[0, 1])

    @property
    def b(self) -> float:
        return float(self.z[0, 2])

    @property
    def log_sigma2(self) -> float:
        return float(self.phi[0])


@dataclass
class MALAConfig:
    """
    Hyperparameters for the three update mechanisms.

    Parameters
    ----------
    step_size_z : float
        Langevin step size ε for Z MALA updates (a, b components).
    step_size_tau : float
        Local proposal std for τ cross-correlation MH (in seconds).
    t_anneal : list[int]
        Annealing sequence of diffusion timesteps for X updates.
    gamma_t : dict[int, float]
        Guidance strength γ_t per diffusion timestep.
    n_inner : int
        Number of inner MCMC steps N per outer k iteration.
    burn_in : int
        Burn-in steps B.
    tau_min, tau_max : float
        Physiological range for τ in seconds.
    """
    step_size_z     : float      = 1e-3
    step_size_tau   : float      = 0.02      # local std for tau proposal (seconds)
    t_anneal        : list[int]  = field(
        default_factory=lambda: list(range(500, 0, -10)) + [1]
    )
    gamma_t         : dict       = field(default_factory=dict)
    n_inner         : int        = 100
    burn_in         : int        = 20
    target_accept_z : float      = 0.574
    adapt_step_size : bool       = True
    tau_min         : float      = 0.05      # seconds
    tau_max         : float      = 0.40      # seconds

    def gamma(self, t: int) -> float:
        return self.gamma_t.get(t, 1.0)


# ======================================================================
# Parameter helpers
# ======================================================================

def z_to_params(z: torch.Tensor) -> dict:
    """
    Map Z = [τ, a, b] → likelihood parameter dict.
    z shape: (1, 3) or (3,)
    """
    zv = z.squeeze()
    return {
        "tau"  : zv[0],
        "log_a": zv[1].abs().log(),
        "b"    : zv[2],
    }


def phi_to_params(phi: torch.Tensor) -> dict:
    """
    Map Φ = [log_σ²] → likelihood parameter dict.
    phi shape: (1,) or scalar
    """
    return {
        "log_diag": phi.view(1),
    }


def _set_likelihood_params(likelihood, params: dict) -> None:
    """Set likelihood parameters in-place from a parameter dict."""
    if "log_a"    in params:
        likelihood.log_a.data.fill_(float(params["log_a"]))
    if "b"        in params:
        likelihood.b.data.fill_(float(params["b"]))
    if "tau"      in params:
        likelihood.tau.data.fill_(float(params["tau"]))
    if "log_diag" in params:
        likelihood.noise_cov.log_diag.data.copy_(
            params["log_diag"].view_as(likelihood.noise_cov.log_diag)
        )


def set_state_params(likelihood, state: MCMCState) -> None:
    """Set all likelihood params from current state (Z and Φ)."""
    _set_likelihood_params(likelihood, z_to_params(state.z))
    _set_likelihood_params(likelihood, phi_to_params(state.phi))


# ======================================================================
# Prior for Z = [τ, a, b]
# ======================================================================

def make_z_prior(
    tau_mu: float = 0.20, tau_sigma: float = 0.05,
    a_mu:   float = 0.80, a_sigma:   float = 0.20,
    b_mu:   float = 0.05, b_sigma:   float = 0.10,
) -> Callable:
    """
    Independent Gaussian prior on Z = [τ, a, b].
    Returns a callable: z (1,3) → scalar log-prior tensor.
    """
    def log_prior(z: torch.Tensor) -> torch.Tensor:
        zv = z.squeeze()
        lp  = -0.5 * ((zv[0] - tau_mu) / tau_sigma).pow(2)
        lp += -0.5 * ((zv[1] - a_mu)   / a_sigma  ).pow(2)
        lp += -0.5 * ((zv[2] - b_mu)   / b_sigma  ).pow(2)
        return lp
    return log_prior


# ======================================================================
# 1. τ update — cross-correlation MH
# ======================================================================

def _apply_lowpass_kernel(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply 1-D lowpass convolution (same padding)."""
    K = kernel.shape[0]
    return F.conv1d(x, kernel.view(1,1,K), padding=K//2)


def _xcorr_tau_estimate(
    y       : torch.Tensor,   # (1, 1, L) observed PPG
    x       : torch.Tensor,   # (1, 1, L) current ECG sample
    a       : float,
    b       : float,
    kernel  : torch.Tensor,   # lowpass kernel from likelihood
    fs      : float,
    tau_min : float,
    tau_max : float,
) -> float:
    """
    Estimate τ by cross-correlating y with H_{a,b,τ=0}(x).

    Computes H without delay, then finds the lag that maximizes
    cross-correlation with y. Constrains result to [tau_min, tau_max].

    Returns τ_xcorr in seconds.
    """
    with torch.no_grad():
        # Apply lowpass and amplitude scaling — no delay
        h = _apply_lowpass_kernel(x, kernel)   # (1, 1, L)
        h = a * h + b                           # (1, 1, L)

        # Cross-correlation: how much does y align with h at each lag?
        # Using conv1d: xcorr[lag] = sum_t y[t] * h[t - lag]
        L = x.shape[-1]
        h_norm = h - h.mean()
        y_norm = y - y.mean()

        # Pad h for full cross-correlation
        xcorr = F.conv1d(
            y_norm.view(1, 1, L),
            h_norm.view(1, 1, L),
            padding=L - 1,
        ).squeeze()                              # (2L-1,)

        # Lag 0 is at index L-1
        center = L - 1
        lag = int(xcorr.argmax().item()) - center  # in samples, positive = y lags h

        tau_est = lag / fs
        # Clamp to physiological range
        tau_est = float(max(tau_min, min(tau_max, tau_est)))

    return tau_est


def tau_xcorr_mh_step(
    state   : MCMCState,
    y       : torch.Tensor,
    likelihood,
    cfg     : MALAConfig,
    log_z_prior : Callable,
    device  : torch.device,
) -> tuple[MCMCState, bool]:
    """
    MH update for τ using cross-correlation initialization + local proposal.

    Proposal:
        τ_xcorr = argmax_lag xcorr(y, H_{a,b,τ=0}(x))  [data-informed center]
        τ'      = τ_xcorr + δ · ξ,   ξ ~ N(0,1),  δ = cfg.step_size_tau

    This is an asymmetric proposal (depends on current x, a, b) so we
    need MH correction. The proposal density:
        q(τ'|τ) = N(τ'; τ_xcorr, δ²)   — same center regardless of τ_current
    so q(τ'|τ) / q(τ|τ') = N(τ';τ_xcorr,δ²) / N(τ;τ_xcorr,δ²)

    Acceptance:
        log α = [log L(y|x,z') + log ρ(z') + log q(τ|τ')]
              - [log L(y|x,z)  + log ρ(z)  + log q(τ'|τ)]
    """
    # Current values
    tau_curr = state.tau
    a_curr   = state.a
    b_curr   = state.b

    # Cross-correlation estimate for proposal center
    tau_xcorr = _xcorr_tau_estimate(
        y, state.x, a_curr, b_curr,
        likelihood.kernel.to(device),
        likelihood.fs,
        cfg.tau_min, cfg.tau_max,
    )

    # Local Gaussian proposal centered at τ_xcorr
    delta = cfg.step_size_tau
    xi    = float(torch.randn(1).item())
    tau_prop = float(max(cfg.tau_min, min(cfg.tau_max,
                                          tau_xcorr + delta * xi)))

    # Build proposed z
    z_prop = state.z.clone()
    z_prop[0, 0] = tau_prop

    # Log-proposal densities (symmetric about τ_xcorr)
    def log_q_tau(tau_val):
        return -0.5 * ((tau_val - tau_xcorr) / delta)**2

    log_q_fwd = log_q_tau(tau_prop)
    log_q_rev = log_q_tau(tau_curr)

    # Log-likelihood at proposal
    with torch.no_grad():
        _set_likelihood_params(likelihood, z_to_params(z_prop))
        ll_prop = float(likelihood.log_likelihood(y, state.x, z_prop).sum())

    # Log-prior
    lp_curr = float(log_z_prior(state.z))
    lp_prop = float(log_z_prior(z_prop))

    # MH ratio
    log_alpha = ((ll_prop + lp_prop + log_q_rev)
               - (state.log_lik + lp_curr + log_q_fwd))
    accepted = bool(math.log(max(float(torch.rand(1)), 1e-10)) < log_alpha)

    if accepted:
        _set_likelihood_params(likelihood, z_to_params(z_prop))
        new_state = MCMCState(
            x=state.x, z=z_prop, phi=state.phi,
            log_lik=ll_prop, step=state.step+1,
            n_acc_phi=state.n_acc_phi,
            n_acc_z=state.n_acc_z,
            n_acc_tau=state.n_acc_tau+1,
        )
    else:
        _set_likelihood_params(likelihood, z_to_params(state.z))
        new_state = MCMCState(
            x=state.x, z=state.z, phi=state.phi,
            log_lik=state.log_lik, step=state.step+1,
            n_acc_phi=state.n_acc_phi,
            n_acc_z=state.n_acc_z,
            n_acc_tau=state.n_acc_tau,
        )

    return new_state, accepted


# ======================================================================
# 2. a, b update — MALA on Z[1:] = [a, b]
# ======================================================================

def ab_mala_step(
    state       : MCMCState,
    y           : torch.Tensor,
    likelihood,
    log_z_prior : Callable,
    cfg         : MALAConfig,
    device      : torch.device,
) -> tuple[MCMCState, bool]:
    """
    MALA update for (a, b) — the amplitude and offset components of Z.

    τ is held fixed at state.tau during this step.
    Only Z[1] (a) and Z[2] (b) are updated via Langevin dynamics.
    """
    eps = cfg.step_size_z

    # We only differentiate w.r.t. a and b (Z[1:])
    ab_curr = state.z[:, 1:].detach().requires_grad_(True)  # (1, 2)

    # Reconstruct full z with grad on a,b
    z_with_grad = torch.cat([
        state.z[:, :1].detach(),   # tau fixed
        ab_curr
    ], dim=1)

    # Gradient of log target w.r.t. [a, b]
    _set_likelihood_params(likelihood, z_to_params(z_with_grad))
    ll  = likelihood.log_likelihood(y, state.x, z_with_grad).sum()
    lp  = log_z_prior(z_with_grad)
    grad = torch.autograd.grad(ll + lp, ab_curr)[0].detach()

    # Langevin proposal for [a, b]
    noise  = torch.randn_like(ab_curr)
    ab_prop = (ab_curr.detach() + 0.5 * eps**2 * grad + eps * noise).detach()

    # Build proposed z
    z_prop = torch.cat([
        state.z[:, :1].detach(),
        ab_prop,
    ], dim=1)

    # Log-densities at proposal
    with torch.no_grad():
        _set_likelihood_params(likelihood, z_to_params(z_prop))
        ll_prop = float(likelihood.log_likelihood(y, state.x, z_prop).sum())
        lp_prop = float(log_z_prior(z_prop))

    # Reverse gradient for MH correction
    ab_prop_g = ab_prop.requires_grad_(True)
    z_prop_g  = torch.cat([state.z[:, :1].detach(), ab_prop_g], dim=1)
    _set_likelihood_params(likelihood, z_to_params(z_prop_g))
    ll_r  = likelihood.log_likelihood(y, state.x, z_prop_g).sum()
    lp_r  = log_z_prior(z_prop_g)
    grad_r = torch.autograd.grad(ll_r + lp_r, ab_prop_g)[0].detach()
    ab_back = ab_prop.detach() + 0.5 * eps**2 * grad_r

    def _log_gaussian(x, mu, sigma):
        return -0.5 * ((x - mu)/sigma).pow(2).sum() - math.log(sigma)*x.numel()

    log_q_fwd = float(_log_gaussian(ab_prop, ab_curr.detach()+0.5*eps**2*grad, eps))
    log_q_rev = float(_log_gaussian(ab_curr.detach(), ab_back, eps))

    lp_curr = float(log_z_prior(state.z))
    log_alpha = ((ll_prop + lp_prop + log_q_rev)
               - (state.log_lik + lp_curr + log_q_fwd))
    accepted = bool(math.log(max(float(torch.rand(1)), 1e-10)) < log_alpha)

    if accepted:
        with torch.no_grad():
            new_ll = float(likelihood.log_likelihood(y, state.x, z_prop).sum())
        _set_likelihood_params(likelihood, z_to_params(z_prop))
        new_state = MCMCState(
            x=state.x, z=z_prop, phi=state.phi,
            log_lik=new_ll, step=state.step,
            n_acc_phi=state.n_acc_phi,
            n_acc_z=state.n_acc_z+1,
            n_acc_tau=state.n_acc_tau,
        )
        if cfg.adapt_step_size:
            cfg.step_size_z *= 1.02
    else:
        _set_likelihood_params(likelihood, z_to_params(state.z))
        new_state = MCMCState(
            x=state.x, z=state.z, phi=state.phi,
            log_lik=state.log_lik, step=state.step,
            n_acc_phi=state.n_acc_phi,
            n_acc_z=state.n_acc_z,
            n_acc_tau=state.n_acc_tau,
        )
        if cfg.adapt_step_size:
            cfg.step_size_z *= 0.98

    return new_state, accepted


# ======================================================================
# 3. Φ = log_σ² update (fixed in Part 1, kept for Part 2 compatibility)
# ======================================================================

def phi_mh_step(
    state       : MCMCState,
    y           : torch.Tensor,
    likelihood,
    fix_log_sigma2 : float | None = None,
    device      : torch.device = None,
) -> tuple[MCMCState, bool]:
    """
    For Part 1: Φ = log_σ² is fixed at true value.
    This is a no-op that just ensures likelihood params are consistent.

    For Part 2: this would be an MH step with q_λ* proposal over Φ.
    """
    if fix_log_sigma2 is not None:
        phi_new = torch.tensor([fix_log_sigma2], dtype=torch.float32)
        if device is not None:
            phi_new = phi_new.to(device)
        _set_likelihood_params(likelihood, phi_to_params(phi_new))
        new_state = MCMCState(
            x=state.x, z=state.z, phi=phi_new,
            log_lik=state.log_lik, step=state.step,
            n_acc_phi=state.n_acc_phi,
            n_acc_z=state.n_acc_z,
            n_acc_tau=state.n_acc_tau,
        )
        return new_state, False  # not a proposal, just enforcement
    # For Part 2 with q_lambda*, implement independence MH here
    return state, False


# ======================================================================
# 4. X update — annealed guided DDIM (unchanged from previous version)
# ======================================================================

def x_diffusion_step(
    state          : MCMCState,
    y              : torch.Tensor,
    prior_score_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    likelihood,
    diffusion,
    cfg            : MALAConfig,
    device         : torch.device,
) -> MCMCState:
    """
    Guided DDIM reverse pass for X.

    Starts from pure noise, denoises monotonically with likelihood guidance.
    Professor's correct score-space guidance formula.
    """
    x_t = torch.randn(1, 1, state.x.shape[-1], device=device)

    for i, t_val in enumerate(cfg.t_anneal):
        t_batch  = torch.full((1,), t_val,  dtype=torch.long, device=device)
        t_next   = cfg.t_anneal[i + 1] if i < len(cfg.t_anneal) - 1 else 0
        t_next_b = torch.full((1,), t_next, dtype=torch.long, device=device)

        abar_t    = diffusion._extract(diffusion.alphas_bar,      t_batch,  x_t.shape)
        abar_next = diffusion._extract(diffusion.alphas_bar_prev, t_next_b, x_t.shape)
        alpha_t   = abar_t.sqrt()
        sigma_t   = (1.0 - abar_t).clamp(min=1e-8).sqrt()
        alpha_next = abar_next.sqrt()
        sigma_next = (1.0 - abar_next).clamp(min=0).sqrt()

        x_t_g = x_t.requires_grad_(True)
        raw_g  = prior_score_fn(x_t_g, t_batch)
        x0hat_g, _ = diffusion.to_x0_and_eps(x_t_g, t_batch, raw_g)
        ll     = likelihood.log_likelihood(y, x0hat_g, state.z).sum()
        grad   = torch.autograd.grad(ll, x_t_g, allow_unused=True)[0]

        if grad is None or torch.isnan(grad).any():
            grad = torch.zeros_like(x_t)
        else:
            grad = grad.detach()

        with torch.no_grad():
            raw_d      = prior_score_fn(x_t.detach(), t_batch)
            x0hat_d, eps_d = diffusion.to_x0_and_eps(x_t.detach(), t_batch, raw_d)
            prior_norm = (eps_d.norm() / (sigma_t.mean() + 1e-8)).clamp(min=1e-8)

        lam      = cfg.gamma(t_val)
        snr_t    = abar_t / (1.0 - abar_t + 1e-8)
        omega_t  = snr_t / (1.0 + snr_t)
        lik_norm = grad.norm().clamp(min=1e-8)
        gamma_t  = float((omega_t * lam * prior_norm / lik_norm).clamp(max=50.0))

        with torch.no_grad():
            eps_post   = eps_d - sigma_t * gamma_t * grad
            x0hat_post = (x_t.detach() - sigma_t * eps_post) / (alpha_t + 1e-8)
            x0hat_post = x0hat_post.clamp(-5.0, 5.0)
            x_t = (alpha_next * x0hat_post + sigma_next * eps_post).detach()

        if torch.isnan(x_t).any():
            with torch.no_grad():
                x_t = (alpha_next * x0hat_d + sigma_next * eps_d).detach()

    x = x_t.detach()
    with torch.no_grad():
        new_log_lik = float(likelihood.log_likelihood(y, x, state.z).sum())

    return MCMCState(
        x=x, z=state.z, phi=state.phi,
        log_lik=new_log_lik, step=state.step,
        n_acc_phi=state.n_acc_phi,
        n_acc_z=state.n_acc_z,
        n_acc_tau=state.n_acc_tau,
    )
