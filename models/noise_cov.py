"""
models/noise_cov.py
===================
Configurable noise covariance Σ for the likelihood Y = H_φ(X,Z) + Σ^{1/2} ε,
ε ~ N(0, I_m).

Design
------
Three modes, all reached through the same diagonal path so the interface is
identical regardless of mode. "Isotropic" is diagonal with all entries equal;
"full diagonal" allows per-timestep entries. "Full" (future) would store a
Cholesky factor but is not needed for Part 1.

The class stores the *log* of the diagonal entries so that σ² > 0 is enforced
automatically during optimization. VI updates log_diag directly.

Why not hardcode σ²I
---------------------
Your advisor confirmed Σ = σ²I is a special case reached via the diagonal
path, not a distinct implementation. This module makes the transition from
isotropic (Part 1) to full diagonal (Part 2) a one-line config change.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn


class CovMode(str, Enum):
    ISOTROPIC = "isotropic"   # single σ², all diagonal entries equal
    DIAGONAL  = "diagonal"    # per-timestep σ_i², full diagonal vector
    # FULL    = "full"        # future: Cholesky factor, not needed Part 1


class NoiseCov(nn.Module):
    """
    Parametrizes the noise covariance Σ as a diagonal matrix.

    Parameters
    ----------
    dim : int
        Observation dimension m (number of PPG timesteps per window).
    mode : CovMode
        "isotropic" — one learnable log σ², broadcast to all dims.
        "diagonal"  — m learnable log σ_i², one per timestep.
    init_log_var : float
        Initial value of log σ² (log variance, not log std). Default 0.0
        corresponds to σ² = 1, i.e. unit noise before any updates.

    Learnable parameters
    --------------------
    log_diag : nn.Parameter, shape (1,) or (dim,)
        Log of the diagonal variance entries. Always positive after exp().
        VI updates this directly.

    Notes
    -----
    *   The module is owned by the likelihood (likelihood.py), not by VI.
        VI calls .log_diag and updates it via the ELBO gradient.
    *   All operations are in log space to avoid positivity constraints.
    *   σ² = exp(log_diag); Σ^{1/2} diag entries = exp(0.5 * log_diag).
    """

    def __init__(
        self,
        dim: int,
        mode: CovMode | Literal["isotropic", "diagonal"] = CovMode.ISOTROPIC,
        init_log_var: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim  = dim
        self.mode = CovMode(mode)

        n_params = 1 if self.mode == CovMode.ISOTROPIC else dim
        self.log_diag = nn.Parameter(
            torch.full((n_params,), init_log_var)
        )

    # ------------------------------------------------------------------
    def diag_var(self) -> torch.Tensor:
        """
        Returns the diagonal variance vector σ², shape (dim,).

        Always positive (exp of log_diag, broadcast if isotropic).
        """
        v = self.log_diag.exp()
        return v.expand(self.dim) if self.mode == CovMode.ISOTROPIC else v

    def diag_std(self) -> torch.Tensor:
        """
        Returns the diagonal std vector σ, shape (dim,).
        Σ^{1/2} applied to a zero-mean unit vector = diag_std() * vector.
        """
        return self.diag_var().sqrt()

    def log_det(self) -> torch.Tensor:
        """
        Returns log |Σ| = sum of log diagonal variances. Scalar.
        Used in the Gaussian log-likelihood normalisation term.
        """
        return self.log_diag.expand(self.dim).sum()

    def mahal(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mahalanobis term r^T Σ^{-1} r for a residual vector.

        Parameters
        ----------
        residual : torch.Tensor, shape (..., dim)
            Residual r = y - H_φ(x, z), already flattened to (dim,) or
            batched as (batch, dim).

        Returns
        -------
        torch.Tensor, shape (,) or (batch,)
            Scalar (or per-batch scalar) r^T Σ^{-1} r.
        """
        inv_var = 1.0 / self.diag_var()          # (dim,)
        return (residual.pow(2) * inv_var).sum(dim=-1)

    def sample_noise(self, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Draw noise ε and return Σ^{1/2} ε, shape (*shape, dim).
        Used in forward simulation (generate.py) to construct synthetic PPG.

        Parameters
        ----------
        shape : tuple[int, ...]
            Batch dimensions before the dim axis, e.g. (N,) for N windows.

        Returns
        -------
        torch.Tensor, shape (*shape, dim)
        """
        eps = torch.randn(*shape, self.dim, device=self.log_diag.device)
        return eps * self.diag_std()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mode={self.mode.value}"
