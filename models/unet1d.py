"""
models/unet1d.py
================
1-D denoising U-Net for the unconditional ECG diffusion prior.

Architecture
------------
Standard encoder–decoder U-Net with:
*   Sinusoidal timestep embedding → MLP → FiLM conditioning at every block
*   Additive skip connections (not concatenation — keeps channel count clean)
*   GroupNorm + SiLU activations throughout
*   Stride-2 Conv1d downsampling, ConvTranspose1d upsampling

Input/output
------------
x_t  : (B, 1, L)    noisy ECG window,  L = 4000 at 500 Hz
t    : (B,)  long   diffusion timestep index
→      (B, 1, L)    predicted x̂_0 (or ε̂ if pred_type="eps")

This U-Net is UNCONDITIONAL — it takes only (x_t, t), no conditioning
on PPG or any other signal. The likelihood provides the measurement
information; the prior provides ECG morphology.

Memory budget for RTX 3050 Ti (4 GB VRAM)
------------------------------------------
With base_ch=64, n_res=2, L=4000, B=8:
    ~1.2 GB activation memory + ~180 MB parameters → fits with room
    for gradients. Use B=4 if OOM during training.

Compatibility
-------------
*   ECGDiffusionPrior (models/prior.py) calls UNet1D(x_t, t) → raw_pred
*   GaussianDiffusion.to_x0_and_eps() converts raw_pred to (x̂_0, ε̂)
*   No changes needed here when switching between pred_type="x0"/"eps"
    — that logic lives in GaussianDiffusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Timestep embedding
# ======================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep t,
    projected through a 2-layer MLP to dimension `dim`.

    Output shape: (B, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) long — timestep indices

        Returns
        -------
        emb : (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ======================================================================
# Building blocks
# ======================================================================

class FiLMResBlock1D(nn.Module):
    """
    1-D residual block with FiLM conditioning on the timestep embedding.

    FiLM: h ← (1 + γ) ⊙ h + β
    where (γ, β) = Linear(cond_emb).unsqueeze(-1)

    This lets the timestep modulate feature maps at every layer,
    which is standard in diffusion U-Nets.

    Parameters
    ----------
    ch       : int   number of channels (in and out, same)
    cond_dim : int   dimension of the conditioning vector (= time_dim)
    groups   : int   GroupNorm groups (default 8)
    dropout  : float dropout probability on the second conv (default 0.0)
    """

    def __init__(
        self,
        ch       : int,
        cond_dim : int,
        groups   : int   = 8,
        dropout  : float = 0.0,
    ) -> None:
        super().__init__()
        g = min(groups, ch)
        self.norm1   = nn.GroupNorm(g, ch)
        self.conv1   = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.norm2   = nn.GroupNorm(g, ch)
        self.conv2   = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.film    = nn.Linear(cond_dim, 2 * ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, ch, L)
        cond : (B, cond_dim)
        → (B, ch, L)
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM modulation from timestep embedding
        gamma_beta = self.film(cond).unsqueeze(-1)          # (B, 2*ch, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)             # each (B, ch, 1)
        h = (1 + gamma) * h + beta

        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h                                         # residual


def _make_stage(n_blocks: int, ch: int, cond_dim: int,
                dropout: float = 0.0) -> nn.ModuleList:
    return nn.ModuleList([
        FiLMResBlock1D(ch, cond_dim, dropout=dropout)
        for _ in range(n_blocks)
    ])


# ======================================================================
# U-Net
# ======================================================================

class UNet1D(nn.Module):
    """
    Unconditional 1-D denoising U-Net for ECG.

    Architecture (default base_ch=64, n_res=2):

        in_conv  : 1   → 64    (stem)
        down1    : 64  → 64    (2 FiLM-ResBlocks)
        ds1      : 64  → 128   (stride-2 Conv1d, L → L/2)
        down2    : 128 → 128   (2 FiLM-ResBlocks)
        ds2      : 128 → 256   (stride-2 Conv1d, L/2 → L/4)
        mid      :             (2 FiLM-ResBlocks at bottleneck)
        us2      : 256 → 128   (ConvTranspose1d, L/4 → L/2) + skip
        up2      : 128 → 128   (2 FiLM-ResBlocks)
        us1      : 128 → 64    (ConvTranspose1d, L/2 → L) + skip
        up1      : 64  → 64    (2 FiLM-ResBlocks)
        out_conv : 64  → 1     (GroupNorm + SiLU + Conv1d)

    Parameters
    ----------
    base_ch  : int   stem channel width (default 64)
    time_dim : int   timestep embedding dimension (default 128)
    n_res    : int   residual blocks per U-Net stage (default 2)
    dropout  : float dropout in residual blocks (default 0.1)

    Notes
    -----
    *   Skip connections are additive (not concatenation).
        This keeps channel count constant through the decoder and
        avoids doubling memory at each skip.
    *   GroupNorm groups are clamped to min(8, ch) so narrow stages
        (ch < 8) don't crash.
    *   At L=4000 with 2 downsamples: bottleneck is at L/4 = 1000.
        This is wider than image U-Nets but appropriate for 8s ECG
        windows where the bottleneck must preserve beat-level structure.
    """

    def __init__(
        self,
        base_ch  : int   = 64,
        time_dim : int   = 128,
        n_res    : int   = 2,
        dropout  : float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # Timestep embedding
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        # Encoder
        self.in_conv  = nn.Conv1d(1, c1, kernel_size=3, padding=1)
        self.down1    = _make_stage(n_res, c1, time_dim, dropout)
        self.ds1      = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.down2    = _make_stage(n_res, c2, time_dim, dropout)
        self.ds2      = nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.mid      = _make_stage(n_res, c3, time_dim, dropout)

        # Decoder
        self.us2      = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up2      = _make_stage(n_res, c2, time_dim, dropout)
        self.us1      = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up1      = _make_stage(n_res, c1, time_dim, dropout)

        # Output head
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, c1), c1),
            nn.SiLU(),
            nn.Conv1d(c1, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, 1, L)   noisy ECG window
        t   : (B,)  long  diffusion timestep

        Returns
        -------
        pred : (B, 1, L)
            Predicted x̂_0 (or ε̂). GaussianDiffusion.to_x0_and_eps()
            converts this to whichever representation is needed.
        """
        cond = self.time_emb(t)                    # (B, time_dim)

        # Encoder
        h = self.in_conv(x_t)                      # (B, c1, L)
        for blk in self.down1:
            h = blk(h, cond)
        s1 = h                                     # skip at resolution L

        h = self.ds1(h)                            # (B, c2, L/2)
        for blk in self.down2:
            h = blk(h, cond)
        s2 = h                                     # skip at resolution L/2

        h = self.ds2(h)                            # (B, c3, L/4)

        # Bottleneck
        for blk in self.mid:
            h = blk(h, cond)

        # Decoder
        h = self.us2(h) + s2                       # (B, c2, L/2)  additive skip
        for blk in self.up2:
            h = blk(h, cond)

        h = self.us1(h) + s1                       # (B, c1, L)    additive skip
        for blk in self.up1:
            h = blk(h, cond)

        return self.out_conv(h)                    # (B, 1, L)


# ======================================================================
# Parameter count utility
# ======================================================================

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape and parameter check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet1D(base_ch=64, time_dim=128, n_res=2).to(device)
    B, L   = 4, 4000

    x_t = torch.randn(B, 1, L, device=device)
    t   = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        out = model(x_t, t)

    print(f"Input  : {tuple(x_t.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"Params : {count_parameters(model):,}")
    assert out.shape == x_t.shape, f"Shape mismatch: {out.shape} != {x_t.shape}"
    print("UNet1D shape check passed.")
