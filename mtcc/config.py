"""All tunable parameters in one place. Cylinder/matching values follow Table IV of the MTCC paper."""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Params:
    # --- Segmentation (block-wise variance + morphology) ---
    seg_block: int = 16
    seg_thresh: float = 0.2        # block std threshold, relative to global std
    # --- SMQT ---
    smqt_levels: int = 8
    # --- STFT analysis / enhancement (Chikkerur et al.) ---
    stft_block: int = 14           # block step
    stft_overlap: int = 6          # overlap on each side -> window = block + 2 * overlap
    stft_nfft: int = 32
    min_period: float = 3.0        # plausible ridge period range (pixels)
    max_period: float = 25.0
    # --- Gabor filtering ---
    gabor_orients: int = 24
    gabor_k: float = 0.65          # sigma = k / ridge frequency
    # --- Minutiae ---
    trace_len: int = 10            # pixels followed along a ridge to estimate direction
    min_minutia_dist: float = 6.0  # closer pairs are removed as spurious
    border_dist: float = 12.0      # minutiae closer than this to the background are dropped
    # --- Cylinder (Table IV) ---
    R: float = 65.0
    NS: int = 18
    ND: int = 5
    sigma_s: float = 6.0
    sigma_d: float = 5 * np.pi / 36
    mu_psi: float = 5 / 1000
    tau_psi: float = 400.0
    min_vc: float = 0.20
    min_m: int = 1
    # --- Matching (Table IV) ---
    min_me: float = 0.20
    delta_theta: float = 2 * np.pi / 3
    mu_p: float = 30.0
    tau_p: float = 2 / 5
    min_np: int = 4
    max_np: int = 10
    w_r: float = 0.6
    mu_rho: tuple = (12.0, np.pi / 12, np.pi / 28)
    tau_rho: tuple = (-0.8, -30.0, -10.0)
    n_rel: int = 4

    def to_dict(self):
        """Parameters as a plain dict (for saving with results)."""
        return asdict(self)
