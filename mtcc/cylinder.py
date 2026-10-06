"""MCC / MTCC cylinders (Cappelli et al. 2010; Baig et al. 2018, Eqs. 1-14).

Variants (the directional contribution C^D is replaced by texture for MTCC):
  o  : minutia angle difference d_theta(m, m_t)                  (classic MCC)
  f  : frequency at m vs frequency at neighbour m_t              (Eq. 10)
  e  : energy at m vs energy at neighbour m_t                    (Eq. 11)
  co : minutia angle vs orientation at the cell centre p_ij      (Eq. 12)
  cf : frequency at m vs frequency at the cell centre p_ij       (Eq. 13)
  ce : energy at m vs energy at the cell centre p_ij             (Eq. 14)
"""
import numpy as np
from scipy.special import erf, expit

VARIANTS = ('o', 'f', 'e', 'co', 'cf', 'ce')
_MAP = {'o': 'orientation', 'co': 'orientation', 'f': 'frequency', 'cf': 'frequency', 'e': 'energy', 'ce': 'energy'}


def d_phi(a, b):
    """Difference of two angles wrapped to [-pi, pi) (Eq. 8)."""
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def texture_angles(maps, mask):
    """I_o as is; I_f and I_e min-max normalised over the foreground to [-pi, pi]."""
    out = {'orientation': maps['orientation']}
    for k in ('frequency', 'energy'):
        lo, hi = np.percentile(maps[k][mask], [1, 99]) if mask.any() else (0, 1)
        out[k] = ((np.clip((maps[k] - lo) / (hi - lo + 1e-12), 0, 1) * 2 - 1) * np.pi).astype(np.float32)
    return out


def _sample(img, x, y, fill=0.0):
    """Nearest-pixel lookup of img at (x, y); `fill` outside the image."""
    h, w = img.shape
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(np.shape(x), fill, np.float64)
    out[ok] = img[yi[ok], xi[ok]]
    return out


def _g_s(t, p):
    """Spatial Gaussian G_S of distance t."""
    return np.exp(-t ** 2 / (2 * p.sigma_s ** 2)) / (p.sigma_s * np.sqrt(2 * np.pi))


def _g_d(alpha, p):
    """Gaussian integrated over the angular extent of one directional cell."""
    half, s = np.pi / p.ND, p.sigma_d * np.sqrt(2)
    return 0.5 * (erf((alpha + half) / s) - erf((alpha - half) / s))


def cylinders(minutiae, texture, mask, variant, p):
    """Cylinder values and cell validity, both (n, NS*NS*ND), plus per-cylinder validity (n,)."""
    x, y, t = minutiae[:, 0], minutiae[:, 1], minutiae[:, 2]
    n = len(x)
    off = (np.arange(1, p.NS + 1) - (p.NS + 1) / 2) * (2 * p.R / p.NS)
    I, J = np.meshgrid(off, off, indexing='ij')

    # Cell centres p_ij, rotated by the minutia direction (Eq. 2, image axes).
    c, s = np.cos(t)[:, None, None], np.sin(t)[:, None, None]
    px = x[:, None, None] + c * I - s * J
    py = y[:, None, None] + s * I + c * J
    inside = I ** 2 + J ** 2 <= p.R ** 2
    valid = inside & (_sample(mask.astype(np.float32), px, py) > 0)

    # Spatial contribution of each neighbour within 3 sigma_S of each cell centre (Eq. 6).
    dist = np.hypot(px[..., None] - x, py[..., None] - y)          # (n, NS, NS, n)
    cs = np.where(dist <= 3 * p.sigma_s, _g_s(dist, p), 0)
    cs[np.arange(n), :, :, np.arange(n)] = 0

    phi_k = -np.pi + (np.arange(1, p.ND + 1) - 0.5) * 2 * np.pi / p.ND   # Eq. 1
    tex = texture[_MAP[variant]]
    v_m = t if variant in ('o', 'co') else _sample(tex, x, y)
    if variant in ('o', 'f', 'e'):
        d_theta = d_phi(v_m[:, None], v_m[None, :])                 # m vs neighbour m_t
        cd = _g_d(d_phi(phi_k, d_theta[..., None]), p)              # (n, n, ND)
        vals = np.einsum('mijt,mtk->mijk', cs, cd)
    else:
        d_theta = d_phi(v_m[:, None, None], _sample(tex, px, py))  # m vs cell centre p_ij
        cd = _g_d(d_phi(phi_k, d_theta[..., None]), p)              # (n, NS, NS, ND)
        vals = cs.sum(-1)[..., None] * cd
    vals = expit(p.tau_psi * (vals - p.mu_psi))                       # Psi sigmoid (Eq. 3)

    d_mm = np.hypot(x[:, None] - x, y[:, None] - y)
    np.fill_diagonal(d_mm, np.inf)
    ok = (valid.sum((1, 2)) >= p.min_vc * inside.sum()) & ((d_mm <= p.R + 3 * p.sigma_s).sum(1) >= p.min_m)
    valid = np.repeat(valid[..., None], p.ND, -1)
    return (vals * valid).reshape(n, -1), valid.reshape(n, -1), ok


def make_template(minutiae, texture, mask, variant, p):
    """Template = valid cylinders and their minutiae."""
    if len(minutiae) < 2:
        nc = p.NS * p.NS * p.ND
        return {'variant': variant, 'minutiae': np.zeros((0, 5)), 'values': np.zeros((0, nc), np.float16),
                'valid': np.zeros((0, nc), bool)}
    vals, valid, ok = cylinders(minutiae, texture, mask, variant, p)
    return {'variant': variant, 'minutiae': minutiae[ok], 'values': vals[ok].astype(np.float16), 'valid': valid[ok]}
