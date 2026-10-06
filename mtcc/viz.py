"""Figures: pipeline stages / texture maps, and one MTCC cylinder slice per variant."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

from .cylinder import VARIANTS, cylinders


def _minutiae(ax, m, ms=6):
    """Draw minutiae (green endings, cyan bifurcations) with their directions."""
    for x, y, t, kind, _ in m:
        c = 'tab:green' if kind == 1 else 'tab:cyan'
        ax.plot(x, y, 'o', mfc='none', mec=c, ms=ms)
        ax.plot([x, x + 12 * np.cos(t)], [y, y + 12 * np.sin(t)], c=c, lw=1.5)


def plot_features(img, f, path):
    """Twelve-panel figure of every enhancement stage, texture map and the minutiae."""
    fg = lambda k: np.where(f['mask'], f[k], np.nan)
    panels = [('Original', img, 'gray'), ('Segmentation mask', f['mask'], 'gray'),
              ('SMQT', fg('smqt'), 'gray'), ('STFT enhanced', fg('stft'), 'gray'),
              ('Gabor response', fg('gabor'), 'gray'), ('Binary ridges', f['ridges'], 'gray'),
              ('Skeleton', ~skeletonize(f['ridges']), 'gray'), ('Orientation $I_o$', fg('orientation'), 'hsv'),
              ('Coherence', fg('coherence'), 'magma'), ('Frequency $I_f$', fg('frequency'), 'viridis'),
              ('Energy $I_e$', fg('energy'), 'viridis'), (f"Minutiae ({len(f['minutiae'])})", img, 'gray')]
    fig, axes = plt.subplots(3, 4, figsize=(16, 14))
    for ax, (title, im, cmap) in zip(axes.flat, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    _minutiae(axes.flat[-1], f['minutiae'])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_cylinder(img, f, p, path, k=None):
    """The cylinder of the minutia with most neighbours (among valid cylinders), and slice k for every variant."""
    m = f['minutiae']
    k = p.ND // 2 if k is None else k
    cyl = {v: cylinders(m, f['texture'], f['mask'], v, p) for v in VARIANTS}
    valid = cyl['o'][1].reshape(len(m), p.NS, p.NS, p.ND)[..., 0]
    neighbours = (np.hypot(m[:, None, 0] - m[:, 0], m[:, None, 1] - m[:, 1]) <= p.R).sum(1)
    idx = int(np.argmax(neighbours * cyl['o'][2]))
    x, y, t = m[idx, :3]

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 5)
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(img, cmap='gray')
    _minutiae(ax, m, ms=4)
    off = (np.arange(1, p.NS + 1) - (p.NS + 1) / 2) * (2 * p.R / p.NS)
    I, J = np.meshgrid(off, off, indexing='ij')
    px, py = x + np.cos(t) * I - np.sin(t) * J, y + np.sin(t) * I + np.cos(t) * J
    ax.plot(px[valid[idx]], py[valid[idx]], '.', c='tab:orange', ms=3)
    ax.add_patch(plt.Circle((x, y), p.R, fill=False, ec='tab:red', lw=1.5))
    ax.plot(x, y, 'o', mfc='none', mec='tab:red', ms=10, mew=2)
    ax.set_title(f'Cylinder of minutia #{idx} (R={p.R:g}, cell centres in orange)')
    ax.axis('off')

    phi_k = -np.pi + (k + 0.5) * 2 * np.pi / p.ND
    for n, v in enumerate(VARIANTS):
        vals = cyl[v][0].reshape(len(m), p.NS, p.NS, p.ND)[idx, :, :, k]
        a = fig.add_subplot(gs[n // 3, 2 + n % 3])
        im = a.imshow(np.where(valid[idx], vals, np.nan).T, cmap='inferno', vmin=0, vmax=1)
        a.set_title(f'MCC_{v}')
        a.axis('off')
    fig.colorbar(im, ax=fig.axes[1:], shrink=0.6, label='cell value $C_m(i,j,k)$')
    fig.suptitle(f'Slice k={k + 1} of {p.ND} ($d\\varphi_k$={np.degrees(phi_k):.0f}°); '
                 'minutia direction points right, invalid cells blank')
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
