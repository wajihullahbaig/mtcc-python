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


def _singular(ax, sp, ms=14):
    """Draw cores (magenta circles) and deltas (orange triangles)."""
    for (key, marker, c) in (('core', 'o', 'tab:pink'), ('delta', '^', 'tab:orange')):
        if len(sp.get(key, ())):
            ax.plot(sp[key][:, 0], sp[key][:, 1], marker, mfc='none', mec=c, ms=ms, mew=2.5, label=key)


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
    _singular(axes.flat[-1], f['singular'])
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


def plot_eer_comparison(eer, path, variant='co', title=None):
    """Grouped horizontal bars of EER per database, one bar per run; eer = {run name: {database: EER %}}."""
    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colors = ['#2a78d6', '#eb6834', '#1baf7a']                    # categorical slots 1-3, in fixed order
    runs = list(eer)
    dbs = sorted(set.intersection(*(set(e) for e in eer.values())), reverse=True)
    h = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(dbs) + 1.6), facecolor=surface)
    ax.set_facecolor(surface)
    for n, r in enumerate(runs):
        y = np.arange(len(dbs)) + (len(runs) - 1 - 2 * n) * h / 2
        ax.barh(y, [eer[r][d] for d in dbs], height=h, color=colors[n], ec=surface, lw=2, label=r, zorder=2)
    ax.set_yticks(np.arange(len(dbs)), [d.replace('_a', '').replace('_', ' ').upper() for d in dbs])
    ax.set_xlabel('EER (%), lower is better', color=muted)
    ax.grid(axis='x', color=grid, lw=0.8, zorder=0)
    ax.tick_params(colors=muted, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title or f'MCC_{variant} EER by minutiae extractor (FVC protocol)', loc='left', color=ink, pad=28)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(runs), frameon=False, labelcolor=muted,
              handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def plot_matcher_heatmap(eer, path, title):
    """EER heatmap, matchers (rows, best mean first) x databases plus their mean; eer = {matcher: {database: EER %}}."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    ink, muted, surface = '#0b0b0b', '#52514e', '#fcfcfb'
    ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # sequential blue 100-700
    dbs = sorted(set.intersection(*(set(e) for e in eer.values())))
    rows = sorted(eer, key=lambda m: np.mean([eer[m][d] for d in dbs]))
    z = np.array([[eer[m][d] for d in dbs] + [np.mean([eer[m][d] for d in dbs])] for m in rows])
    cols = [d.replace('_a', '').replace('_', ' ').upper() for d in dbs] + ['Mean']
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.4, 0.36 * len(rows) + 1.6), facecolor=surface)
    norm = LogNorm(max(z.min(), 0.1), z.max())
    ax.imshow(z, cmap=LinearSegmentedColormap.from_list('seq', ramp), norm=norm, aspect='auto')
    best = np.round(z, 2) == np.round(z, 2).min(0)                  # ties at the shown precision
    for (i, j), v in np.ndenumerate(z):
        dark = norm(v) > 0.55
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8.5, color='white' if dark else ink,
                fontweight='bold' if best[i, j] else 'normal')
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha='right')
    ax.set_yticks(range(len(rows)), rows)
    ax.axvline(len(dbs) - 0.5, color=surface, lw=4)                  # spacer before the mean column
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color=surface, lw=2)
    ax.tick_params(which='both', colors=muted, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', color=ink, pad=12)
    fig.text(0.01, 0.005, 'EER (%), lower is better; bold = best per column; rows sorted by mean EER; log colour scale',
             color=muted, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
