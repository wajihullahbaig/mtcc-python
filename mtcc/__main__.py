"""Command line interface.

  python -m mtcc extract  IMAGE -o template.npz [--variant cf] [--plot pipeline.png] [--plot-cylinder cyl.png]
  python -m mtcc match    A B [--variant cf]          (A, B: images or .npz templates)
  python -m mtcc evaluate DB_FOLDER [--variants o cf co] [--jobs 8] [--out results.json]
  python -m mtcc compare  RUN_FOLDER RUN_FOLDER [--variant co] [--plot eer.png]   (folders of evaluate .json)

All commands take --extractor {crossing_number,fingerjet_fx_ose} and --dpi (scan resolution, default 500).
"""
import argparse
import json
from pathlib import Path

from . import EXTRACTORS, VARIANTS, Params, extract_features, load_template, make_template, match, read_image, save_template


def _template(path, variant, p):
    """Load a .npz template or build one from an image."""
    if Path(path).suffix == '.npz':
        return load_template(path)
    f = extract_features(read_image(path), p)
    return make_template(f['minutiae'], f['texture'], f['mask'], variant, p)


def _compare(runs, variant, plot):
    """Print EER and FMR1000 of one variant side by side for several result folders; optionally plot the EERs."""
    res = {Path(r).name: {f.stem: json.loads(f.read_text())['results'][variant] for f in sorted(Path(r).glob('*.json'))}
           for r in runs}
    dbs = sorted(set.intersection(*(set(v) for v in res.values())))
    print(f"MCC_{variant}: EER % / FMR1000 %")
    print(f"{'database':16s}" + ''.join(f'{r:>26s}' for r in res))
    for d in dbs:
        print(f'{d:16s}' + ''.join(f"{res[r][d]['eer']:15.2f} / {res[r][d]['fmr1000']:7.2f}" for r in res))
    if plot:
        from .viz import plot_eer_comparison
        plot_eer_comparison({r: {d: res[r][d]['eer'] for d in dbs} for r in res}, plot, variant)


def main():
    """Parse arguments and run the extract / match / evaluate / compare command."""
    ap = argparse.ArgumentParser(prog='mtcc')
    sub = ap.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--extractor', default='crossing_number', choices=sorted(EXTRACTORS))
    common.add_argument('--dpi', type=int, default=500)
    e = sub.add_parser('extract', parents=[common])
    e.add_argument('image')
    e.add_argument('-o', '--out', required=True)
    e.add_argument('--variant', default='cf', choices=VARIANTS)
    e.add_argument('--plot')
    e.add_argument('--plot-cylinder')
    m = sub.add_parser('match', parents=[common])
    m.add_argument('a')
    m.add_argument('b')
    m.add_argument('--variant', default='cf', choices=VARIANTS)
    v = sub.add_parser('evaluate', parents=[common])
    v.add_argument('folder')
    v.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    v.add_argument('--jobs', type=int)
    v.add_argument('--out')
    c = sub.add_parser('compare')
    c.add_argument('runs', nargs='+', help='folders of evaluate --out .json files, one per run')
    c.add_argument('--variant', default='co', choices=VARIANTS)
    c.add_argument('--plot')
    args = ap.parse_args()
    for path in (getattr(args, k, None) for k in ('out', 'plot', 'plot_cylinder')):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == 'compare':
        return _compare(args.runs, args.variant, args.plot)
    p = Params(extractor=args.extractor, dpi=args.dpi)

    if args.cmd == 'extract':
        img = read_image(args.image)
        f = extract_features(img, p)
        save_template(args.out, make_template(f['minutiae'], f['texture'], f['mask'], args.variant, p))
        print(f"{len(f['minutiae'])} minutiae -> {args.out}")
        if args.plot:
            from .viz import plot_features
            plot_features(img, f, args.plot)
        if args.plot_cylinder:
            from .viz import plot_cylinder
            plot_cylinder(img, f, p, args.plot_cylinder)
    elif args.cmd == 'match':
        print(f"{match(_template(args.a, args.variant, p), _template(args.b, args.variant, p), p):.4f}")
    else:
        from .evaluate import evaluate
        res = evaluate(args.folder, args.variants, p, args.jobs)
        print(f"{'variant':8s}{'EER %':>8s}{'FMR1000 %':>11s}")
        for k, r in res.items():
            print(f"MCC_{k:4s}{r['eer']:8.2f}{r['fmr1000']:11.2f}")
        if args.out:
            summary = {k: {'eer': r['eer'], 'fmr1000': r['fmr1000']} for k, r in res.items()}
            Path(args.out).write_text(json.dumps({'folder': args.folder, 'params': p.to_dict(), 'results': summary},
                                                 indent=2, default=float))


if __name__ == '__main__':
    main()
