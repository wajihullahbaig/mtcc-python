"""Command line interface.

  python -m mtcc extract  IMAGE -o template.npz [--variant cf] [--plot pipeline.png] [--plot-cylinder cyl.png]
  python -m mtcc match    A B [--variant cf]          (A, B: images or .npz templates)
  python -m mtcc evaluate DB_FOLDER [--variants o cf co] [--jobs 8] [--out results.json]
"""
import argparse
import json
from pathlib import Path

from . import VARIANTS, Params, extract_features, load_template, make_template, match, read_image, save_template


def _template(path, variant, p):
    """Load a .npz template or build one from an image."""
    if Path(path).suffix == '.npz':
        return load_template(path)
    f = extract_features(read_image(path), p)
    return make_template(f['minutiae'], f['texture'], f['mask'], variant, p)


def main():
    """Parse arguments and run the extract / match / evaluate command."""
    ap = argparse.ArgumentParser(prog='mtcc')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('image')
    e.add_argument('-o', '--out', required=True)
    e.add_argument('--variant', default='cf', choices=VARIANTS)
    e.add_argument('--plot')
    e.add_argument('--plot-cylinder')
    m = sub.add_parser('match')
    m.add_argument('a')
    m.add_argument('b')
    m.add_argument('--variant', default='cf', choices=VARIANTS)
    v = sub.add_parser('evaluate')
    v.add_argument('folder')
    v.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    v.add_argument('--jobs', type=int)
    v.add_argument('--out')
    args = ap.parse_args()
    p = Params()

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
