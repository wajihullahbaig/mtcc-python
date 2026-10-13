"""Command line interface.

  python -m mtcc extract  IMAGE -o template.npz [--variant cf] [--plot pipeline.png] [--plot-cylinder cyl.png]
  python -m mtcc match    A B [--variant cf] [--matcher LSAR]   (A, B: images or .npz templates)
  python -m mtcc evaluate DB_FOLDER [--variants o co] [--matchers LSSR NLSADTR] [--jobs 8] [--out results.json] [--scores scores.npz]
  python -m mtcc compare  RUN_FOLDER RUN_FOLDER [--variant co] [--matcher LSAR] [--plot eer.png]
                          (folders of evaluate .json)
  python -m mtcc matchers RUN_FOLDER [--variant co] [--metric eer] [--plot matchers.png]   (Markdown table)

All commands take --extractor {crossing_number,fingerjet_fx_ose} and --dpi (scan resolution, default 500).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import (EXTRACTORS, MATCHERS, VARIANTS, Params, extract_features, load_template, make_template, match,
               read_image, save_template, stages)


def _template(path, variant, p):
    """Load a .npz template or build one from an image."""
    if Path(path).suffix == '.npz':
        return load_template(path)
    f = extract_features(read_image(path), p)
    return make_template(f, variant, p)


def _matcher(name):
    """argparse type: a valid matcher name."""
    try:
        stages(name)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return name


def _compare(runs, variant, matcher, plot):
    """Print EER and FMR1000 of one variant and matcher side by side for several result folders; optionally plot."""
    res = {Path(r).name: {f.stem: json.loads(f.read_text())['results'][matcher][variant]
                          for f in sorted(Path(r).glob('*.json'))} for r in runs}
    dbs = sorted(set.intersection(*(set(v) for v in res.values())))
    print(f"MCC_{variant}, {matcher}: EER % / FMR1000 %")
    print(f"{'database':16s}" + ''.join(f'{r:>26s}' for r in res))
    for d in dbs:
        print(f'{d:16s}' + ''.join(f"{res[r][d]['eer']:15.2f} / {res[r][d]['fmr1000']:7.2f}" for r in res))
    if plot:
        from .viz import plot_eer_comparison
        plot_eer_comparison({r: {d: res[r][d]['eer'] for d in dbs} for r in res}, plot, variant)


def _matchers(run, variant, metric, plot):
    """Print a Markdown table of one metric for every matcher x database of a result folder; optionally plot EERs."""
    res = {f.stem: json.loads(f.read_text())['results'] for f in sorted(Path(run).glob('*.json'))}
    dbs = list(res)
    names = [m for m in next(iter(res.values())) if all(m in r for r in res.values())]
    val = {m: {d: res[d][m][variant][metric] for d in dbs} for m in names}
    names.sort(key=lambda m: sum(val[m].values()))
    print('| Matcher | ' + ' | '.join(d.replace('_a', '').replace('_', ' ').upper() for d in dbs) + ' | Mean |')
    print('|---' + '|---:' * (len(dbs) + 1) + '|')
    best = {d: min(round(val[m][d], 2) for m in names) for d in dbs}
    cell = lambda v, b: f'**{v:.2f}**' if round(v, 2) == b else f'{v:.2f}'
    for m in names:
        mean = sum(val[m].values()) / len(dbs)
        print(f'| {m} | ' + ' | '.join(cell(val[m][d], best[d]) for d in dbs) + f' | {mean:.2f} |')
    if plot:
        from .viz import plot_matcher_heatmap
        plot_matcher_heatmap(val, plot, f'MCC_{variant} {metric.upper()} by matcher ({Path(run).name} minutiae)')


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
    m.add_argument('--matcher', default=Params.matcher, type=_matcher)
    v = sub.add_parser('evaluate', parents=[common])
    v.add_argument('folder')
    v.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    v.add_argument('--matchers', nargs='+', default=[Params.matcher], type=_matcher, help=f"e.g. {' '.join(MATCHERS)}")
    v.add_argument('--jobs', type=int)
    v.add_argument('--out')
    v.add_argument('--scores', help='save per-pair genuine/impostor scores (.npz) for later fusion or learning')
    c = sub.add_parser('compare')
    c.add_argument('runs', nargs='+', help='folders of evaluate --out .json files, one per run')
    c.add_argument('--variant', default='co', choices=VARIANTS + ('o+co',))
    c.add_argument('--matcher', default=Params.matcher, type=_matcher)
    c.add_argument('--plot')
    t = sub.add_parser('matchers')
    t.add_argument('run', help='folder of evaluate --out .json files')
    t.add_argument('--variant', default='co', choices=VARIANTS + ('o+co',))
    t.add_argument('--metric', default='eer', choices=('eer', 'fmr1000'))
    t.add_argument('--plot')
    args = ap.parse_args()
    for path in (getattr(args, k, None) for k in ('out', 'plot', 'plot_cylinder')):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == 'compare':
        return _compare(args.runs, args.variant, args.matcher, args.plot)
    if args.cmd == 'matchers':
        return _matchers(args.run, args.variant, args.metric, args.plot)
    p = Params(extractor=args.extractor, dpi=args.dpi, matcher=getattr(args, 'matcher', Params.matcher))

    if args.cmd == 'extract':
        img = read_image(args.image)
        f = extract_features(img, p)
        save_template(args.out, make_template(f, args.variant, p))
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
        res = evaluate(args.folder, args.variants, args.matchers, p, args.jobs)
        print('EER % / FMR1000 %')
        shown = list(next(iter(res.values())))
        print(f"{'matcher':14s}" + ''.join(f'{"MCC_" + v:>16s}' for v in shown))
        for m, r in res.items():
            print(f'{m:14s}' + ''.join(f"{r[v]['eer']:7.2f} / {r[v]['fmr1000']:6.2f}" for v in shown))
        if args.scores:
            Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.scores, **{f'{m}/{v}/{k}': x[k] for m, r in res.items() for v, x in r.items()
                                                for k in ('genuine', 'impostor')})
        if args.out:
            summary = {m: {v: {'eer': x['eer'], 'fmr1000': x['fmr1000']} for v, x in r.items()} for m, r in res.items()}
            Path(args.out).write_text(json.dumps({'folder': args.folder, 'params': p.to_dict(), 'results': summary},
                                                 indent=2, default=float))


if __name__ == '__main__':
    main()
