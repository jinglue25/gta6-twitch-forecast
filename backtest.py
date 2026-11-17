"""Reproducible leave-one-launch-out evaluation and a small parameter search.

Uses only p00.in's supplied comparables. No practice-case answers, networking,
release-day target leakage, or external dependencies. Run with --tune to search.
The best search result is evidence, not an unbiased estimate of future loss.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import main as model

DEFAULT_INPUT = Path('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/x/gta6-twitch-ccu-forecast-01-kit/input/p00.in')


def build_folds(comparables):
    folds = []
    for i, comp in enumerate(comparables):
        prepared = model.prepare_comparable(comp)
        if prepared is None:
            continue
        start = model.stamp(comp['day_start'])
        # A pseudo-case contains only information available by the checkpoint.
        history = []
        channels = None
        for row in comp['hourly']:
            try:
                at = model.stamp(row['from'])
                if at < start + 3600:
                    history.append(row)
                if start <= at < start + 3600:
                    channels = row.get('channels')
            except (ValueError, KeyError, TypeError):
                pass
        case = {'id': str(i), 'day_start': comp['day_start'],
                'known': [{'hour': 1, 'viewers': prepared['p1'], 'channels': channels}],
                'history_hourly': history, 'levels': list(model.LEVELS),
                'forecast': [{'hour': h} for h in range(2, 25)],
                'comparables': [c for j, c in enumerate(comparables) if j != i]}
        peers = [model.prepare_comparable(c) for c in case['comparables']]
        peers = [p for p in peers if p is not None]
        peers.sort(key=lambda p: (p['features'][0], p['features'][1] or 0,
                                  tuple(sorted(p['ratios'].items()))))
        folds.append((comp['name'], case, prepared, peers))
    return folds


def score(output, actual):
    loss, n = 0.0, 0
    for row in output['hours']:
        hour = row['hour']
        if hour - 1 not in actual:
            continue
        y = math.log(max(1, actual[hour - 1]))
        for tau, q in zip(model.LEVELS, row['q']):
            d = y - math.log(q)
            loss += 2 * max(tau * d, (tau - 1) * d)
            n += 1
    return loss, n


def evaluate(folds, verbose=True):
    losses, baselines, count = [], [], 0
    if verbose:
        print(f"{'Launch':45} {'Hours':>5} {'Model':>10} {'Baseline':>10}")
    for name, case, prepared, _ in folds:
        loss, n = score(model.forecast_case(case), prepared['hours'])
        baseline, nb = score(model.persistence(prepared['p1']), prepared['hours'])
        assert n == nb and n >= 16 * 5
        losses.append(loss)
        baselines.append(baseline)
        count += n
        if verbose:
            print(f'{name[:45]:45} {n // 5:5d} {loss / n:10.6f} {baseline / n:10.6f}')
    result = {'cases': len(folds), 'hours': count // 5, 'items': count,
              'model': math.fsum(losses) / count,
              'baseline': math.fsum(baselines) / count}
    if verbose:
        print(json.dumps(result, sort_keys=True))
    return result


def tune(folds):
    # Four interpretable knobs. No name/unlock matching or launch-specific rules.
    # Compute each distribution once, then test three global spread multipliers.
    results = []
    for scale, heat, beta in itertools.product((.6, .9, 1.5), (.4, .6, 1.0), (.1, .25, .4)):
        base_logs = [model.predict_logs(p['features'], peers, scale, heat, beta, 1.0)
                     for _, _, p, peers in folds]
        for alpha in (.85, 1.0, 1.15):
            total, count = 0.0, 0
            for (_, _, prepared, _), logs in zip(folds, base_logs):
                calibrated = [[qs[2] + alpha * (q - qs[2]) for q in qs] for qs in logs]
                loss, n = score(model.render(prepared['p1'], calibrated), prepared['hours'])
                total += loss
                count += n
            results.append((total / count, scale, heat, beta, alpha))
    results.sort()
    print('Top 10 grid choices: loss, s_scale, s_heat, beta, alpha')
    for result in results[:10]:
        print(' '.join(f'{x:.6f}' for x in result))
    best = results[0]
    print('Best:', best, flush=True)
    # Also expose launch-to-launch sensitivity for the selected choice.
    model.S_SCALE, model.S_HEAT, model.BETA, model.ALPHA = best[1:]
    print('Selected configuration (same folds used for tuning):')
    evaluate(folds)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    folds = build_folds(data['comparables'])
    print(f"Eligible launches: {len(folds)} / {len(data['comparables'])}")
    print('Current constants:', model.S_SCALE, model.S_HEAT, model.BETA, model.ALPHA)
    evaluate(folds)
    if args.tune:
        tune(folds)


if __name__ == '__main__':
    main()
