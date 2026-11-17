"""Deterministic empirical launch-curve quantiles; Python standard library only."""
import bisect
import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import NormalDist

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
# Chosen by the expanded-grid leave-one-launch-out tuning search in backtest.py.
S_SCALE = 1.5
S_HEAT = 1.0
BETA = 0.1
ALPHA = 1.0
FLOOR = 0.02
# Tiny hour-one categories can grow by thousands of times at a global unlock.
# Cover those observed ratios rather than truncating all their upper quantiles.
GRID = tuple(-10.0 + i * 0.01 for i in range(2001))
NORMAL_CDFS = {h: tuple(NormalDist(0, .15 * math.sqrt(h - 1)).cdf(x)
                        for x in GRID) for h in range(2, 25)}


def positive(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def stamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def index_hours(rows, start):
    """Ignore bad samples and average duplicates in each relative hour."""
    buckets = {}
    for row in rows or []:
        try:
            value = positive(row.get('viewers'))
            if value is None:
                continue
            hour = math.floor((stamp(row['from']) - start) / 3600)
            buckets.setdefault(hour, []).append(value)
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            continue
    return {h: math.fsum(sorted(buckets[h])) / len(buckets[h])
            for h in sorted(buckets)}


def features(hours, p1):
    pre = [hours[h] for h in range(-24, 0) if h in hours]
    return (math.log(p1), math.log(math.fsum(pre) / len(pre) / p1) if pre else None)


def prepare_comparable(comp):
    try:
        hours = index_hours(comp.get('hourly'), stamp(comp['day_start']))
        if 0 not in hours or sum(h in hours for h in range(1, 24)) < 16:
            return None
        p1 = hours[0]
        return {'features': features(hours, p1),
                'ratios': {h: math.log(hours[h - 1]) - math.log(p1)
                           for h in range(2, 25) if h - 1 in hours},
                'hours': hours, 'p1': p1}
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return None


def predict_logs(target, peers, s_scale, s_heat, beta, alpha):
    weighted = []
    for peer in peers:
        distance = 0.0
        for a, b, bandwidth in zip(target, peer['features'], (s_scale, s_heat)):
            if a is not None and b is not None:
                distance += ((a - b) / bandwidth) ** 2
        weighted.append((peer, max(FLOOR, math.exp(-.5 * distance))))
    result = []
    for h in range(2, 25):
        samples = sorted((p['ratios'][h], w) for p, w in weighted if h in p['ratios'])
        if not samples:
            result.append([z * .15 * math.sqrt(h - 1) for z in Z])
            continue
        xs = [x for x, _ in samples]
        total = math.fsum(w for _, w in samples)
        prefix = [0.0]
        for _, w in samples:
            prefix.append(prefix[-1] + w / total)
        normal = NORMAL_CDFS[h]
        def cdf(i):
            return (1 - beta) * prefix[bisect.bisect_right(xs, GRID[i])] + beta * normal[i]
        quantiles = []
        for tau in LEVELS:
            lo, hi = 0, len(GRID) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cdf(mid) < tau:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0:
                q = GRID[0]
            else:
                left, right = cdf(lo - 1), cdf(lo)
                q = GRID[lo - 1] + .01 * min(1.0, max(0.0, (tau - left) / max(1e-15, right - left)))
            quantiles.append(q)
        median = quantiles[2]
        result.append([median + alpha * (q - median) for q in quantiles])
    return result


def render(p1, logs):
    rows = []
    for h, qs in zip(range(2, 25), logs):
        values = []
        for q in qs:
            log_value = min(math.log(1e9), max(math.log(.01), math.log(p1) + q))
            value = float(Decimal(str(min(1e9, math.exp(log_value)))).quantize(
                Decimal('.01'), rounding=ROUND_HALF_UP))
            values.append(max(.01, values[-1] if values else .01, value))
        rows.append({'hour': h, 'q': values})
    return {'hours': rows}


def first_count(case, default=100000.0):
    try:
        known = case.get('known', [])
        if isinstance(known, dict):
            known = [known]
        for row in known:
            if row.get('hour') == 1:
                return positive(row.get('viewers')) or default
    except (AttributeError, TypeError):
        pass
    return default


def persistence(p1):
    return render(p1, [[z * .15 * math.sqrt(h - 1) for z in Z] for h in range(2, 25)])


def forecast_case(case: dict) -> dict:
    """Forecast all 23 hours, recovering to persistence on unusable input."""
    actual_p1 = first_count(case, default=None)
    p1 = actual_p1 or 100000.0
    if actual_p1 is None:
        return persistence(p1)
    try:
        history = index_hours(case.get('history_hourly'), stamp(case['day_start']))
        peers = []
        for comp in case.get('comparables') or []:
            prepared = prepare_comparable(comp)
            if prepared is not None:
                peers.append(prepared)
        # Canonical order also makes forecasts invariant to comparable list order.
        peers.sort(key=lambda p: (p['features'][0], p['features'][1] or 0,
                                  tuple(sorted(p['ratios'].items()))))
        if not peers:
            return persistence(p1)
        return render(p1, predict_logs(features(history, p1), peers,
                                      S_SCALE, S_HEAT, BETA, ALPHA))
    except Exception:
        return persistence(p1)


def main():
    source = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
    destination = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('*.in')):
        case = None
        try:
            case = json.loads(path.read_text(encoding='utf-8'))
            output = forecast_case(case)
        except Exception:
            output = persistence(first_count(case))
        # Use a safe case id, with filename stem as a fallback for malformed data.
        identity = case.get('id') if isinstance(case, dict) else None
        if (not isinstance(identity, str) or not identity or identity in ('.', '..')
                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. '
                       for c in identity) or len(identity) > 128):
            identity = path.stem
        (destination / (identity + '.out')).write_text(
            json.dumps(output, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
