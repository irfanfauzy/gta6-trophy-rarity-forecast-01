"""GTA VI PS5 trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's trophies, five quantiles (5/25/50/75/95%) of
its Exophase earn rate at the case's target, scored by pinball loss on the
logit scale. Python 3.13, standard library only, offline, deterministic.

Model
-----
Exophase's page is a snapshot: rate = earners / players_tracked, refreshed on
Exophase's own schedule (hours apart), so consecutive reads are often
byte-identical. The final case spans ~2 days of real history; the practice
cases are flat launch-night reads straddling one refresh, which no model can
foresee -- the practice only checks format and spread.

Per trophy the median is the median (in logit space) of the three FIT
members below (persistence is the shrinkage anchor, not a member: a flat
last rate is not a forecast of growth, and including it drags every
rising trophy's median back toward launch night):

  1. Log-time trend: OLS of logit(rate) on ln(1 + days since release),
     read at the target, fit over the whole history. Adoption after a
     launch is roughly linear in log-time on the days scale (a power law
     in time since launch): story trophies rise fast and saturate, grind/
     collectible trophies rise slowly. (The house curve baseline fits the
     same idea on ln(1 + hours) over the last 24 h only.)
  2. Recent trend: the same log-time fit over the last 24 h of reads
     only. If adoption bends toward saturation, or a player surge dilutes
     rates, the recent slope is the honest one; if it does not, this
     agrees with member 1. It guards the full-history fit against
     over-extrapolating a curve that is levelling or has turned.
  3. Earners/players decomposition: OLS of log(earners + 0.5) on log-time
     per trophy, divided by a global OLS of log(players_tracked) on
     log-time. This separates genuine adoption (more earners) from dilution
     (newly tracked players who have earned little yet) -- the effect that
     made every practice rate fall at the first refresh.

Each member's deviation from the last read (persistence) is shrunk by
n/(n+2) and clamped to +/-3.5 logit. Fits use only distinct snapshots
(identical consecutive reads collapsed) and need >= 3 distinct points
with real time spread; with no usable fit the median is persistence.
Regime change: when the recent fit and the full-history fit disagree by
more than 0.7 logit at the target, the median is the recent fit -- a bent
or turned curve makes the stale members outvote the honest one.

Structure the game itself imposes is enforced on the medians afterwards:
the platinum rate cannot exceed any other trophy's (it requires them all),
and the story chain (Chapter 1 -> ... -> final mission -> 100% Completion)
is non-increasing, matched by trophy name, never by hardcoded id.

Spread (logit sigma per trophy) combines a floor for refresh/staleness
uncertainty, a random-walk term growing with sqrt(horizon), the trend
fit's residual SD, the fit members' disagreement, and an extrapolation
term for how far the target sits beyond the last read in log-time,
capped at 2.0.
Quantiles are median + z_tau * sigma on the logit scale, z =
(-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals half-up.

I/O: reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Spread budget (logit units). FLOOR covers Exophase refresh / staleness
# uncertainty even on a 90-minute horizon; RW is the random-walk term.
SIGMA_FLOOR = 0.50
SIGMA_RW = 0.50
SIGMA_CAP = 2.0
MAX_DEVIATION = 3.5  # logit clamp on any fit's move away from persistence


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (same as JS Math.round(x*10000)/10000)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def distinct_snapshots(case):
    """Collapse consecutive identical reads; keep first-seen time of each."""
    snaps = []
    prev_key = None
    for h in case.get("history", []):
        rates = tuple(h.get("rates") or [])
        key = (h.get("players_tracked"), rates)
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(h)
    return snaps


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0

    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    try:
        rd = case.get("game", {}).get("release_date", "")
        y, m, d = (int(v) for v in rd.split("-"))
        release = datetime(y, m, d, tzinfo=timezone.utc)
    except Exception:
        pass

    def x_of(when):
        hours = (when - release).total_seconds() / 3600.0
        # Log-time on the DAYS scale: ln(1 + days since release). A power
        # law in time since launch is linear in this x, and the slope in x
        # is the same early and late; on the ln(1+hours) scale the slope
        # steepens with time, so a full-history fit there under-predicts.
        return math.log1p(max(hours, 0.0) / 24.0)

    snaps = distinct_snapshots(case)
    if not snaps:
        # No history at all: flat 1% with a wide spread (never crash).
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(logit(1.0) + z * 1.2)) for z in Z]}
            for t in trophies]}

    xs = [x_of(parse_time(h["at"])) for h in snaps]
    x_target = x_of(parse_time(case["target"]))
    n_distinct = len(snaps)
    shrink = n_distinct / (n_distinct + 2.0)

    # Global players-tracked growth: log(P) ~ log-time.
    players_fit = ols(xs, [math.log(max(h["players_tracked"], 1)) for h in snaps])
    if players_fit is not None:
        log_p_target = players_fit[0] + players_fit[1] * x_target
    else:
        log_p_target = math.log(max(snaps[-1]["players_tracked"], 1))

    # Last non-null rate per trophy.
    last_rate = []
    for i in range(n_tr):
        r = None
        for h in reversed(snaps):
            v = (h.get("rates") or [None] * n_tr)[i]
            if v is not None:
                r = float(v)
                break
        last_rate.append(0.5 if r is None else r)

    medians = []
    sigmas = []
    for i in range(n_tr):
        last_lg = logit(last_rate[i])
        # Fit members only; persistence is the shrinkage anchor, not a
        # member (see module docstring). No fit -> persistence.
        members = []
        resid = 0.1  # default fit uncertainty when no trend can be fit

        # Member 1: log-time trend on logit(rate).
        pairs = []
        for h, x in zip(snaps, xs):
            v = (h.get("rates") or [None] * n_tr)[i]
            if v is not None:
                pairs.append((x, logit(float(v)), parse_time(h["at"])))
        xs_ok = [p[0] for p in pairs]
        ys_logit = [p[1] for p in pairs]
        fit_a = ols(xs_ok, ys_logit) if len(xs_ok) >= 3 else None
        if fit_a is not None:
            a, b, rsd = fit_a
            resid = max(resid, rsd)
            pred = a + b * x_target
            dev = max(-MAX_DEVIATION, min(MAX_DEVIATION, pred - last_lg))
            members.append(last_lg + shrink * dev)

        # Member 2: recent (last 24 h) log-time trend on logit(rate).
        t_end = parse_time(snaps[-1]["at"])
        recent = [p for p in pairs if (t_end - p[2]).total_seconds() <= 24 * 3600]
        xs_r = [p[0] for p in recent]
        ys_r = [p[1] for p in recent]
        fit_r = ols(xs_r, ys_r) if len(xs_r) >= 3 else None
        recent_lg = None
        if fit_r is not None and (max(xs_r) - min(xs_r)) > 0.05:
            a, b, rsd = fit_r
            resid = max(resid, rsd)
            pred = a + b * x_target
            dev = max(-MAX_DEVIATION, min(MAX_DEVIATION, pred - last_lg))
            recent_lg = last_lg + shrink * dev
            members.append(recent_lg)

        # Member 3: earners / players decomposition.
        ys_log_e = []
        xs_e = []
        for h, x in zip(snaps, xs):
            v = (h.get("rates") or [None] * n_tr)[i]
            if v is not None:
                earners = float(v) / 100.0 * max(h["players_tracked"], 1)
                ys_log_e.append(math.log(earners + 0.5))
                xs_e.append(x)
        fit_b = ols(xs_e, ys_log_e) if len(xs_e) >= 3 else None
        if fit_b is not None:
            a, b, _ = fit_b
            log_e_target = a + b * x_target
            rate_b = 100.0 * math.exp(log_e_target - log_p_target)
            pred_lg = logit(rate_b)
            dev = max(-MAX_DEVIATION, min(MAX_DEVIATION, pred_lg - last_lg))
            members.append(last_lg + shrink * dev)

        members.sort()
        if members:
            median_lg = members[len(members) // 2] if len(members) % 2 == 1 else (
                members[len(members) // 2 - 1] + members[len(members) // 2]) / 2.0
            # Regime change: when the recent fit and the full-history fit
            # disagree by more than 0.7 logit at the target, the curve
            # has bent or turned (saturation, player surge) and the
            # stale members outvote the honest one in a median -- trust
            # the recent fit, which reflects the current regime.
            if recent_lg is not None and fit_a is not None:
                pred_a = fit_a[0] + fit_a[1] * x_target
                dev_a = max(-MAX_DEVIATION, min(MAX_DEVIATION, pred_a - last_lg))
                if abs(recent_lg - (last_lg + shrink * dev_a)) > 0.7:
                    median_lg = recent_lg
        else:
            median_lg = last_lg
        if len(members) > 1:
            mean_m = sum(members) / len(members)
            disagree = math.sqrt(sum((v - mean_m) ** 2 for v in members) / len(members))
        else:
            disagree = 0.0
        # Extrapolation risk: the target sits beyond the last read in
        # log-time; the further beyond, the less the fits can be trusted.
        extrap = 0.25 * max(0.0, x_target - xs[-1]) if xs else 0.0
        sigma = math.sqrt(SIGMA_FLOOR ** 2 + (SIGMA_RW * math.sqrt(horizon_days)) ** 2
                           + resid ** 2 + disagree ** 2 + extrap ** 2)
        sigma = min(SIGMA_CAP, sigma)
        medians.append(inv_logit(median_lg))
        sigmas.append(sigma)

    medians = enforce_structure(trophies, medians)

    rows = []
    for t, med, sigma in zip(trophies, medians, sigmas):
        med_lg = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(med_lg + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def enforce_structure(trophies, medians):
    """Game-imposed ordering on medians (percent): platinum <= every other
    trophy; the story chain Chapter 1..4 -> final mission -> 100% Completion
    is non-increasing. Matched by name/tier, never by hardcoded id."""
    med = list(medians)
    names = [(t.get("name") or "") for t in trophies]

    # Story chain, in play order.
    chain = []
    chapters = []
    for idx, name in enumerate(names):
        m = re.search(r"chapter\s+(\d+)", name, re.IGNORECASE)
        if m:
            chapters.append((int(m.group(1)), idx))
    chapters.sort()
    chain.extend(idx for _, idx in chapters)
    for idx, name in enumerate(names):
        if "final mission" in name.lower() and idx not in chain:
            chain.append(idx)
    for idx, name in enumerate(names):
        if "100% completion" in name.lower() and idx not in chain:
            chain.append(idx)
    for a, b in zip(chain, chain[1:]):
        if med[b] > med[a]:
            med[b] = med[a]

    # Platinum needs every other trophy.
    plat = [idx for idx, t in enumerate(trophies) if (t.get("tier") or "") == "platinum"]
    if plat:
        others = [med[i] for i in range(len(med)) if i not in plat]
        if others:
            cap = min(others)
            for idx in plat:
                if med[idx] > cap:
                    med[idx] = cap
    return med


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
