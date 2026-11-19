# GTA VI PS5 trophy rarity forecaster

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast,
for each of the 55 GTA VI PlayStation 5 trophies, five quantiles (5/25/50/75/95%)
of its Exophase earn rate at 2026-11-26 00:00 UTC, scored by pinball loss on
the logit scale.

## Run

```
python3 main.py            # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Python 3.13, standard library only, offline, deterministic (two runs write
byte-identical output), ~0.1 s per case.

## Model (`main.py`)

Exophase's page is a snapshot — `rate = earners / players_tracked`, refreshed
on Exophase's own schedule (hours apart), so consecutive reads are often
identical. The median forecast per trophy is the median (in logit space) of
three fit members, each shrunk toward the last read and clamped to ±3.5 logit:

1. **Log-time trend** — OLS of `logit(rate)` on `ln(1 + days since release)`
   over the whole history. Adoption after a launch is roughly linear in
   log-time on the days scale (a power law in time since launch).
2. **Recent trend** — the same fit over the last 24 h only; guards against
   a curve that is levelling off or has turned (saturation, player surge).
3. **Earners/players decomposition** — OLS of `log(earners + 0.5)` on
   log-time per trophy, divided by a global OLS of `log(players_tracked)`
   on log-time. Separates genuine adoption from dilution by newly tracked
   players who have earned little yet — the effect that made every practice
   rate fall at the first refresh.

Fits use only distinct snapshots (identical consecutive reads collapsed).
**Regime change:** when the recent and full-history fits disagree by more
than 0.7 logit at the target, the median is the recent fit. Game structure
is enforced on the medians: platinum ≤ every other trophy, and the story
chain (Chapter 1 → … → final mission → 100% Completion) is non-increasing,
matched by name, never by hardcoded id.

**Spread** (logit σ per trophy) combines a floor for refresh/staleness
uncertainty, a random-walk term growing with √horizon, the fits' residual
SD, the fit members' disagreement, and an extrapolation term for how far
the target sits beyond the last read in log-time (capped at 2.0).

## Validation

- Practice cases p00/p01: valid format (55 rows, ordered quantiles in
  (0, 100)), deterministic across runs, ~0.1 s per case. Practice histories
  are flat launch-night reads, so no model can foresee the refresh in their
  answers; the practice checks format and spread only.
- Synthetic backtests (2 days of history generated from known adoption
  curves, target +4.5 days): power-law world pinball 0.246 vs persistence
  baseline 0.304; player-surge world 0.474 vs 0.630. An extreme-saturation
  world where the bend happens after the checkpoint (unforeseeable from
  history) scores 0.330 vs baseline 0.305 — the known failure mode of any
  trend extrapolation.
- Edge cases (single snapshot, empty history, all-null rates) all produce
  valid output and exit 0.
