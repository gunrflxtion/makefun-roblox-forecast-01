#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun: roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (reads of Roblox's Up-and-Coming chart) and
/input/task.json, and writes /output/forecast.json: for each of the top 10
places, a probability for each game named being at that place, plus the
best-guess top 10 in order. One static forecast is scored against every
hourly read of the 72 hours after entries close, so the program forecasts
the *time-average* of each game's place distribution over that window.

Method (per game, each place scored on its own, so only the marginals
matter):
  1. Rank dynamics. A game's rank is a noisy walk: from its reads, take a
     recency-weighted (12 h time constant) linear trend of rank against time
     (the slope is the game's momentum, clamped to +-0.6 ranks/hour, and
     it decays with a 36 h time constant, so a climb cannot run forever),
     plus diffusion whose per-hour volatility depends on the rank itself
     (measured from the archive: ~0.19 ranks/h at the top, ~1.46 at the
     bottom). At each future hour the game's rank is a normal
     distribution around the projected mean with that spread.
  2. Rank prior. The normal model alone underweights the chart's churn:
     games ranked 11-20 reach the top 10 within 72 h in 27% of cases,
     and even games at rank 71-101 do so in 3%. So each game's
     distribution is blended (40%) with a geometric rank prior
     (p ~ rank^-1 over the last chart), which spreads honest mass over
     the climbers.
  3. Age. Games whose universe is older than 28 days leave the chart
     more often (median age of leavers is 28.2 days), so when
     details.created is available (house-recorder reads) their mass is
     scaled by 0.7. This never zeroes a game: a third of the chart is
     older than 28 days and still ranks.
  4. New games. Games never seen cannot be named, yet about 8% of the
     top-10 slots in the practice window were games that first appeared
     after the history ended. So at least 30% of every place is left
     unnamed: the scorer shares it equally over the games of each read
     the forecast does not name there, which is what covers the
     newcomers. Only the 12 most likely games are named at each place:
     naming a game whose model probability is below the unnamed share
     per game wastes mass, so the tail is deliberately left to the
     unnamed share. (The 30% is a floor, not an exact share: the named
     probabilities are capped at 70% of a place.) One exception: at the
     tail places (8-10 on the kit input) the top 12 sum to less than
     the 70% budget (0.68 / 0.62 / 0.58), so the next candidates are
     named too, up to 30 per place, until the budget is filled or the
     next candidate would overshoot it — a candidate with model
     probability well above the unnamed share per game (~0.004) earns
     its row. On the kit input that adds one row at place 9 (final sum
     0.66) and three at place 10 (final sum 0.68); at place 8 the next
     candidate would overshoot the budget, so no row is added. With a
     chart this churny, the tail beyond the top 12 at
     a place is mostly noise: capping at 12 (down from 15) and leaving
     at least 30% of each place unnamed (up from 20%) puts the mass
     where it is earned, on the likeliest games plus the newcomers the
     unnamed share covers.

The parameters were tuned by walk-forward backtesting on the public
archive (github.com/Kenny-Dhub/roblox-snapshots): five cutoffs, each
forecasting the next 72 h, plus the kit's practice window. The chosen
set scores a mean -36.82 over the five backtest windows and -34.67 on
the practice window (house model -37.25, house baseline -43.51).
Revision (post-open): MAX_NAMED 15 -> 12 and NEW_MASS 0.20 -> 0.30,
each verified separately on the same five windows (MAX_NAMED=12 alone:
-37.01; NEW_MASS=0.30 alone: -37.12; combined: -36.91, with the
practice window improving -35.21 -> -34.59). The tail extension above
was added in the final consolidation pass: it moves the mean to -36.82
(the practice window moves -34.59 -> -34.67, inside the noise band of
a 10-read window; the gain is concentrated in two backtest windows,
+0.5 and +0.4, against small losses in the other three, and the result
is identical for extension caps of 20, 30 and 40, so it is not a
cap-tuning artifact). A slope-shrinkage variant (per-game momentum
shrunk toward the rank-bin mean drift) was prototyped and rejected: it
scored worse (-37.25 at shrink weight 0.7).

Python 3.13, standard library only. No network, no randomness, no
wall-clock reads: two runs on the same history write the same forecast
byte for byte. Runs in well under the 300 s / 1 CPU / 1024 MB limits.

Input, output and edge cases (what the code below actually does):
  * Input: /input/history.ndjson, one JSON object per line, each a
    chart read with an "at" timestamp and an "up_and_coming" list of
    {"universe_id", "rank"} rows in chart order; /input/task.json
    carries "places" (10), the scoring "window" (its "from" start and
    "slots", 72 hourly slots) and the output filename.
  * Per game the reads become a (time, rank) series; a game seen once
    has no trend to fit, so its slope is 0 and its forecast is a flat
    projection widened by diffusion only.
  * For every future slot the game is a normal variable centred on
    the decayed momentum projection, clipped to ranks 1..105, with
    spread SIG0 plus a rank-dependent hourly volatility that grows
    with the square root of the hours since its last read; the
    place probabilities are the normal mass between half-rank
    boundaries, renormalised over ranks 1..105, then averaged over
    all 72 slots, so one static forecast stands for the whole
    scoring window.
  * The geometric prior is taken over the LAST read only: a game
    that dropped off the chart gets no prior mass, it keeps only its
    own history. The age factor needs details.created from a
    house-recorder read; without it no scaling is applied at all.
  * Output: one JSON object, {"probabilities": [{"universe_id",
    "place", "p"}] with "p" rounded to six decimals, "top10":
    [universe_id, ...]}. The top-10 ordering is by total named
    mass across the ten places, ties broken by expected rank, so
    the ordering can differ from any single place's ranking.
  * Rounding: probabilities are rounded only at the very end, after
    the budget scaling; the internal distributions keep full
    precision, so the rounding never moves mass between games.
  * Runtime: the whole history is read once and the forecast is
    built in a single pass over games and slots; there is no
    caching between runs and no state on disk, so a second run
    starts from the same inputs and reaches the same numbers.
  * The 105 cap: projections are clipped to ranks 1..105 and the
    place probabilities are renormalised over that band, so no
    probability mass ever leaks outside the places the forecast
    can name; a game projected far below the chart still keeps a
    thin, honest tail over the bottom places.
"""

import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned parameters (see module docstring) ---
TAU_REC = 12.0        # hours: recency half-life for the trend fit
TAU_V = 36.0          # hours: momentum decay time
SIG0 = 2.0            # ranks: base rank spread at the last observation
SIG_H_SCALE = 1.5     # multiplier on the measured per-hour rank volatility
NEW_MASS = 0.30       # minimum share of each place left unnamed (new entrants)
MAX_NAMED = 12        # games named at each place
MAX_NAMED_TAIL = 30   # ...unless the top 12 sum below the named budget (see
                      # forecast): then further candidates are named, up to
                      # this many, until the budget is filled
MIX_GEO = 0.4         # blend weight of the geometric rank prior
GEO_A = 1.0           # geometric prior exponent: p ~ rank^-GEO_A
AGE_FACTOR = 0.7      # mass scale for universes older than AGE_THR_DAYS
AGE_THR_DAYS = 28.0
MIN_P = 0.0004        # do not emit rows below this probability
SLOPE_CLAMP = 0.6     # ranks/hour

# Per-hour rank volatility by rank bin, measured from the archive
# (standard deviation of the hourly rank change).
VOL = [(1, 3, 0.19), (4, 10, 0.37), (11, 20, 0.66),
       (21, 40, 1.08), (41, 70, 1.34), (71, 200, 1.46)]


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def vol_at(rank):
    for lo, hi, v in VOL:
        if lo <= rank <= hi:
            return v
    return 1.46


def load_history(input_dir):
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_task(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
        return json.load(f)


def latest_created(hist):
    """universe_id -> created timestamp, from the latest details that carry it.

    Archive reads carry no details; house-recorder reads do. The details
    value may be a dict keyed by universe id or a list of game objects.
    """
    created = {}
    for read in hist:
        det = read.get("details")
        if not det:
            continue
        if isinstance(det, dict):
            items = det.items()
        else:
            items = ((d.get("universe_id"), d) for d in det
                     if isinstance(d, dict))
        for key, val in items:
            if key is None or not isinstance(val, dict):
                continue
            if not val.get("created"):
                continue
            try:
                created[int(key)] = val["created"]
            except (TypeError, ValueError):
                continue
    return created


def forecast(hist, task):
    places = int(task.get("places", 10))
    window = task["window"]
    win_from = parse_ts(window["from"])
    slots = int(window.get("slots", 72))

    t0 = parse_ts(hist[0]["at"])
    series = {}
    for read in hist:
        t = (parse_ts(read["at"]) - t0).total_seconds() / 3600.0
        for row in read["up_and_coming"]:
            series.setdefault(row["universe_id"], []).append((t, row["rank"]))
    created = latest_created(hist)

    slot_hours = [(win_from - t0).total_seconds() / 3600.0 + h
                  for h in range(slots)]
    nslots = len(slot_hours)

    last_rank = {r["universe_id"]: r["rank"]
                 for r in hist[-1]["up_and_coming"]}
    geo_tot = sum(r ** (-GEO_A) for r in last_rank.values())
    geo_w = {uid: (r ** (-GEO_A)) / geo_tot for uid, r in last_rank.items()}

    # P[uid][k]: average probability that uid holds place k+1 over the window.
    P = {}
    for uid, obs in series.items():
        t_end = obs[-1][0]
        r0 = obs[-1][1]
        ws = [math.exp(-(t_end - t) / TAU_REC) for t, _ in obs]
        sw = sum(ws)
        mt = sum(w * t for w, (t, _) in zip(ws, obs)) / sw
        mr = sum(w * r for w, (_, r) in zip(ws, obs)) / sw
        num = sum(w * (t - mt) * (r - mr) for w, (t, r) in zip(ws, obs))
        den = sum(w * (t - mt) ** 2 for w, (t, _) in zip(ws, obs))
        slope = num / den if den > 1e-9 else 0.0
        slope = max(-SLOPE_CLAMP, min(SLOPE_CLAMP, slope))
        af = 1.0
        c = created.get(uid)
        if c:
            age_days = (win_from - parse_ts(c)).total_seconds() / 86400.0
            if age_days > AGE_THR_DAYS:
                af = AGE_FACTOR
        dist = [0.0] * places
        for s in slot_hours:
            dt = max(0.0, s - t_end)
            mu = r0 + slope * TAU_V * (1.0 - math.exp(-dt / TAU_V))
            mu = max(1.0, min(105.0, mu))
            sig = SIG0 + SIG_H_SCALE * vol_at(r0) * math.sqrt(dt)
            tot = phi((105.5 - mu) / sig) - phi((0.5 - mu) / sig)
            for k in range(1, places + 1):
                m = ((phi((k + 0.5 - mu) / sig) - phi((k - 0.5 - mu) / sig))
                     / tot if tot > 1e-12 else 0.0)
                dist[k - 1] += m / nslots
        g = geo_w.get(uid, 0.0)
        P[uid] = [((1.0 - MIX_GEO) * d + MIX_GEO * g) * af for d in dist]

    probabilities = []
    cand_mass = {}
    for k in range(places):
        cands = sorted(((P[uid][k], uid) for uid in P), reverse=True)
        # The named rows may sum to at most 1 - NEW_MASS; if the top
        # MAX_NAMED already exceed that budget they are scaled down to
        # it. If they sum below it (the tail places, where the model
        # spreads thin), the next candidates are named too, up to
        # MAX_NAMED_TAIL, until the budget is filled or the next
        # candidate would overshoot it: a candidate whose model
        # probability is well above the unnamed share per game
        # (~0.004) earns its row, and leaving the mass unnamed would
        # spread it thin over ~88 games instead.
        budget = 1.0 - NEW_MASS
        rows = []
        total = 0.0
        for pr, uid in cands[:MAX_NAMED]:
            if pr < MIN_P:
                break
            rows.append((uid, pr))
            total += pr
        if total < budget:
            for pr, uid in cands[MAX_NAMED:MAX_NAMED_TAIL]:
                if pr < MIN_P or total + pr > budget:
                    break
                rows.append((uid, pr))
                total += pr
        if total > budget:
            f = budget / total
            rows = [(uid, pr * f) for uid, pr in rows]
        for uid, pr in rows:
            probabilities.append({"universe_id": uid, "place": k + 1,
                                  "p": round(pr, 6)})
            cand_mass[uid] = cand_mass.get(uid, 0.0) + pr
    exprank = {uid: sum((k + 1) * P[uid][k] for k in range(places))
               for uid in P}
    top10 = sorted(P.keys(),
                   key=lambda u: (-cand_mass.get(u, 0.0), exprank[u]))[:places]
    return {"probabilities": probabilities, "top10": top10}


def main():
    task = load_task(INPUT_DIR)
    hist = load_history(INPUT_DIR)
    result = forecast(hist, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
