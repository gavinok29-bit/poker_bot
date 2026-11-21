#!/usr/bin/env python3
"""Track one bot's win rate against another over many matches.

Plays the tournament's format: duplicate pairs of freezeout matches (same cards,
seats swapped). Winning both matches wins the pairing, one each is a draw.

    python3 tools/winrate.py my_bots/monte_carlo my_bots/nash                # 20 pairs
    python3 tools/winrate.py my_bots/monte_carlo my_bots/nash --pairs 100 --jobs 4
    python3 tools/winrate.py my_bots/counter_nash bots/caller --seed 1      # reproducible
    python3 tools/winrate.py A B --mode fixed --hands 500                   # chips won instead
    python3 tools/winrate.py A B --time-limit-ms 200                        # a stricter clock

Every decision runs on the clock: the tournament's 2,000 ms unless you pass
--time-limit-ms. A decision over the limit is a timeout, and the bot folds that
hand, exactly as in the tournament. The summary lists each bot's timeouts,
errors and slowest decision.

Run it from the repo root. A running tally prints after every pair.
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if os.name == "nt":  # Python on Windows needs SYSTEMROOT in the bot processes
    os.environ["BOT_ENV_ALLOW"] = ",".join(filter(None, [os.environ.get("BOT_ENV_ALLOW"), "SYSTEMROOT"]))


def _name(path):
    p = Path(path)
    return p.resolve().parent.name if p.name == "bot.py" else (p.stem if p.suffix else p.resolve().name)


def play_one(a_path, b_path, a_name, b_name, seed, args):
    """One match; returns the chip delta for each bot, hands played and problems."""
    import play
    from pokercore import structures
    from pokercore.seats import BotSeat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match

    limit = getattr(args, "time_limit_ms", None)          # None = the tournament's 2,000 ms
    bots = [play.Bot(a_name, a_path), play.Bot(b_name, b_path)]
    config = MatchConfig(mode=args.mode, hands=args.hands, starting_stack=args.stack,
                         structure=structures.parse(args.blinds, args.stack), seed=seed,
                         time_limit_ms=limit)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), time_limit_ms=limit, use_docker=args.docker)
             for b in bots]
    result = play_match(seats, config, "local", play.Log())
    delta = {r["bot"]: r["delta"] for r in result["results"]}
    problems = {r["bot"]: {"timeouts": r.get("timeouts") or 0, "errors": r.get("errors") or 0,
                           "max_ms": r.get("max_think_ms") or 0.0} for r in result["results"]}
    return delta, result["hands"], problems


def play_pair(a_path, b_path, a_name, b_name, seed, args):
    """Duplicate pair: same seed, seats swapped."""
    d1, h1, p1 = play_one(a_path, b_path, a_name, b_name, seed, args)
    d2, h2, p2 = play_one(b_path, a_path, b_name, a_name, seed, args)
    return seed, [d1[a_name], d2[a_name]], h1 + h2, {k: _merge(p1.get(k), p2.get(k)) for k in (a_name, b_name)}


def _merge(*ps):
    out = {"timeouts": 0, "errors": 0, "max_ms": 0.0}
    for p in ps:
        if p:
            out["timeouts"] += p["timeouts"]
            out["errors"] += p["errors"]
            out["max_ms"] = max(out["max_ms"], p["max_ms"])
    return out


def ci95(wins, n):
    """Wilson 95% interval for a win rate."""
    if n == 0:
        return 0.0, 1.0
    p, z = wins / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bot_a")
    p.add_argument("bot_b")
    p.add_argument("--pairs", type=int, default=20, help="duplicate pairs to play (2 matches each)")
    p.add_argument("--seed", type=int, default=None, help="first seed (default: random); pair i uses seed+i")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="pairs in parallel")
    p.add_argument("--mode", default="freezeout", choices=["freezeout", "fixed", "cash"])
    p.add_argument("--hands", type=int, default=2000, help="hands per match (fixed/cash) or safety cap")
    p.add_argument("--blinds", default="standard")
    p.add_argument("--stack", type=int, default=10_000)
    p.add_argument("--docker", action="store_true")
    p.add_argument("--time-limit-ms", type=int, default=None,
                   help="per-decision clock; over it the bot times out and folds (default: 2000, the tournament's)")
    args = p.parse_args()

    os.chdir(ROOT)
    a_name, b_name = _name(args.bot_a), _name(args.bot_b)
    if a_name == b_name:
        b_name += "_2"
    seed0 = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big") % 1_000_000_000
    print(f"{a_name} vs {b_name}: {args.pairs} duplicate pairs, {args.mode}, blinds {args.blinds}, "
          f"first seed {seed0}, {args.jobs} jobs, clock {args.time_limit_ms or 2000} ms\n")

    match_w = match_l = 0
    pair_w = pair_d = pair_l = 0
    chips, hands, probs = 0, 0, {a_name: _merge(), b_name: _merge()}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(play_pair, args.bot_a, args.bot_b, a_name, b_name, seed0 + i, args)
                for i in range(args.pairs)]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                seed, deltas, h, pr = f.result()
            except Exception as e:
                print(f"pair failed: {e!r}")
                continue
            won = sum(1 for d in deltas if d > 0)
            lost = sum(1 for d in deltas if d < 0)
            match_w += won
            match_l += lost
            if won == 2:
                pair_w += 1
            elif lost == 2:
                pair_l += 1
            else:
                pair_d += 1
            chips += sum(deltas)
            hands += h
            for k in probs:
                probs[k] = _merge(probs[k], pr.get(k))
            played = match_w + match_l
            rate = match_w / played if played else 0
            print(f"[{n:>3}/{args.pairs}] seed {seed}: {' / '.join(f'{d:+,}' for d in deltas):>21}  |  "
                  f"matches {match_w}-{match_l} ({rate:.0%})  pairs W{pair_w} D{pair_d} L{pair_l}", flush=True)

    played = match_w + match_l
    lo, hi = ci95(match_w, played)
    pairs = pair_w + pair_d + pair_l
    print(f"\n{'=' * 64}\n{a_name} vs {b_name}   ({time.time() - t0:.0f}s, {hands:,} hands)")
    print(f"  match win rate : {match_w}/{played} = {match_w / max(played, 1):.1%}   (95% CI {lo:.0%}-{hi:.0%})")
    print(f"  pairings       : {pair_w} won, {pair_d} drawn, {pair_l} lost"
          + (f"   (score {(pair_w + 0.5 * pair_d) / pairs:.1%})" if pairs else ""))
    if args.mode != "freezeout":
        print(f"  chips won      : {chips:+,} total, {chips / max(hands, 1) * 100:+.0f} per 100 hands")
    else:
        print(f"  avg match      : {hands / max(2 * pairs, 1):.0f} hands")
    for k, v in probs.items():
        print(f"  {k:<15}: {v['timeouts']} timeouts, {v['errors']} errors, slowest decision {v['max_ms']:.0f} ms")


if __name__ == "__main__":
    main()
