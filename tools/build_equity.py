#!/usr/bin/env python3
"""Precompute the 169x169 preflop equity matrix by Monte Carlo simulation.

EQ[i, j] = equity of starting-hand class i against class j (ties count half),
with the classes indexed exactly as in the bots' class_index():
    0..12      pocket pairs 22..AA
    13..90     suited hands      (13 + hi*(hi-1)/2 + lo)
    91..168    offsuit hands     (91 + hi*(hi-1)/2 + lo)

The bots load it from their data/ folder at import time; it is what lets them
solve the heads-up push/fold Nash equilibrium and compute equity against a
range in microseconds.

    python tools/build_equity.py                 # 1000 runouts per matchup
    python tools/build_equity.py --sims 3000     # slower, more precise
"""

import argparse
import itertools
import os
import random
import shutil
import sys
import time
from multiprocessing import Pool

import eval7
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]


def class_index(c1, c2):
    r1, r2 = RANKS.index(c1[0]), RANKS.index(c2[0])
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1
    k = r1 * (r1 - 1) // 2 + r2
    return 13 + k if c1[1] == c2[1] else 91 + k


COMBOS = [[] for _ in range(169)]
for a, b in itertools.combinations(DECK, 2):
    COMBOS[class_index(a, b)].append((a, b))


def matchup(args):
    i, j, sims, seed = args
    rng = random.Random(seed)
    ci, cj = COMBOS[i], COMBOS[j]
    total = 0.0
    done = 0
    while done < sims:
        h1 = rng.choice(ci)
        h2 = rng.choice(cj)
        if h1[0] in h2 or h1[1] in h2:
            continue
        dead = set(h1) | set(h2)
        board = []
        while len(board) < 5:
            c = DECK[rng.randrange(52)]
            if c not in dead:
                dead.add(c)
                board.append(c)
        b = [eval7.Card(c) for c in board]
        v1 = eval7.evaluate([eval7.Card(h1[0]), eval7.Card(h1[1])] + b)
        v2 = eval7.evaluate([eval7.Card(h2[0]), eval7.Card(h2[1])] + b)
        total += 1.0 if v1 > v2 else 0.5 if v1 == v2 else 0.0
        done += 1
    return i, j, total / sims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=1000)
    p.add_argument("--out", default="my_bots/nash/data/equity169.npy")
    p.add_argument("--copy-to", nargs="*", default=["my_bots/monte_carlo/data", "my_bots/counter_nash/data"])
    args = p.parse_args()

    jobs = [(i, j, args.sims, i * 1000 + j) for i in range(169) for j in range(i, 169)]
    eq = np.full((169, 169), 0.5)
    t0 = time.time()
    with Pool() as pool:
        for n, (i, j, e) in enumerate(pool.imap_unordered(matchup, jobs, chunksize=64)):
            eq[i, j] = e
            eq[j, i] = 1.0 - e
            if n % 2000 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    for i in range(169):
        eq[i, i] = 0.5  # mirror matchups are 50% by symmetry
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, eq.astype(np.float32))
    for d in args.copy_to:
        os.makedirs(d, exist_ok=True)
        shutil.copy(args.out, os.path.join(d, "equity169.npy"))
    print(f"wrote {args.out} (+{len(args.copy_to)} copies) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
