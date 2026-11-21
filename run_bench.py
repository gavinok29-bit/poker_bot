"""Duplicate-pair benchmark: for each seed play twice with seats swapped.
usage: python3 run_bench.py my_bots/team_v1 bots/caller 10"""
import subprocess, sys, re
me, opp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
w = d = l = 0
for seed in range(1, n + 1):
    res = []
    for order in ((me, opp), (opp, me)):
        out = subprocess.run(["python3", "play.py", *order, "--seed", str(seed)],
                             capture_output=True, text=True).stdout
        m = re.search(r"^\s*1\.\s+(\S+)", out, re.M)
        res.append(m.group(1) if m else "?")
    mine = sum(1 for r in res if r == me.split("/")[-1])
    w += mine == 2; d += mine == 1; l += mine == 0
print(opp, "pairs won/drawn/lost:", w, d, l)
