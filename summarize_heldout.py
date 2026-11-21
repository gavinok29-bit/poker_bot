#!/usr/bin/env python3
"""Merge several run_bench reports per bot and print a per-opponent W/D/L table (markdown)."""
import glob, json, sys
SC = {'win': 1.0, 'draw': .5, 'loss': 0.0}
def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for p in json.load(open(f))['pairs']:
            out[(p['opponent'], p['seed'])] = p['outcome']
    return out
bots = {'team_v1': 'results/heldout_v1*/report.json', 'heads_up_v2': 'results/heldout_v2/report.json',
        'heads_up_v3': 'results/heldout_v3/report.json'}
data = {b: load(g) for b, g in bots.items()}
opps = sorted({k[0] for d in data.values() for k in d})
print('| Opponent | team_v1 W/D/L | heads_up_v2 W/D/L | heads_up_v3 W/D/L |\n|---|---|---|---|')
tot = {b: [0, 0, 0, 0, 0.0] for b in bots}
for o in opps:
    row = [o.replace('opponents/', '').replace('bots/', 'official ')]
    for b, d in data.items():
        ks = [k for k in d if k[0] == o]
        w, dr, l, inv = (sum(d[k] == x for k in ks) for x in ('win', 'draw', 'loss', 'invalid'))
        row.append(f'{w}/{dr}/{l}' + (f' ({len(ks)} pairs, {inv} invalid)' if len(ks) != 8 or inv else ''))
        t = tot[b]; t[0] += w; t[1] += dr; t[2] += l; t[3] += inv; t[4] += w+.5*dr
    print('| '+' | '.join(row)+' |')
print('| **Total** | '+' | '.join(f'**{t[0]}/{t[1]}/{t[2]}** (score {t[4]} of {t[0]+t[1]+t[2]}, invalid {t[3]})' for t in tot.values())+' |')
