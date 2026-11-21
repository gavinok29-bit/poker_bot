# Release test results

Tested November 21, 2026, in a Python 3.10 local development environment. No Docker runtime was available.

## Checks

- 42 tests passed (original 37, 3 for the acpc_style opponents, 2 for v3), run after adding the new files.
- Official validator passed.
- Original competition engine, runner and kit files are unchanged.
- No remote writes, pushes, merges or submissions were performed.

## Release code identity

`my_bots/heads_up_v2/bot.py` SHA-256:

`1f1964c702ccc617b95e563b748c98718dae222186d4d152c8beb4a8b696d909`

## Standard-stack practice suite

Seeds: 601, 602, 603, 604, 605, 606, 607, 608, 609, 610. Starting chips per player: 10000. Blinds: standard.

| Opponent | Pair wins | Draws | Losses | Invalid |
|---|---:|---:|---:|---:|
| bots/caller | 10 | 0 | 0 | 0 |
| bots/template | 9 | 1 | 0 | 0 |
| opponents/always_fold | 10 | 0 | 0 | 0 |
| opponents/always_shove | 3 | 6 | 1 | 0 |
| opponents/tight_value | 5 | 5 | 0 | 0 |
| opponents/tight_value_nit | 7 | 3 | 0 | 0 |
| opponents/loose_aggressive | 6 | 4 | 0 | 0 |
| opponents/loose_aggressive_maniac | 7 | 3 | 0 | 0 |
| opponents/short_stack | 3 | 7 | 0 | 0 |
| opponents/short_stack_20bb | 3 | 7 | 0 | 0 |
| opponents/trapper | 4 | 5 | 1 | 0 |
| opponents/trapper_tight | 4 | 5 | 1 | 0 |
| opponents/mixed | 5 | 5 | 0 | 0 |
| opponents/mixed_wild | 7 | 3 | 0 | 0 |
| opponents/calling_station | 10 | 0 | 0 | 0 |
| opponents/calling_station_sticky | 10 | 0 | 0 | 0 |

Total: 160 pairs / 320 freezeouts. Pair W/D/L/invalid: 103/54/3/0.

Largest recorded decision time across BOTH bots in these local runs: 25.338 ms. This is not a sandbox speed guarantee.

Evidence: `results/release_holdout/report.json` and the matching event logs in the archive.

## Direct v1 comparison

Seeds: 701, 702, 703, 704, 705, 706, 707, 708, 709, 710. Starting chips per player: 10000. Blinds: standard.

| Opponent | Pair wins | Draws | Losses | Invalid |
|---|---:|---:|---:|---:|
| baselines/team_v1 | 1 | 7 | 2 | 0 |

Total: 10 pairs / 20 freezeouts. Pair W/D/L/invalid: 1/7/2/0.

Largest recorded decision time across BOTH bots in these local runs: 101.469 ms. This is not a sandbox speed guarantee.

Evidence: `results/release_v1/report.json` and the matching event logs in the archive.

## 10-big-blind starting-stack stress

Seeds: 801, 802, 803. Starting chips per player: 500. Blinds: standard.

| Opponent | Pair wins | Draws | Losses | Invalid |
|---|---:|---:|---:|---:|
| opponents/short_stack | 1 | 2 | 0 | 0 |
| opponents/always_shove | 0 | 2 | 1 | 0 |
| opponents/trapper | 3 | 0 | 0 | 0 |

Total: 9 pairs / 18 freezeouts. Pair W/D/L/invalid: 4/4/1/0.

Largest recorded decision time across BOTH bots in these local runs: 6.162 ms. This is not a sandbox speed guarantee.

Evidence: `results/release_stress/report.json` and the matching event logs in the archive.

## Self-play integration check

Seeds: 901, 902, 903. Starting chips per player: 10000. Blinds: standard.

| Opponent | Pair wins | Draws | Losses | Invalid |
|---|---:|---:|---:|---:|
| my_bots/heads_up_v2 | 0 | 3 | 0 | 0 |

Total: 3 pairs / 6 freezeouts. Pair W/D/L/invalid: 0/3/0/0.

Largest recorded decision time across BOTH bots in these local runs: 4.92 ms. This is not a sandbox speed guarantee.

Evidence: `results/release_self_play/report.json` and the matching event logs in the archive.

## Interpretation

The final release completed 182 pairs / 364 freezeouts without an invalid run. The standard 16-opponent suite scored 103 wins, 54 draws and 3 losses. Always-shove and both trapping settings each caused a pair loss. The 10bb-start stress test also lost one pair against always-shove.

The direct v1 comparison scored only 1 win, 7 draws and 2 losses for v2. That does NOT establish a strength improvement. It also does not prove that v2 is worse: ten pairs is a small sample, and decisions within a match are not independent statistical trials. Preserve both candidates and use more unseen seeds to choose an entry.

These are synthetic practice opponents. Success against them is not evidence of beating a real tournament field or a GTO strategy. Settings and seeds were frozen before the release suite; the reported release was not retuned in response to its results. Earlier development versions and logs are kept separately from the final release.

A pairing win means both swapped-seat freezeouts were won. One win each is a draw. Incomplete freezeouts, crashes, errors, restarts, failed warmups and visible fallbacks are invalid, never silently counted as draws.

The primary suite and direct v1 comparison used four parallel workers for throughput. Stress and self-play checks used one worker. All remained local-process tests. Official Docker verification is still required.

No persistent learning or data training occurred. Playing these matches did not change bot.py or save trained model weights.

## Archive layout

Release reports and compressed event logs are in `results/release_*`. Older development evidence is grouped under `development_evidence/`, and must not be mixed with release results. An earlier v1 comparison was interrupted by a time limit after nine complete pairs; the tenth was run separately. That old experiment is not the release v1 comparison above.

Each JSON report records the exact seeds, core hash, code hashes (except the earliest smoke report), match summaries and per-run logs. Compressed `.jsonl.gz` logs can be inspected with `gzip -dc FILE.jsonl.gz`. Paths in reports are relative to the project root; append `.gz` for archived event logs.

Official source: https://github.com/FreddyyAndrews/poker-hackathon-kit


## Added later: data-informed opponents

See `IMPROVEMENT_LOOP.md` and `results/acpc_v1_vs_v2_comparison.txt`. On three synthetic acpc_style opponents, team_v1 11/16/3 vs heads_up_v2 4/18/8 (pair W/D/L, seeds 701-710, 30 pairs each). Small, synthetic, not proof. The v1 passive-opponent rows come from a separate complete run (`results/acpc_v1_passive`); an interrupted partial run's passive rows were removed from `results/acpc_v1`.

## Held-out comparison

See `RESULTS_BY_PLAYSTYLE.md`: 19 opponents x 8 seeds (801-808). team_v1 119.5/152, v2 117.5, v3 118.0. No version shown stronger; keep team_v1 as the best-supported bot.
