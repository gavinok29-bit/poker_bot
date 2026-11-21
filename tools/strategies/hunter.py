"""
HUNTER (sparring partner): built to beat cycler, by attacking the weaknesses of
an adaptive mode-switching bot.

1. Reads the mode from bet sizes. cycler's modes each bet from their own sizes:
       exploit      1/3, 2/3, 1 pot
       pressure     1/2, 3/4, 1 1/4 pot     (bluffs more, expects folds)
       value        2/3, 1, 1 1/2 pot       (never bluffs)
       bluff_catch  1/3, 2/3 pot            (calls light, never bluffs)
   It records the opponent's lead bets (first bet on a street), snaps each to
   the nearest of those sizes, and reads the mode: a size only one mode uses
   among the last three bets (counting the one it's facing now) decides it at
   once; otherwise the mode whose set fits the last six best. Then it
   best-responds:
       pressure     call down lighter, trap strong hands, bluff less
       value        fold medium hands to its bets, bluff it more
       bluff_catch  never bluff, value-bet thin and big, fold to its bets
2. Slow reads. cycler's opponent stats add up over the whole match, so early
   behaviour sticks. For the first 18 hands hunter folds more, so cycler reads
   it as a folder; after that it stops folding to cycler's bets.
3. Chip-lead swings. cycler plays tighter when ahead and looser when behind,
   so hunter bluffs more when cycler leads and calls lighter when it trails.

Nash preflop. When nothing points anywhere, it plays the Nash baseline.
Against a bot whose sizes don't give its mode away, attack 1 finds nothing.
"""
"""(end of description)"""

MODE_SIZES = {
    "exploit": (0.33, 0.66, 1.0),
    "pressure": (0.5, 0.75, 1.25),
    "value": (0.66, 1.0, 1.5),
    "bluff_catch": (0.33, 0.66),
}
ALL_SIZES = sorted({s for v in MODE_SIZES.values() for s in v})
UNIQUE = {0.5: "pressure", 0.75: "pressure", 1.25: "pressure", 1.5: "value"}   # sizes only one mode uses
BAIT_HANDS = 18


def _snap(frac):
    best = min(ALL_SIZES, key=lambda s: abs(s - frac))
    return best if abs(best - frac) <= 0.05 else None


def _facing_lead(gs, c):
    """The pot fraction of a lead bet we're facing right now, if any."""
    seq = [e for e in c.log if e["street"] == c.street]
    aggr = [e for e in seq if _aggressive(e, seq)]
    if len(aggr) == 1 and aggr[0]["bot_id"] != c.me and aggr[0]["action"] == "raise" \
            and gs["your_bet_this_street"] == 0:
        owed = gs["amount_owed"]
        return owed / max(gs["pot"] - owed, 1)
    return None


def read_mode(gs, c):
    sizes = list(STATS.lead_sizes[-6:])
    now = _facing_lead(gs, c)
    if now is not None:
        sizes.append(now)
    snapped = [s for s in (_snap(f) for f in sizes) if s is not None]
    # A size only one mode uses, among the last 3 lead bets, settles it at once:
    # modes change mid-match, so the newest evidence counts most.
    for s in reversed(snapped[-3:]):
        if s in UNIQUE:
            return UNIQUE[s], snapped
    if len(snapped) < 2:
        return None, snapped
    score = {m: sum(1 if s in v else -2 for s in snapped) for m, v in MODE_SIZES.items()}
    best = max(score, key=score.get)
    distinctive = {"pressure": {0.5, 0.75, 1.25}, "value": {1.5}}.get(best)
    if distinctive and not (set(snapped) & distinctive):
        return None, snapped                        # only shared sizes: not enough to tell
    if best == "bluff_catch" and 1.0 in snapped:
        best = "exploit"
    return (best if score[best] > 0 else None), snapped


RESPONSES = {
    "pressure":    dict(call_shift=-0.12, bluff_scale=0.3, trap_thr=0.75, trap_p=0.5),
    "value":       dict(call_shift=0.12, bluff_scale=1.6),
    "bluff_catch": dict(call_shift=0.10, bluff_scale=0.0, value_thr=0.58, size=0.9, raise_thr=0.9),
}


def _my_share(gs, c):
    mine = gs["your_stack"] + gs["your_bet_this_street"]
    theirs = c.opp_max
    return mine / max(mine + theirs, 1)


def _postflop(gs, c):
    mode, snapped = read_mode(gs, c)
    kw = dict(RESPONSES.get(mode, {}))
    phase = "bait" if gs["hand_num"] < BAIT_HANDS else "punish"
    if mode is None:
        kw["call_shift"] = 0.08 if phase == "bait" else -0.06
        kw["bluff_scale"] = 0.6 if phase == "bait" else 1.0
    share = _my_share(gs, c)
    if share < 0.45:                                # they lead, so they play tight
        kw["bluff_scale"] = kw.get("bluff_scale", 1.0) * 1.5
    elif share > 0.55:                              # they trail, so they play loose
        kw["call_shift"] = kw.get("call_shift", 0.0) - 0.05
    dbg = {"read": mode, "sizes": snapped[-6:], "phase": phase, "share": round(share, 2)}
    return baseline_postflop(gs, c, dbg=dbg, **kw)


def decide(game_state: dict) -> dict:
    return run(game_state, gto_preflop, _postflop)
