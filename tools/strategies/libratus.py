"""
LIBRATUS-STYLE: polarised overbets, constant mixing, and nightly gap-filling.

Libratus (Carnegie Mellon, 2017) beat four heads-up pros. It was known for
  * big overbets (more than the pot) with a polarised range: the nuts and
    bluffs, nothing in between,
  * a "perfectly executed mixed strategy": the same hand played several ways,
  * patching itself: each night it found the bet sizes the humans had been
    probing it with and solved those spots properly.

This bot:
  * Checked to, it plays a polarised mix: very strong hands overbet
    (1.25x-2x pot, randomly), medium-strong hands sometimes bet half pot and
    sometimes check, and weak hands bluff at the balanced rate for an overbet
    (B / (P + 2B), about 40% of bets at 2x pot). Everything is a random mix.
  * Facing a bet, it defends close to minimum defence frequency, and fills
    gaps: it tracks how often it has folded to each size bucket (small,
    pot-ish, overbet). If the opponent has found a size it folds to too often
    (more than B / (P + B) + 10%), it widens its calling range for that size.
"""
"""(end of description)"""


def _preflop(gs, c):
    return gto_preflop(gs, c)


def _size_bucket(frac):
    return "small" if frac < 0.55 else "big" if frac <= 1.05 else "over"


def _postflop(gs, c):
    res = postflop_equity(gs, c)
    eq = res.eq
    river = c.street == "river"
    r = random.random()
    dbg = {"eq": round(eq, 3)}
    if gs["can_check"]:
        if c.eff / max(gs["pot"], 1) < 1.2 and eq > 0.7:
            return {"action": "all_in", "debug": dbg}
        if eq > 0.85:
            if r < 0.85:
                s = random.choice([1.25, 1.5, 2.0])
                return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {**dbg, "overbet": s}}
            return {"action": "check", "debug": {**dbg, "trap": 1}}
        if eq > 0.65:
            if r < 0.45:
                return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 0.5)), "debug": {**dbg, "size": 0.5}}
            return {"action": "check", "debug": dbg}
        weak = eq < 0.28 or (not river and eq < 0.42)
        if weak and r < (0.28 if river else 0.22):
            s = random.choice([1.25, 1.5, 2.0])
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {**dbg, "overbet": s, "bluff": 1}}
        return {"action": "check", "debug": dbg}

    owed, pot = gs["amount_owed"], gs["pot"]
    before = max(pot - owed, 1)
    frac = owed / before
    bucket = _size_bucket(frac)
    fold_ok = owed / (before + owed)                     # most we may fold (1 - MDF)
    our_fold = STATS.we_fold_by_size[bucket].get(prior=fold_ok, k=4)
    shift = 0.0
    if STATS.we_fold_by_size[bucket].n >= 5 and our_fold > fold_ok + 0.10:
        shift = -0.07                                    # gap found: defend this size wider
        dbg["gap_fill"] = bucket
    if c.can_raise and eq > 0.85 and r < 0.6:
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, random.choice([1.0, 1.5]))), "debug": dbg}
    return baseline_postflop(gs, c, res=res, call_shift=shift, raise_thr=0.92, dbg=dbg)


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
