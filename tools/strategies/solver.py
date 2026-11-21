"""
SOLVER: plays the patterns that GTO Wizard and PioSolver solutions keep finding.

  * Preflop: small opens (2x-2.5x), wide button range, the big blind defends
    very wide because of the price it gets, and out of position against a
    3-bet or 4-bet it re-raises or folds rather than flatting.
  * Range advantage decides the bet:
      - Dry, high boards that favour the preflop raiser (K-7-2): bet small
        (a third of the pot) with most of the range.
      - Wet boards, or low connected boards that favour the caller: check
        more; when betting, bet big (3/4 to full pot) and polarised (strong
        hands plus draws as bluffs).
  * Nut advantage: on the turn and river, with a near-nut hand as the
    aggressor, overbet 1.5x pot (and bluff at the matching rate).
  * Bet-size maths:
      - Bluffs make up B / (P + 2B) of a river betting range (1 bluff for every
        2 value bets at pot size).
      - Facing a bet it defends close to minimum defence frequency P / (P + B):
        it calls slightly below the bare pot-odds price so it can't be bluffed
        off too often.
"""
"""(end of description)"""


def _preflop(gs, c):
    return gto_preflop(gs, c, open_size=lambda: random.choice([2.0, 2.25, 2.5]),
                       oop_flat=False, realize_oop=0.92)


def _bluff_p(size, value_to_weak=0.6):
    """P(bluff | weak hand) so that bluffs are size/(1+2*size) of all bets."""
    ratio = size / (1 + 2 * size)
    return min(ratio / (1 - ratio) * value_to_weak, 0.9)


def _we_have_initiative(c):
    streets = ["preflop", "flop", "turn", "river"]
    prev = streets[streets.index(c.street) - 1]
    seq = [e for e in c.log if e["street"] == prev]
    last = None
    for e in seq:
        if _aggressive(e, seq):
            last = e["bot_id"]
    return last == c.me


def _postflop(gs, c):
    res = postflop_equity(gs, c)
    eq = res.eq
    river = c.street == "river"
    t = board_texture(gs["community_cards"])
    init = _we_have_initiative(c)
    r = random.random()
    dbg = {"eq": round(eq, 3)}

    if not gs["can_check"]:
        owed = gs["amount_owed"]
        before = max(gs["pot"] - owed, 1)
        dbg["mdf"] = round(before / (before + owed), 2)
        # check-raise bluffs with draws on the flop
        if c.street == "flop" and c.can_raise and 0.30 < eq < 0.45 and t.flush_draw and r < 0.12:
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 0.8)), "debug": {**dbg, "semibluff": 1}}
        return baseline_postflop(gs, c, res=res, call_shift=-0.03, raise_thr=0.85, dbg=dbg)

    if c.eff / max(gs["pot"], 1) < 1.2 and eq > 0.68:
        return {"action": "all_in", "debug": dbg}
    # nut advantage: overbet
    if init and c.street != "flop" and eq > 0.9:
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 1.5)), "debug": {**dbg, "plan": "overbet"}}
    if init and c.street != "flop" and eq < 0.2 and r < _bluff_p(1.5, 0.25):
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 1.5)), "debug": {**dbg, "plan": "overbet", "bluff": 1}}
    # range bet small on boards that favour the raiser
    if c.street == "flop" and init and t.raiser_favoured and not t.wet:
        if eq > 0.45 or r < 0.6:
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 0.33)), "debug": {**dbg, "plan": "range_small"}}
        return {"action": "check", "debug": {**dbg, "plan": "range_small"}}
    # wet / caller-favoured boards: check more, bet big and polarised
    if t.wet or t.caller_favoured or not init:
        s = 1.0 if river else 0.75
        if eq > 0.75:
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {**dbg, "plan": "polar"}}
        draw = not river and 0.30 < eq < 0.48
        if (draw or (river and eq < 0.2)) and r < _bluff_p(s, 0.35):
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {**dbg, "plan": "polar", "bluff": 1}}
        return {"action": "check", "debug": {**dbg, "plan": "polar"}}
    # everything else: two-thirds pot, balanced
    s = 0.66
    if eq > 0.66:
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": dbg}
    if eq < (0.2 if river else 0.42) and r < _bluff_p(s, 0.4):
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {**dbg, "bluff": 1}}
    return {"action": "check", "debug": dbg}


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
