"""
PLURIBUS-STYLE: a blueprint strategy with Pluribus's two signature habits.

Pluribus (Meta / Carnegie Mellon, 2019) played a self-play "blueprint" and
refined it with a short look-ahead. What stood out to the pros who played it:
  * Donk betting. Having called on one street, it often LEADS the next one out
    of position instead of checking to the aggressor. Humans rarely do this.
  * Heavily mixed bet sizes. It picks among many sizes (a third of the pot up
    to 1.5x pot), so the size tells you little about the hand.

This bot uses the Nash bot's equilibrium play as the blueprint, then:
  * mixes its open size preflop (2x / 2.25x / 2.5x / 3x),
  * picks each postflop bet size at random from five sizes (the same mix for
    value hands and bluffs), and scales its bluff frequency to match the chosen
    size (bluffs = B / (P + 2B) of bets), so every size stays balanced,
  * donk-leads out of position after calling the previous street, with value
    hands and a balanced share of bluffs.
"""
"""(end of description)"""

SIZES = [0.33, 0.5, 0.75, 1.0, 1.5]
STREETS = ["preflop", "flop", "turn", "river"]


def _open_size():
    return random.choice([2.0, 2.25, 2.5, 3.0])


def _mixed_size(eq):
    """Random size, independent of the hand: value hands and bluffs use the same
    mix, so the size gives nothing away."""
    return random.choice(SIZES)


def _bluff_scale(size):
    """Balanced bluff share for this size, relative to the blueprint's 0.6 pot."""
    return (size / (1 + 2 * size)) / (0.6 / 2.2)


def _opp_aggressor_last_street(c):
    k = STREETS.index(c.street)
    prev = STREETS[k - 1]
    seq = [e for e in c.log if e["street"] == prev]
    last = None
    for e in seq:
        if _aggressive(e, seq):
            last = e["bot_id"]
    return last is not None and last != c.me


def _preflop(gs, c):
    return gto_preflop(gs, c, open_size=_open_size)


def _postflop(gs, c):
    res = postflop_equity(gs, c)
    eq = res.eq
    first_on_street = not any(e["street"] == c.street for e in c.log)
    # donk lead: out of position, first to act, opponent had the initiative
    if gs["can_check"] and not c.button and first_on_street and _opp_aggressor_last_street(c):
        s = random.choice([0.33, 0.5, 0.75])
        r = random.random()
        if eq > 0.62 and r < 0.55:
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {"eq": round(eq, 3), "donk": s}}
        if eq < 0.35 and r < 0.55 * _bluff_scale(s) * 0.6:
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)),
                    "debug": {"eq": round(eq, 3), "donk": s, "bluff": 1}}
        return {"action": "check", "debug": {"eq": round(eq, 3)}}
    s = _mixed_size(eq)
    return baseline_postflop(gs, c, res=res, size=s, bluff_scale=_bluff_scale(s), dbg={"size": s})


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
