### leak_folder
"""
LEAK_FOLDER (sparring partner): folds too much to bets.

Nash preflop. After the flop it bets its good hands but folds to any bet or
raise unless its equity is above 72%. The bot exploit_folder is built to beat.
"""
"""(end of description)"""


def _postflop(gs, c):
    res = postflop_equity(gs, c)
    if gs["can_check"]:
        return baseline_postflop(gs, c, res=res, bluff_scale=0.0)
    return act_call(gs) if res.eq > 0.72 else {"action": "fold", "debug": {"eq": round(res.eq, 3)}}


def decide(game_state: dict) -> dict:
    return run(game_state, gto_preflop, _postflop)
### leak_station
"""
LEAK_STATION (sparring partner): a calling station.

Nash preflop. After the flop it calls almost everything (any hand with 18%+
equity), only raises the near-nuts, and never bluffs. The bot exploit_station
is built to beat.
"""
"""(end of description)"""


def _postflop(gs, c):
    res = postflop_equity(gs, c)
    if gs["can_check"]:
        return baseline_postflop(gs, c, res=res, bluff_scale=0.0, value_thr=0.72)
    if c.can_raise and res.eq > 0.93:
        return act_raise_to(gs, c, bet_pot_frac(gs, c, 0.7))
    return act_call(gs) if res.eq > 0.18 else {"action": "fold"}


def decide(game_state: dict) -> dict:
    return run(game_state, gto_preflop, _postflop)
### leak_maniac
"""
LEAK_MANIAC (sparring partner): far too aggressive.

Raises 80% of hands preflop when first in. After the flop it bets 75% of the
pot with 75% of hands when checked to, raises 35% of bets whatever it holds,
and calls the rest with any reasonable equity. The bot exploit_maniac is built
to beat.
"""
"""(end of description)"""


def _preflop(gs, c):
    if not c.pf and c.button and c.eff_bb > PUSH_FOLD_BB:
        return act_raise_to(gs, c, 3 * c.bb) if random.random() < 0.8 else {"action": "fold"}
    return gto_preflop(gs, c)


def _postflop(gs, c):
    r = random.random()
    if gs["can_check"]:
        return act_raise_to(gs, c, bet_pot_frac(gs, c, 0.75)) if r < 0.75 else {"action": "check"}
    if c.can_raise and r < 0.35:
        return act_raise_to(gs, c, bet_pot_frac(gs, c, 0.9))
    res = postflop_equity(gs, c)
    need = gs["amount_owed"] / (gs["pot"] + gs["amount_owed"])
    return act_call(gs) if res.eq > need * 0.8 else {"action": "fold"}


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
### leak_sizer
"""
LEAK_SIZER (sparring partner): its bet size gives its hand away.

Nash preflop and Nash defence. When it bets after the flop the size tells you
everything: strong hands (70%+ equity) bet the full pot, medium hands bet a
third of the pot, and its rare bluffs use the small size too. The bot
exploit_sizing is built to beat.
"""
"""(end of description)"""


def _postflop(gs, c):
    res = postflop_equity(gs, c)
    if not gs["can_check"]:
        return baseline_postflop(gs, c, res=res)
    eq = res.eq
    if eq > 0.70:
        return act_raise_to(gs, c, bet_pot_frac(gs, c, 1.0))
    if eq > 0.50 or random.random() < 0.15:
        return act_raise_to(gs, c, bet_pot_frac(gs, c, 0.33))
    return {"action": "check"}


def decide(game_state: dict) -> dict:
    return run(game_state, gto_preflop, _postflop)
### leak_cbettor
"""
LEAK_CBETTOR (sparring partner): continuation-bets every flop.

Opens 90% of hands from the small blind. Whenever it raised preflop it bets
half the pot on the flop, every time, whatever it holds. Facing a raise it
plays honestly (folds its air). The bot exploit_cbet is built to beat.
"""
"""(end of description)"""


def _preflop(gs, c):
    if not c.pf and c.button and c.eff_bb > PUSH_FOLD_BB:
        return act_raise_to(gs, c, 2.5 * c.bb) if c.pct < 0.9 else {"action": "fold"}
    return gto_preflop(gs, c)


def _postflop(gs, c):
    if c.street == "flop" and gs["can_check"] and we_were_pf_aggressor(c):
        return act_raise_to(gs, c, bet_pot_frac(gs, c, 0.5))
    return baseline_postflop(gs, c, bluff_scale=0.5)


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
