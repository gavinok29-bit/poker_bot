"""
DEEPSTACK-STYLE: depth-limited look-ahead with a learned-style value function.

DeepStack (University of Alberta, 2017) doesn't solve the whole game ahead of
time. At each decision it searches only a little way ahead, then asks a neural
network "how much is this future spot worth?" instead of playing it out.

This bot does the same thing in miniature:
  * Look-ahead: for every candidate action (fold, check/call, bet or raise to
    half pot, pot, 1.25x pot or all-in) it works out what happens on this
    street: how often the opponent folds (B / (P + B), a balanced defender) and,
    when they call, its equity against the hands that call.
  * Value function: instead of searching later streets, it values the pot that
    reaches the next street with a small hand-written "value net":
        V(eq) = eq adjusted for position and streets left
    (in position you realise more than your raw equity, out of position less,
    and the gap grows with more streets to play).
  * It then picks between the best actions with a soft-max, so close decisions
    are mixed rather than always played the same way.
"""
"""(end of description)"""

STREETS_LEFT = {"flop": 2, "turn": 1, "river": 0}


def _value(eq, ip, streets_left):
    """The 'value network': share of the pot we expect to end up with."""
    edge = (0.07 if ip else -0.06) * streets_left
    strong = (eq - 0.5) * 0.10 * streets_left            # strong hands grow the pot
    return float(min(max(eq + edge * (1 - abs(eq - 0.5)) + strong, 0.0), 1.0))


def _preflop(gs, c):
    return gto_preflop(gs, c)


def _postflop(gs, c):
    owed, pot, cur, mine = gs["amount_owed"], gs["pot"], gs["current_bet"], gs["your_bet_this_street"]
    res = postflop_equity(gs, c)
    eq = res.eq
    left = STREETS_LEFT[c.street]
    ip = c.button
    evs = {}
    # fold / check / call
    if owed > 0:
        evs["fold"] = 0.0
        all_in = owed >= gs["your_stack"]
        evs["call"] = _value(eq, ip, 0 if all_in else left) * (pot + owed) - owed
    else:
        evs["check"] = _value(eq, ip, left) * pot
    # bets and raises
    if c.can_raise:
        for name, frac in (("half", 0.5), ("pot", 1.0), ("over", 1.25), ("jam", None)):
            total = c.my_max if frac is None else bet_pot_frac(gs, c, frac)
            total = min(max(total, gs["min_raise_to"]), c.my_max, c.opp_max)
            if total <= cur:
                continue
            cost = total - mine
            raise_by = total - cur
            p_fold = raise_by / (pot + owed + raise_by) * (0.85 if owed else 1.0)
            final_pot = pot + cost + (total - cur)
            eqc = eq_vs_callers(res, p_fold)
            jam = total >= c.my_max or total >= c.opp_max
            v = _value(eqc, ip, 0 if jam else left)
            evs[name] = p_fold * pot + (1 - p_fold) * (v * final_pot - cost)
    # soft-max over the best few, so near-ties are mixed
    best = max(evs.values())
    temp = max(0.03 * pot, 1.0)
    keys = list(evs)
    w = [math.exp((evs[k] - best) / temp) for k in keys]
    pick = random.choices(keys, weights=w)[0]
    dbg = {"eq": round(eq, 3), "ev": {k: round(v) for k, v in evs.items()}, "pick": pick}
    if pick in ("fold",):
        return {"action": "fold", "debug": dbg}
    if pick in ("check", "call"):
        return {**act_call(gs), "debug": dbg}
    if pick == "jam":
        return {"action": "all_in", "debug": dbg}
    frac = {"half": 0.5, "pot": 1.0, "over": 1.25}[pick]
    return {**act_raise_to(gs, c, bet_pot_frac(gs, c, frac)), "debug": dbg}


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
