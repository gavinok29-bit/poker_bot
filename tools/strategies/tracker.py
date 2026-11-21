"""
TRACKER: an opponent-modelling bot. It counts what the opponent does and moves
away from balanced play only as far as the evidence supports.

It tracks, with recent hands weighted more (so it notices when they change):
  preflop   how often they open from the small blind, and how often they fold
            to our opens and to our 3-bets
  postflop  how often they fold, call or raise when we bet; what share of
            their actions are bets or raises; how often they c-bet the flop

Every adjustment is (how far the stat is from a balanced opponent) x
(confidence), where confidence = samples / (samples + 10). With 5 samples it
moves a third of the way, with 30 samples three quarters of the way. So a
small, lucky sample can't push it far, and it slides back to balanced play
when the opponent adjusts.

  They fold too much to bets      -> bluff more, bet smaller
  They call too much              -> stop bluffing, value-bet thinner and bigger
  They raise our bets a lot       -> value-bet only strong hands
  Their bets are often bluffs     -> call down lighter, trap strong hands
  Their bets are rarely bluffs    -> fold more marginal hands
  They c-bet nearly every flop    -> float and raise their flop bets
  They fold a lot to our opens    -> open wider (steal)
  They open almost every hand     -> 3-bet them more, wider
  They fold a lot to 3-bets       -> 3-bet bluff more
"""
"""(end of description)"""

K_CONF = 10.0


def conf(rate):
    return rate.n / (rate.n + K_CONF)


def shift(rate, prior, scale, lo, hi):
    """(observed - balanced) x confidence x scale, clamped."""
    v = (rate.get(prior) - prior) * conf(rate) * scale
    return min(max(v, lo), hi)


def _opp_raised_preflop(c):
    last = None
    for e in c.pf:
        if _aggressive(e, c.pf):
            last = e["bot_id"]
    return last is not None and last != c.me


def _preflop(gs, c):
    first_in = not c.pf and c.button
    if c.eff_bb > PUSH_FOLD_BB:
        # steal wider against blinds that fold too much
        if first_in:
            widen = shift(STATS.fold_to_open, 0.35, 1.2, -0.15, 0.22)
            if c.pct < 0.78 + widen:
                return {**act_raise_to(gs, c, 2.5 * c.bb), "debug": {"steal_widen": round(widen, 2)}}
            return {"action": "fold"}
        # 3-bet wider against bots that open too much or fold too much to 3-bets
        if not c.button and c.pf_raises == 1 and c.can_raise and c.opp_is_sb_first(c):
            loose = shift(STATS.opp_open, 0.80, 1.0, 0.0, 0.2) + shift(STATS.fold_to_3bet, 0.45, 0.8, 0.0, 0.25)
            if loose > 0.02 and c.pct < 0.20 + loose:
                return {**act_raise_to(gs, c, gs["current_bet"] * 3.2), "debug": {"3bet_widen": round(loose, 2)}}
    return gto_preflop(gs, c)


def _postflop(gs, c):
    fold = STATS.fold_to_bet
    over_fold = shift(fold, 0.40, 1.0, -0.4, 0.4)                # >0: they fold too much
    raise_up = shift(STATS.raise_vs_bet, 0.10, 1.0, -0.1, 0.3)   # >0: they raise our bets a lot
    bluffy = shift(STATS.aggr, 0.35, 1.0, -0.3, 0.4)             # >0: they bet a lot (bluff a lot)
    cbets = shift(STATS.cbet, 0.60, 1.0, -0.6, 0.4)              # >0: they c-bet too often

    kw = dict(
        bluff_scale=min(max(1 + 5 * over_fold, 0.05), 3.0),
        value_thr=0.68 + 0.3 * min(over_fold, 0) + 0.4 * max(raise_up, 0),   # thinner vs callers
        size=0.5 if over_fold > 0.08 else (0.85 if over_fold < -0.08 else None),
        call_shift=min(max(-0.35 * bluffy, -0.12), 0.06),
        trap_thr=0.9 - 0.4 * max(bluffy, 0),
        trap_p=0.2 + 1.2 * max(bluffy, 0),
        bluff_raise_p=0.06 + 0.3 * max(cbets, 0) * (c.street == "flop" and _opp_raised_preflop(c)),
    )
    kw["value_thr"] = min(max(kw["value_thr"], 0.55), 0.80)
    if c.street == "flop" and _opp_raised_preflop(c) and cbets > 0.05:
        kw["call_shift"] -= 0.3 * cbets                          # float their c-bets
    dbg = {"fold": round(fold.get(0.40), 2), "aggr": round(STATS.aggr.get(0.35), 2),
           "conf": round(conf(fold), 2), "bluff_x": round(kw["bluff_scale"], 2),
           "call_shift": round(kw["call_shift"], 3), "value_thr": round(kw["value_thr"], 2)}
    return baseline_postflop(gs, c, dbg=dbg, **kw)


def decide(game_state: dict) -> dict:
    return run(game_state, _preflop, _postflop)
