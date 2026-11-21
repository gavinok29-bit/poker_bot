"""
Heads-up No-Limit Hold'em bot, v1. Plain rules plus a Monte Carlo equity
estimate. No AI/LLM calls, no network, no files.

HOW IT DECIDES (read this, you both need to explain it):

1. EQUITY. We deal random opponent hands and random remaining board cards many
   times and count how often we win (ties count half). That number is our
   "equity" (0 to 1). We stop after ~0.5 s so the 2 s limit is never at risk.

2. OPPONENT RANGE ADJUSTMENT. Equity against a RANDOM hand is too optimistic
   when the opponent has bet, because people bet better-than-random hands.
   So when facing a bet we shave a bit off our equity, more for bigger bets.

3. POT ODDS. If we owe `owed` chips to win a pot of `pot + owed`, we break
   even when equity >= owed / (pot + owed). We call when adjusted equity beats
   that by a small safety margin, otherwise fold.

4. BETTING. Strong hands (high equity) bet/raise for value. Some medium hands
   make small bets. A few weak hands bluff at a low, fixed frequency, and
   more often if the opponent has been folding a lot (see opponent tracking).

5. SHORT STACKS. At 12 big blinds or less we stop being fancy: shove or fold
   (preflop), and call off when equity beats the pot odds.

6. OPPONENT TRACKING. We also measure how often the opponent raises/shoves; a
   maniac gets no equity shave (their bets mean little).
    From match_action_log we count how often the opponent
   folds after we raise. Over 12+ samples, a high fold rate => bluff more,
   a low fold rate (calling station) => bluff less, value bet more.

7. SAFETY. Any error falls back to a safe action. Raises are returned as
   TOTAL amounts for the street, never the raise-by amount. We never return
   "check" while owing chips (the engine would treat that as a call).
"""

import random
import time

import eval7

TIME_BUDGET = 0.5      # seconds of equity simulation per decision
MIN_SAMPLES = 150
MAX_SAMPLES = 6000

_FULL_DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
_equity_cache = {}     # (cards, board) -> equity; lasts one match


# ---------------------------------------------------------------- equity ----
def estimate_equity(hole, board, budget=TIME_BUDGET, rng=random):
    """Win probability vs a random hand (ties = half). Monte Carlo with deadline."""
    key = (tuple(sorted(hole)), tuple(board))
    if key in _equity_cache:
        return _equity_cache[key]
    h = [eval7.Card(c) for c in hole]
    b = [eval7.Card(c) for c in board]
    known = set(map(str, h + b))
    deck = [c for c in _FULL_DECK if str(c) not in known]
    need_board = 5 - len(b)
    start = time.perf_counter()
    score, n = 0.0, 0
    while n < MAX_SAMPLES:
        if n >= MIN_SAMPLES and time.perf_counter() - start > budget:
            break
        draw = rng.sample(deck, 2 + need_board)
        opp = draw[:2]
        full = b + draw[2:]
        mine = eval7.evaluate(h + full)
        theirs = eval7.evaluate(opp + full)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        n += 1
    eq = score / n
    _equity_cache[key] = eq
    return eq


# ------------------------------------------------------ opponent tracking ----
def opponent_fold_rate(state):
    """(fold_rate, samples): how often the opponent folds right after we raise."""
    me = state["your_bot_id"]
    folds = total = 0
    last = {}  # hand_num -> True if our latest aggressive action awaits a reply
    for e in state.get("match_action_log") or []:
        a, hn = e.get("action"), e.get("hand_num")
        if a in ("small_blind", "big_blind", "ante", "show"):
            continue
        if e.get("bot_id") == me:
            last[hn] = a in ("raise", "all_in")
        elif last.get(hn):
            total += 1
            folds += a == "fold"
            last[hn] = False
    return (folds / total if total else 0.5), total


def opponent_aggression(state):
    """(share of opponent actions that are raises/all-ins, samples)."""
    me = state["your_bot_id"]
    agg = tot = 0
    for e in state.get("match_action_log") or []:
        a = e.get("action")
        if a in ("small_blind", "big_blind", "ante", "show") or e.get("bot_id") == me:
            continue
        tot += 1
        agg += a in ("raise", "all_in")
    return (agg / tot if tot else 0.0), tot


# ----------------------------------------------------------------- helpers ----
def _raise_to(state, target):
    """Clamp a desired street-total to legal range; all_in if it uses the stack."""
    max_total = state["your_stack"] + state["your_bet_this_street"]
    target = int(max(target, state["min_raise_to"]))
    if target >= max_total * 0.85 or max_total <= state["min_raise_to"]:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}


def _safe_fallback(state):
    if state.get("can_check") or state.get("amount_owed", 0) == 0:
        return {"action": "check"}
    if state["amount_owed"] <= 0.1 * state.get("pot", 0):
        return {"action": "call"}
    return {"action": "fold"}


# ---------------------------------------------------------------- strategy ----
def choose(state, eq, fold_rate, samples, aggr=0.0, aggr_n=0):
    pot, owed = state["pot"], state["amount_owed"]
    bb = state["blinds"]["bb"]
    stack = state["your_stack"]
    street = state["street"]
    pre = street == "preflop"
    cur = state["current_bet"]
    mine = state["your_bet_this_street"]
    eff_bb = stack / bb

    # How exploitable is the opponent? clamp the effect so we never go wild.
    exploit = 0.0
    if samples >= 12:
        exploit = max(-0.08, min(0.12, (fold_rate - 0.4) * 0.4))
    bluff_freq = 0.10 + exploit
    r = random.random()

    # ---- facing a bet (or the big blind posted) ----
    if owed > 0:
        pot_before = max(pot - owed, 1)
        bet_frac = owed / pot_before
        # Opponent's bet range is stronger than random: shave equity.
        adj = eq - min(0.15, 0.06 + 0.07 * bet_frac)
        if pre:
            # Blinds are small relative to ranges; avoid over-shaving a min-raise.
            adj = eq - min(0.12, 0.02 + 0.05 * (cur / bb) / 3)
        # A maniac (raises/shoves most of the time) bets random hands: no shave.
        if aggr_n >= 10 and aggr >= 0.6:
            adj = eq
        need = owed / (pot + owed)

        # Short stack: call off when odds are good.
        if eff_bb <= 12 or owed >= stack:
            return ({"action": "call"} if adj > need + 0.02 else {"action": "fold"})

        # Big hands re-raise for value.
        if adj > 0.72 and stack > owed * 2:
            size = cur + (pot + owed) * (0.75 if not pre else 1.0)
            return _raise_to(state, size if not pre else max(size, cur * 3))
        # Occasional bluff re-raise with a weak hand and fold equity.
        if (not pre) and adj < 0.30 and r < bluff_freq * 0.5 and eff_bb > 20:
            return _raise_to(state, cur + (pot + owed) * 0.75)
        if adj > need + 0.04:
            return {"action": "call"}
        return {"action": "fold"}

    # ---- nobody has bet: we can check or bet ----
    if pre:
        # Heads-up the button (small blind) acts first; opening wide is correct.
        if eff_bb <= 12:
            return {"action": "all_in"} if eq > 0.52 - (12 - eff_bb) * 0.01 else \
                   ({"action": "check"} if state["can_check"] else {"action": "fold"})
        if eq > 0.47:
            return _raise_to(state, bb * (3 if eq < 0.62 else 3.5))
        return {"action": "check"} if state["can_check"] else {"action": "call"} \
            if owed <= bb / 2 else {"action": "fold"}

    # Postflop value / bluff betting.
    if eq > 0.78:
        frac = 0.75 if fold_rate < 0.35 and samples >= 12 else 0.6
        return _raise_to(state, mine + pot * frac)
    if eq > 0.62:
        return _raise_to(state, mine + pot * 0.5)
    if eq > 0.50:
        return _raise_to(state, mine + pot * 0.33) if r < 0.4 else {"action": "check"}
    if eq < 0.35 and street != "river" and r < bluff_freq:
        return _raise_to(state, mine + pot * 0.5)
    if eq < 0.30 and street == "river" and r < bluff_freq * 0.7:
        return _raise_to(state, mine + pot * 0.6)
    return {"action": "check"}


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        estimate_equity(["As", "Kd"], [], budget=0.05)   # load eval7 / warm caches
        return {"action": "check"}
    try:
        eq = estimate_equity(game_state["your_cards"], game_state["community_cards"])
        fold_rate, samples = opponent_fold_rate(game_state)
        aggr, aggr_n = opponent_aggression(game_state)
        act = choose(game_state, eq, fold_rate, samples, aggr, aggr_n)
        # Final guard: never "check" when we owe chips.
        if act["action"] == "check" and game_state["amount_owed"] > 0:
            act = _safe_fallback(game_state)
        act["debug"] = {"equity": round(eq, 3), "opp_fold": round(fold_rate, 2), "n": samples}
        return act
    except Exception:
        return _safe_fallback(game_state)
