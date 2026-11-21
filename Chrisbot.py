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

P = dict(open_eq=0.47, open_big=0.62, pre_shave_b=0.02, pre_shave_s=0.05, post_shave_b=0.06, post_shave_s=0.07,
         call_margin=0.04, short_margin=0.02, v_hi=0.78, v_mid=0.62, v_thin=0.50, thin_freq=0.4,
         bluff=0.10, push_eq=0.52, exploit_gain=0.4, reraise=0.72, adapt=0, reraise_pct=0.0, maniac_n=10, maniac_margin=-1.0, maniac_aggr=0.6, sd_learn=0, sd_gain=2.4, sd_prior=3, big_extra=0.0, big_frac=0.5, range=1, range_b=0.15, range_s=0.35, range_cap=0.7, pre_range=0.1, win=40, passive_shave=0.04, mid_aggr=0.40, mid_keep=0.5, station_thin=0.7, fold_win=14)
# Settings baked in: v1-logic baseline (best frozen result). No params.json is read.
P.update({"range": 0})
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


# ---------------------------------------------- range-weighted equity (heuristic) ----
_PRE = None
def _pre_score(a, b):
    """Rough preflop hand strength of two eval7 cards (higher is better). Heuristic only."""
    r1, r2 = sorted([a.rank, b.rank], reverse=True)
    sc = r1 + 0.5 * r2
    if r1 == r2:
        sc += 10
    if a.suit == b.suit:
        sc += 2
    if r1 - r2 == 1:
        sc += 1
    return sc

def _pre_table():
    global _PRE
    if _PRE is None:
        sc = []
        for i in range(52):
            for j in range(i + 1, 52):
                sc.append(_pre_score(_FULL_DECK[i], _FULL_DECK[j]))
        sc.sort()
        _PRE = sc
    return _PRE

_SD = {"seen": set(), "sum": 0.0, "n": 0}
def update_showdowns(state):
    """Online learning from public showdowns: when the opponent raised/shoved in a hand that
    reached showdown, record the preflop-heuristic percentile of the cards it showed.
    Only hands that reached showdown are visible (selection bias: hands it folded are unseen)."""
    import bisect
    me = state["your_bot_id"]
    log = state.get("match_action_log") or []
    aggr_hands = set(e.get("hand_num") for e in log
                     if e.get("bot_id") != me and e.get("action") in ("raise", "all_in"))
    tbl = _pre_table()
    for e in log:
        if e.get("action") == "show" and e.get("bot_id") != me and e.get("cards"):
            hn = e.get("hand_num")
            if hn in _SD["seen"]:
                continue
            _SD["seen"].add(hn)
            if hn in aggr_hands:
                a, b = (eval7.Card(c) for c in e["cards"])
                _SD["sum"] += bisect.bisect_left(tbl, _pre_score(a, b)) / len(tbl)
                _SD["n"] += 1

def showdown_multiplier():
    q = (_SD["sum"] + 0.5 * P["sd_prior"]) / (_SD["n"] + P["sd_prior"])
    return max(0.4, min(1.6, 1.0 + (q - 0.5) * P["sd_gain"]))

def estimate_range_equity(hole, board, pct, budget=0.4, rng=random):
    """Equity vs opponent hands restricted to above the `pct` percentile of current strength
    (preflop heuristic or eval7 rank on the current board). Rejection sampling, at most 25
    redraws per sample; if the range is that hard to hit we accept the last draw (bias is small
    and bounded). This is a heuristic range assumption, not Bayesian inference."""
    key = (tuple(sorted(hole)), tuple(board), round(pct, 2))
    if key in _equity_cache:
        return _equity_cache[key]
    if pct <= 0.01:
        return estimate_equity(hole, board, budget, rng)
    h = [eval7.Card(c) for c in hole]
    b = [eval7.Card(c) for c in board]
    known = set(map(str, h + b))
    deck = [c for c in _FULL_DECK if str(c) not in known]
    need_board = 5 - len(b)
    if len(b) == 0:
        tbl = _pre_table()
        thr = tbl[min(len(tbl) - 1, int(pct * len(tbl)))]
        strength = lambda o: _pre_score(o[0], o[1])
    else:
        probe = sorted(eval7.evaluate(b + rng.sample(deck, 2)) for _ in range(250))
        thr = probe[min(len(probe) - 1, int(pct * len(probe)))]
        strength = lambda o: eval7.evaluate(list(o) + b)
    start = time.perf_counter()
    score, n = 0.0, 0
    while n < MAX_SAMPLES:
        if n >= MIN_SAMPLES and time.perf_counter() - start > budget:
            break
        for _ in range(25):
            draw = rng.sample(deck, 2 + need_board)
            if strength(draw[:2]) >= thr:
                break
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


def recent_stats(state, win):
    """Windowed view of the opponent using only the public action log:
    (raise_share, call_share, n) over their last `win` voluntary actions, and
    fold-after-our-raise over the last fold_win samples. Raise share is NOT a
    hand-strength signal (we see few showdowns); it only measures aggression."""
    me = state["your_bot_id"]
    acts = []
    resp = []
    last = {}
    for e in state.get("match_action_log") or []:
        a, hn = e.get("action"), e.get("hand_num")
        if a in ("small_blind", "big_blind", "ante", "show"):
            continue
        if e.get("bot_id") == me:
            last[hn] = a in ("raise", "all_in")
        else:
            acts.append(a)
            if last.get(hn):
                resp.append(a == "fold")
                last[hn] = False
    w = acts[-win:]
    n = len(w)
    rs = sum(x in ("raise", "all_in") for x in w) / n if n else 0.0
    cs = sum(x == "call" for x in w) / n if n else 0.0
    fw = resp[-P["fold_win"]:]
    fr = sum(fw) / len(fw) if fw else 0.5
    return rs, cs, n, fr, len(fw)


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
    rs = cs = fr = 0.0
    n_w = nf = 0
    if P["adapt"]:
        rs, cs, n_w, fr, nf = recent_stats(state, int(P["win"]))
        if nf >= 6:
            fold_rate, samples = fr, max(samples, 12)   # recent window overrides lifetime average
        if n_w >= 10:
            aggr, aggr_n = rs, n_w                      # recent aggression instead of lifetime
    exploit = 0.0
    if samples >= 12:
        exploit = max(-0.08, min(0.12, (fold_rate - 0.4) * P["exploit_gain"]))
    bluff_freq = P["bluff"] + exploit
    r = random.random()

    # ---- facing a bet (or the big blind posted) ----
    if owed > 0:
        pot_before = max(pot - owed, 1)
        bet_frac = owed / pot_before
        # Opponent's bet range is stronger than random: shave equity.
        adj = eq - min(0.15, P["post_shave_b"] + P["post_shave_s"] * bet_frac)
        if pre:
            # Blinds are small relative to ranges; avoid over-shaving a min-raise.
            adj = eq - min(0.12, P["pre_shave_b"] + P["pre_shave_s"] * (cur / bb) / 3)
        # A maniac (raises/shoves most of the time) bets random hands: no shave.
        if P["adapt"] and aggr_n >= 10:
            if aggr < 0.15 and aggr_n >= 20 and not pre:
                adj -= P["passive_shave"]      # rarely bets: a bet usually means strength
            elif aggr >= P["mid_aggr"] and aggr < 0.6:
                adj += (eq - adj) * P["mid_keep"]
        is_maniac = aggr_n >= P["maniac_n"] and aggr >= P["maniac_aggr"]
        if aggr_n >= 10 and aggr >= 0.6:
            adj = eq
        elif is_maniac:
            adj = eq
        if P["range"] and not (aggr_n >= 10 and aggr >= 0.6):
            if pre:
                pct = min(0.5, P["pre_range"] + 0.1 * (cur / bb) / 3)
            else:
                pct = min(P["range_cap"], P["range_b"] + P["range_s"] * bet_frac)
            if aggr_n >= 10 and aggr >= P["mid_aggr"]:
                pct *= 0.5
            if P["sd_learn"]:
                update_showdowns(state)
                pct = min(P["range_cap"], pct * showdown_multiplier())
            if mine > 0 and not pre:
                pct = min(0.9, pct + P["reraise_pct"])   # hypothesis: a raise over our own bet is stronger
            adj = estimate_range_equity(state["your_cards"], state["community_cards"], pct)
        need = owed / (pot + owed)

        # Short stack: call off when odds are good.
        if eff_bb <= 12 or owed >= stack:
            m = P["maniac_margin"] if (is_maniac and P["maniac_margin"] >= 0) else P["short_margin"]
            return ({"action": "call"} if adj > need + m else {"action": "fold"})

        # Big hands re-raise for value.
        if adj > P["reraise"] and stack > owed * 2:
            size = cur + (pot + owed) * (0.75 if not pre else 1.0)
            return _raise_to(state, size if not pre else max(size, cur * 3))
        # Occasional bluff re-raise with a weak hand and fold equity.
        if (not pre) and adj < 0.30 and r < bluff_freq * 0.5 and eff_bb > 20:
            return _raise_to(state, cur + (pot + owed) * 0.75)
        extra = P["big_extra"] if (not pre and street in ("turn", "river") and bet_frac >= P["big_frac"]) else 0.0
        if adj > need + P["call_margin"] + extra:
            return {"action": "call"}
        return {"action": "fold"}

    # ---- nobody has bet: we can check or bet ----
    if pre:
        # Heads-up the button (small blind) acts first; opening wide is correct.
        if eff_bb <= 12:
            return {"action": "all_in"} if eq > P["push_eq"] - (12 - eff_bb) * 0.01 else \
                   ({"action": "check"} if state["can_check"] else {"action": "fold"})
        if eq > P["open_eq"]:
            return _raise_to(state, bb * (3 if eq < P["open_big"] else 3.5))
        return {"action": "check"} if state["can_check"] else {"action": "call"} \
            if owed <= bb / 2 else {"action": "fold"}

    # Postflop value / bluff betting.
    if eq > P["v_hi"]:
        frac = 0.75 if fold_rate < 0.35 and samples >= 12 else 0.6
        return _raise_to(state, mine + pot * frac)
    if eq > P["v_mid"]:
        return _raise_to(state, mine + pot * 0.5)
    thin = P["station_thin"] if (P["adapt"] and nf >= 6 and fr < 0.15) else P["thin_freq"]
    if eq > P["v_thin"]:
        return _raise_to(state, mine + pot * 0.33) if r < thin else {"action": "check"}
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
        act["debug"] = {"sd_n": _SD["n"], "sd_m": round(showdown_multiplier(), 2), "equity": round(eq, 3), "opp_fold": round(fold_rate, 2), "n": samples}
        return act
    except Exception:
        return _safe_fallback(game_state)
