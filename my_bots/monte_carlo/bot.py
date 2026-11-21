"""
MONTE CARLO: an expected-value maximiser driven by simulation.

At every decision it deals thousands of random runouts against hands drawn from
the opponent's likely range (what they could hold given how they've played this
hand), and measures its equity. For each legal action (fold, check/call, and
several bet/raise sizes including all-in) it estimates the chips it expects to
win, using:
    EV(raise) = P(fold) * pot + P(call) * (equity vs calling hands * final pot - our chips)
where P(fold) is the share a balanced defender folds (bet / (pot + bet)) and the
calling hands are the strongest part of the simulated range. It then plays the
best one. It uses the same push/fold Nash ranges as the Nash bot to guess what
a short-stacked opponent jams with, but otherwise doesn't model the opponent.
"""

import os
import random
import time

import eval7
import numpy as np

# ================================= TOOLKIT ==================================
# (shared by the nash, monte_carlo and counter_nash bots, each a single file)

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]
CARD = {c: eval7.Card(c) for c in DECK}
POSTS = ("small_blind", "big_blind", "ante", "show")


def class_index(c1, c2):
    """One of the 169 starting-hand classes (pairs, suited, offsuit)."""
    r1, r2 = RANKS.index(c1[0]), RANKS.index(c2[0])
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1
    k = r1 * (r1 - 1) // 2 + r2
    return 13 + k if c1[1] == c2[1] else 91 + k


W = np.zeros(169)  # combos per class: 6 pairs, 4 suited, 12 offsuit
for _a in range(52):
    for _b in range(_a + 1, 52):
        W[class_index(DECK[_a], DECK[_b])] += 1


def _chen_strength():
    """Fallback hand strength if the equity table is missing."""
    s = np.zeros(169)
    for a in range(52):
        for b in range(a + 1, 52):
            c1, c2 = DECK[a], DECK[b]
            r1, r2 = sorted((RANKS.index(c1[0]), RANKS.index(c2[0])), reverse=True)
            v = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10][r1]
            if r1 == r2:
                v = max(v * 2, 5)
            else:
                v += 2 if c1[1] == c2[1] else 0
                gap = r1 - r2 - 1
                v -= [0, 1, 2, 4, 5][min(gap, 4)]
                if gap <= 1 and r1 < 10:
                    v += 1
            s[class_index(c1, c2)] = v
    return (s - s.min()) / (s.max() - s.min())


def _load_equity():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("BOT_DATA_DIR"), os.path.join(here, "data")):
        if d:
            try:
                return np.load(os.path.join(d, "equity169.npy")).astype(np.float64)
            except Exception:
                pass
    h = _chen_strength()
    return np.clip(0.5 + 0.7 * (h[:, None] - h[None, :]), 0.15, 0.85)


EQ = _load_equity()                        # EQ[i, j]: equity of class i vs class j
EQ_RANDOM = EQ @ W / W.sum()               # equity vs a random hand
_order = np.argsort(-EQ_RANDOM)
START = np.zeros(169)                      # share of combos strictly better than class i
_acc = 0.0
for _i in _order:
    START[_i] = _acc
    _acc += W[_i] / W.sum()
END = START + W / W.sum()
PCT = (START + END) / 2                    # 0 = best hand (AA), 1 = worst (32o)


def top_range(f):
    """Per-class weights (0..1) of the best fraction f of all hands."""
    if f >= 1:
        return np.ones(169)
    return np.clip((f - START) / (END - START), 0.0, 1.0)


def eq_vs_range(i, rw):
    v = W * rw
    s = v.sum()
    return float(EQ[i] @ v / s) if s > 0 else 0.5


def range_share(rw):
    return float((W * rw).sum() / W.sum())


# -- heads-up push/fold Nash equilibrium -------------------------------------
_PF_CACHE = {}


def pushfold_nash(stack_bb, ante_bb=0.0, iters=400):
    """(push, call) probabilities per class for the SB-shove / BB-call game with
    effective stack stack_bb. Solved by fictitious play; cached per depth."""
    key = (round(stack_bb * 2) / 2, round(ante_bb * 20) / 20)
    if key in _PF_CACHE:
        return _PF_CACHE[key]
    S, a = max(key[0], 1.0), key[1]
    pot = 2 * S + 2 * a
    push = np.ones(169)
    call = np.full(169, 0.5)
    for t in range(1, iters + 1):
        pw = W * push
        eq_bb = EQ @ pw / max(pw.sum(), 1e-9)                    # BB hand j vs push range
        call_br = (eq_bb * pot - S - a > -(1 + a)).astype(float)
        cw = W * call
        pc = cw.sum() / W.sum()
        eq_sb = EQ @ cw / max(cw.sum(), 1e-9)
        ev_push = (1 - pc) * (1 + a) + pc * (eq_sb * pot - S - a)
        push_br = (ev_push > -(0.5 + a)).astype(float)
        push += (push_br - push) / (t + 1)
        call += (call_br - call) / (t + 1)
    _PF_CACHE[key] = (push, call)
    return push, call


# -- game-state helpers ------------------------------------------------------
class Ctx:
    pass


def context(gs):
    c = Ctx()
    c.me = gs["your_bot_id"]
    seat = gs["seat_to_act"]
    opp = [p for p in gs["players"] if p["seat"] != seat and p.get("state") != "busted"]
    opp = opp[0] if opp else {"stack": 0, "bet_this_street": 0, "bot_id": "?"}
    c.opp_id = opp["bot_id"]
    c.opp_stack = opp["stack"]
    c.opp_bet = opp["bet_this_street"]
    c.button = seat == gs["button_seat"]          # SB preflop, in position after
    c.bb = max(gs["blinds"]["bb"], 1)
    c.ante = gs["blinds"].get("ante", 0) or 0
    c.my_max = gs["your_stack"] + gs["your_bet_this_street"]
    c.opp_max = c.opp_stack + c.opp_bet
    c.eff = min(c.my_max, c.opp_max)
    c.eff_bb = c.eff / c.bb
    c.street = gs["street"]
    c.cls = class_index(*gs["your_cards"])
    c.pct = PCT[c.cls]
    log = [e for e in gs.get("action_log", []) if e["action"] not in POSTS]
    c.log = log
    c.pf = [e for e in log if e["street"] == "preflop"]
    c.pf_raises = sum(1 for e in c.pf if _aggressive(e, c.pf))
    c.opp_pf_raises = sum(1 for e in c.pf if e["bot_id"] != c.me and _aggressive(e, c.pf))
    c.opp_pf_called = any(e["bot_id"] != c.me and e["action"] == "call" for e in c.pf)
    c.opp_limped = bool(c.pf) and c.pf[0]["bot_id"] != c.me and c.pf[0]["action"] == "call"
    c.opp_jammed = any(e["bot_id"] != c.me and e["action"] == "all_in" for e in log)
    post = [e for e in log if e["street"] != "preflop"]
    c.opp_post_aggr = sum(1 for e in post if e["bot_id"] != c.me and e["action"] in ("raise", "all_in"))
    c.opp_post_calls = sum(1 for e in post if e["bot_id"] != c.me and e["action"] == "call")
    c.can_raise = c.opp_stack > 0 and c.my_max > gs["current_bet"]
    return c


def _aggressive(e, seq):
    if e["action"] == "raise":
        return True
    if e["action"] != "all_in":
        return False
    prev = [x.get("amount", 0) for x in seq[:seq.index(e)]
            if x["action"] in ("raise", "all_in") and x["street"] == e["street"]]
    return e.get("amount", 0) > max(prev, default=0)


def opp_preflop_range(c, gs, f_open=0.80, f_iso=0.35, f_3bet=0.15, f_4bet=0.07, f_5bet=0.035,
                      f_call_open=0.65, f_call_3bet=0.30, push_range=None):
    """Opponent's preflop range, from what they did this hand. The f_* are the
    shares of hands they're assumed to take each line with (equilibrium-ish
    defaults; counter_nash replaces them with what it has observed)."""
    raises = []                                  # (who_is_opp, level) in order
    lvl = 0
    for e in c.pf:
        if _aggressive(e, c.pf):
            lvl += 1
            raises.append((e["bot_id"] != c.me, lvl, e["action"] == "all_in"))
    opp_raises = [r for r in raises if r[0]]
    if opp_raises:
        _, L, jam = opp_raises[-1]
        if L == 1:
            f = f_open if c.opp_is_sb_first(c) else f_iso
            if jam and push_range is not None:
                return push_range
        else:
            f = {2: f_3bet, 3: f_4bet}.get(L, f_5bet)
        if jam:
            f = min(f * 1.15, 1.0)
        return top_range(f)
    if c.opp_pf_called:
        if lvl >= 2:
            return np.clip(top_range(f_call_3bet) - 0.7 * top_range(f_4bet), 0, 1)
        if lvl == 1:
            return np.clip(top_range(f_call_open) - 0.7 * top_range(f_3bet), 0, 1)
        return np.clip(top_range(0.9) - 0.6 * top_range(0.2), 0, 1)       # limped
    if not c.button:
        return np.ones(169)                     # SB hasn't acted yet (we are SB... n/a)
    return np.clip(1.0 - 0.8 * top_range(0.25), 0, 1)                       # BB checked option


Ctx.opp_is_sb_first = staticmethod(lambda c: bool(c.pf) and c.pf[0]["bot_id"] != c.me)


# -- Monte Carlo equity against a range, narrowed by the opponent's actions --
class MCResult:
    pass


def mc_equity(hole, board, range_w, n_aggr=0, n_calls=0, bluff_keep=0.3, budget=0.30, max_iters=6000):
    """Equity of `hole` against an opponent whose hands are drawn from range_w
    (169 class weights), reweighted by their postflop bets (n_aggr) and calls
    (n_calls). Returns eq plus per-sample opponent strength and outcome so
    callers can ask "what's my equity against the hands that call?"."""
    t0 = time.time()
    dead = set(hole) | set(board)
    deck = [c for c in DECK if c not in dead]
    dc = [CARD[c] for c in deck]
    hc = [CARD[c] for c in hole]
    bc = [CARD[c] for c in board]
    n = len(deck)
    pa, pb, wts = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            w = range_w[class_index(deck[a], deck[b])]
            if w > 0:
                pa.append(a)
                pb.append(b)
                wts.append(w)
    if not wts:
        return _mc_res(0.5, np.array([0.5]), np.array([0.5]), np.array([1.0]))
    pa, pb, wts = np.array(pa), np.array(pb), np.array(wts, dtype=float)

    if len(board) >= 3:
        vals = np.array([eval7.evaluate([dc[a], dc[b]] + bc) for a, b in zip(pa, pb)])
        strength = np.argsort(np.argsort(vals)) / max(len(vals) - 1, 1)  # 0 weakest .. 1 nuts
        if len(board) < 5:                       # flush draws play like medium-strong hands
            for s in SUITS:
                on_board = sum(1 for c in board if c[1] == s)
                if on_board >= 2:
                    cnt = on_board + (np.array([deck[a][1] == s for a in pa], int)
                                      + np.array([deck[b][1] == s for b in pb], int))
                    strength = np.where(cnt == 4, np.maximum(strength, 0.62), strength)
        for _ in range(n_aggr):
            wts = wts * (bluff_keep + (1 - bluff_keep) / (1 + np.exp(-(strength - 0.55) / 0.08)))
        for _ in range(n_calls):
            wts = wts * (0.25 + 0.75 * np.clip(strength / 0.4, 0, 1))
    else:
        strength = 1.0 - PCT[[class_index(deck[a], deck[b]) for a, b in zip(pa, pb)]]

    need = 5 - len(board)
    if need == 0:                                # river: exact over the whole range
        me = eval7.evaluate(hc + bc)
        ov = np.array([eval7.evaluate([dc[a], dc[b]] + bc) for a, b in zip(pa, pb)])
        out = np.where(me > ov, 1.0, np.where(me == ov, 0.5, 0.0))
        return _mc_res(float(out @ wts / wts.sum()), strength, out, wts)

    probs = wts / wts.sum()
    picks = np.random.choice(len(wts), size=max_iters, p=probs)
    outs = np.empty(max_iters)
    idx = list(range(n))
    done = 0
    for k in range(max_iters):
        i = picks[k]
        a, b = pa[i], pb[i]
        run = []
        for c in random.sample(idx, need + 2):
            if c != a and c != b:
                run.append(dc[c])
                if len(run) == need:
                    break
        full = bc + run
        mv = eval7.evaluate(hc + full)
        ov = eval7.evaluate([dc[a], dc[b]] + full)
        outs[k] = 1.0 if mv > ov else 0.5 if mv == ov else 0.0
        done = k + 1
        if (k & 127) == 127 and time.time() - t0 > budget:
            break
    outs = outs[:done]
    st = strength[picks[:done]]
    return _mc_res(float(outs.mean()), st, outs, np.ones(done))


def _mc_res(eq, strength, out, w):
    r = MCResult()
    r.eq, r.strength, r.out, r.w = eq, strength, out, w
    return r


def eq_vs_callers(res, fold_frac):
    """Equity against the part of the sampled range that doesn't fold, assuming
    the opponent folds their weakest `fold_frac` of hands."""
    if fold_frac <= 0:
        return res.eq
    order = np.argsort(res.strength)
    cw = np.cumsum(res.w[order]) / res.w.sum()
    keep = order[cw > fold_frac]
    if len(keep) == 0:
        return res.eq
    return float(res.out[keep] @ res.w[keep] / res.w[keep].sum())


# -- actions ----------------------------------------------------------------
def act_fold(gs):
    return {"action": "check"} if gs["can_check"] else {"action": "fold"}


def act_call(gs):
    return {"action": "check"} if gs["can_check"] else {"action": "call"}


def act_raise_to(gs, c, total):
    if not c.can_raise:
        return act_call(gs)
    total = int(max(total, gs["min_raise_to"]))
    if total >= c.my_max * 0.9 or total >= c.opp_max:
        return {"action": "all_in"}
    return {"action": "raise", "amount": total}


def bet_pot_frac(gs, c, frac):
    """Bet/raise to a total that is `frac` of the pot after calling."""
    owed = gs["amount_owed"]
    return gs["current_bet"] + frac * (gs["pot"] + owed)


# =========================== MONTE CARLO STRATEGY ===========================
# Every decision: simulate thousands of runouts against the opponent's likely
# hands, then pick the action with the highest expected value in chips.


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        pushfold_nash(10)
        mc_equity(["As", "Kd"], ["2c", "7h", "Ts"], np.ones(169), budget=0.2)
        return {"action": "check"}
    try:
        return _decide(game_state)
    except Exception as e:
        print("error:", repr(e))
        return act_fold(game_state)


def fold_prob(call_amt, pot_before):
    """Prior chance the opponent folds to a bet: the share of a balanced range
    that minimum defence frequency lets them fold (bet / (pot + bet))."""
    r = call_amt / max(pot_before, 1)
    return min(max(0.92 * r / (1 + r), 0.05), 0.85)


def candidate_raises(gs, c):
    bb, pot = c.bb, gs["pot"]
    out = []
    if c.street == "preflop":
        if gs["current_bet"] <= bb:              # unopened / limped pot
            out += [2.5 * bb, 3.5 * bb]
        else:
            out += [gs["current_bet"] * 3]
        if c.eff_bb <= 25 or gs["current_bet"] >= 0.25 * c.eff:
            out.append(c.my_max)
    else:
        for frac in (0.33, 0.66, 1.0):
            out.append(bet_pot_frac(gs, c, frac))
        if c.eff <= 1.6 * pot or gs["current_bet"] > 0:
            out.append(c.my_max)
    seen, res = set(), []
    for t in out:
        t = int(min(max(t, gs["min_raise_to"]), c.my_max))
        if t not in seen and t > gs["current_bet"]:
            seen.add(t)
            res.append(t)
    return res


def _decide(gs):
    c = context(gs)
    owed, pot = gs["amount_owed"], gs["pot"]
    push_range = pushfold_nash(c.eff_bb, c.ante / c.bb)[0] if c.eff_bb <= 15 else None
    rng = opp_preflop_range(c, gs, push_range=push_range)
    budget = 0.35 if c.street != "preflop" else 0.25
    res = mc_equity(gs["your_cards"], gs["community_cards"], rng,
                    n_aggr=c.opp_post_aggr, n_calls=c.opp_post_calls, budget=budget)
    eq = res.eq
    river = c.street == "river"
    realize = 1.0 if river else (0.92 if c.button else 0.85)

    evs = {}
    if gs["can_check"]:
        evs["check"] = eq * pot * realize
    else:
        evs["fold"] = 0.0
        all_in_call = owed >= gs["your_stack"] or c.opp_stack == 0
        evs["call"] = eq * (pot + owed) * (1.0 if all_in_call else realize) - owed

    if c.can_raise:
        for total in candidate_raises(gs, c):
            add = total - gs["your_bet_this_street"]                 # chips we put in
            opp_add = min(total, c.opp_max) - c.opp_bet              # chips they must call
            fe = fold_prob(opp_add, pot + add)
            eq_c = eq_vs_callers(res, fe)
            jam = total >= c.my_max or total >= c.opp_max
            r_c = 1.0 if (jam or river) else realize
            evs[("raise", total)] = fe * pot + (1 - fe) * (eq_c * (pot + add + opp_add) * r_c - add)

    best = max(evs, key=evs.get)
    dbg = {"eq": round(eq, 3), "ev": {str(k): round(v) for k, v in evs.items()}}
    if best == "check":
        return {"action": "check", "debug": dbg}
    if best == "fold":
        return {"action": "fold", "debug": dbg}
    if best == "call":
        return {"action": "call", "debug": dbg}
    return {**act_raise_to(gs, c, best[1]), "debug": dbg}
