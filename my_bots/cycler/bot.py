"""
CYCLER: switches between strategies, keeping whichever one is winning.

Every exploit leaves you open to a counter-exploit, and the counter to any
counter is to go back toward equilibrium. The modes follow that cycle:

    exploit -> pressure -> value -> bluff_catch -> gear_shift -> gto

It credits each hand's result (in big blinds) to the mode that played it. If the
active mode has lost more than SWITCH_BELOW_BB over the last WINDOW hands, it
jumps to the mode with the best whole-match result per hand (shrunk toward 0
for small samples), excluding the one that just failed. Untried modes score 0,
so each one gets tried, in cycle order, before the bot settles on proven ones.
A mode that is still winning and has the best whole-match record is kept
through a bad stretch, and bluff_catch is only tried against opponents that
bet or raise at least 40% of the time after the flop. Otherwise it keeps the
current mode.

Postflop it won't get into raise wars with draws: each re-raise from the
opponent narrows their range twice as hard, and a raise must beat calling by a
margin per extra chip risked (the EVs are estimates), or it just calls.

It also reads how honest the opponent is: if their big bets keep winning at
showdown, it believes their bets more (folds medium hands to them), counts only
the folds they've actually shown to big bets as fold equity, rules out pressure
and bluff_catch, and doesn't jam deep stacks into small pots. The active mode and its running total are
in every decision's "debug".

All modes share one engine: Monte Carlo equity against the opponent's range,
then the highest-EV action. The modes differ in what they assume about the
opponent: fold frequency, how often their bets are bluffs, whether bluffing is
allowed, and how often to trap. Every mode bets from the same sizes, so a bet's
size doesn't reveal the mode.

On top of the active mode it scales risk with the chip count: ahead in the
match it plays more conservatively to protect the lead, behind it takes more
risk to get back into it (see RISK_* below).

Every decision is held to THINK_LIMIT (200 ms); a decision that runs long
finishes with the main strategy (exploit, the default mode).
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


PF_MAX_BB = 16           # depths solved at warmup: 1..PF_MAX_BB bb, with and without antes


def _pf_key(stack_bb, ante_bb):
    return (int(min(max(round(stack_bb), 1), PF_MAX_BB)), 0.1 if ante_bb > 0.05 else 0.0)


def presolve_pushfold(deadline):
    """Solve every depth during the warmup call so decisions only look them up."""
    for ante in (0.0, 0.1):
        for s in range(1, PF_MAX_BB + 1):
            if time.time() > deadline:
                return
            pushfold_nash(s, ante, iters=200)


def pushfold_nash(stack_bb, ante_bb=0.0, iters=60):
    """(push, call) probabilities per class for the SB-shove / BB-call game with
    effective stack stack_bb. Solved by fictitious play and cached per whole big
    blind. Depths deeper than PF_MAX_BB use the PF_MAX_BB solution (only the
    <=15bb spots use it). A depth missed at warmup gets a quick 60-iteration
    solve (~40 ms) instead of a full one."""
    key = _pf_key(stack_bb, ante_bb)
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
        if (k & 7) == 7 and time.time() - t0 > budget:
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


# ============================ CYCLER STRATEGY ===============================
# Several strategies ("modes"), one active at a time. After every hand it adds
# up how many big blinds the active mode has won over the last WINDOW hands. If
# that total drops below SWITCH_BELOW_BB, the mode isn't working: jump to the
# best other mode by whole-match bb/hand (see pick_next_mode). If it's holding
# up, keep it.
#
# The order (used to try untried modes and to break ties) follows the
# exploit / counter-exploit cycle:
#   exploit      DEFAULT. Best response to the opponent's measured frequencies
#                (beats counter_nash, which is itself an exploiter)
#   pressure     vs over-folders: bluff more, expect more folds
#   value        vs calling stations: value-bet thin and big, never bluff
#   bluff_catch  vs over-aggressive bots: believe their bets less, call down
#                lighter, trap strong hands by checking
#   gear_shift   vs opponent-modelling bots: bait them with tight, passive play,
#                then call down and trap once they start bluffing more (below)
#   gto          balanced own play: fold-equity guesses from equilibrium,
#                not from the opponent (the counter to whatever counter they found)
# ...and then back to exploit.
# Every mode reads the opponent's *ranges* from what it has observed (how wide
# they open, re-raise and shove), so none of them keeps folding to a bot that
# shoves far wider than equilibrium.
from collections import defaultdict

# Every mode bets from the same sizes (pot fractions), so the size of a bet
# never tells the opponent which mode is active. Modes still differ in how
# often they pick each size, since each mode values the outcomes differently.
BET_SIZES = (0.33, 0.66, 1.0, 1.5)

MODES = [
    dict(name="exploit",     use_model=True,  fe_mult=1.00, bluff_keep=None, sizes=BET_SIZES, bluffs=True,  trap=0.00),
    dict(name="pressure",    use_model=True,  fe_mult=1.30, bluff_keep=0.30, sizes=BET_SIZES, bluffs=True,  trap=0.00),
    dict(name="value",       use_model=True,  fe_mult=0.50, bluff_keep=0.30, sizes=BET_SIZES, bluffs=False, trap=0.00),
    dict(name="bluff_catch", use_model=True,  fe_mult=0.80, bluff_keep=0.65, sizes=BET_SIZES, bluffs=False, trap=0.35),
    dict(name="gear_shift",  use_model=True,  fe_mult=0.60, bluff_keep=0.15, sizes=BET_SIZES, bluffs=False, trap=0.00),
    dict(name="gto",         use_model=False, fe_mult=1.00, bluff_keep=0.30, sizes=BET_SIZES, bluffs=True,  trap=0.00),
]
WINDOW = 15               # hands a mode is judged over (rolling)
SWITCH_BELOW_BB = -8.0    # switch when the last WINDOW hands lost more than this many big blinds

MODE_IDX = 0
RECENT = []               # big blinds won per hand under the active mode (rolling window)
LAST = None               # (hand_num, stack at hand start, bb) of the last hand we saw
MODE_LOG = defaultdict(lambda: [0, 0.0])   # mode -> [hands played, bb won], whole match

# Chip-lead risk. Ahead in the match: play conservatively to protect the lead.
# Behind: take more risk to get back into it. This sits on top of whichever
# mode is active. It works as a per-chip charge on the chips an action puts at
# risk: a penalty when ahead (each chip risked must earn RISK more), a bonus
# when behind (risky lines get credit for RISK per chip).
RISK_DEADZONE = 0.05       # between 45% and 55% of the chips: neutral
RISK_FULL_SHARE = 0.75     # full effect at 75% (ahead) or 25% (behind) of the chips
RISK_AHEAD_MAX = 0.12      # conservative: 12% charge per chip risked
RISK_BEHIND_MAX = 0.10     # aggressive: 10% credit per chip risked
HAND_SHARE = {}            # hand_num -> our share of the chips at the start of the hand


def chip_share(gs, c):
    hn = gs["hand_num"]
    if hn not in HAND_SHARE:
        if c.street == "preflop":
            mine, theirs = c.my_max + c.ante, c.opp_max + c.ante
        else:
            mine, theirs = gs["your_stack"] + gs["pot"] / 2, c.opp_stack + gs["pot"] / 2
        HAND_SHARE[hn] = mine / max(mine + theirs, 1)
    return HAND_SHARE[hn]


def risk_charge(share):
    d = share - 0.5
    if abs(d) <= RISK_DEADZONE:
        return 0.0
    scale = min((abs(d) - RISK_DEADZONE) / (RISK_FULL_SHARE - 0.5 - RISK_DEADZONE), 1.0)
    return RISK_AHEAD_MAX * scale if d > 0 else -RISK_BEHIND_MAX * scale

STATS = defaultdict(lambda: [0, 0])     # key -> [times it happened, opportunities]
_SEEN = set()                           # hand_nums already folded into STATS
_EFF_BB = {}                            # hand_num -> effective stack in bb
SHRINK = 8                              # pseudo-observations given to each prior
PUSH_FOLD_BB = 12


# ---- the cycle ----------------------------------------------------------------
SCORE_SHRINK = 10         # pseudo-hands at 0 bb added to each mode's record when scoring


def mode_score(name):
    """A mode's whole-match bb per hand, shrunk toward 0 so a few lucky hands
    don't make it look best. An untried mode scores exactly 0."""
    hands, bb = MODE_LOG[name]
    return bb / (hands + SCORE_SHRINK)


BLUFF_CATCH_MIN_AGGR = 0.40   # only try bluff_catch against bots that bet/raise this often postflop


PRESSURE_MIN_BIG_FOLD = 0.25  # pressure needs them to fold at least this often to big bets
HONEST_AT = 0.75              # their big bets win at showdown this often: treat them as honest
JAM_MAX_SPR = 3.0             # postflop jam allowed facing a bet only if stacks <= this many pots


def mode_allowed(m):
    """bluff_catch assumes they bluff a lot; against a bot that rarely bets
    (like Nash) it just pays off value bets, so it needs evidence first.
    pressure assumes they fold to big bets; against a bot that calls big bets
    with strong hands and only bets big with strong hands it just burns chips."""
    if m["name"] == "bluff_catch":
        return est("post_aggr", 0.35) >= BLUFF_CATCH_MIN_AGGR and est("big_honest", 0.6) < HONEST_AT
    if m["name"] == "pressure":
        return est("bigbet_fold", 0.35) >= PRESSURE_MIN_BIG_FOLD and est("big_honest", 0.6) < HONEST_AT
    return True


def pick_next_mode(cur, force=False):
    """After a bad stretch: if the current mode still has the best whole-match
    record and is winning overall, the stretch was bad luck, so keep it.
    Otherwise jump to the best-scoring other mode. A mode that hasn't been
    tried scores 0, which beats any mode that has been losing, so every mode
    gets a turn before the bot settles on proven ones. Ties go to whichever
    mode comes next in the cycle order."""
    n = len(MODES)
    order = [(cur + k) % n for k in range(1, n) if mode_allowed(MODES[(cur + k) % n])]
    if not order:
        return cur
    best = max(order, key=lambda i: (mode_score(MODES[i]["name"]), -order.index(i)))
    cur_score = mode_score(MODES[cur]["name"])
    if not force and cur_score > 0 and cur_score >= mode_score(MODES[best]["name"]):
        return cur
    return best


def track_results(gs, c):
    """Once per hand: credit the stack change since the last hand to the mode
    that played it, then decide whether to keep that mode."""
    global LAST, MODE_IDX, RECENT
    hn = gs["hand_num"]
    if LAST is not None and LAST[0] == hn:
        return
    start = c.my_max + c.ante if c.street == "preflop" else None
    if start is None:
        return
    if LAST is not None:
        won_bb = (start - LAST[1]) / max(LAST[2], 1)
        HAND_RESULT[LAST[0]] = won_bb
        resolve_honesty()
        mode = MODES[MODE_IDX]["name"]
        MODE_LOG[mode][0] += 1
        MODE_LOG[mode][1] += won_bb
        RECENT.append(won_bb)
        RECENT = RECENT[-WINDOW:]
        if len(RECENT) >= WINDOW and sum(RECENT) < SWITCH_BELOW_BB:
            lost = sum(RECENT)
            prev = MODE_IDX
            MODE_IDX = pick_next_mode(MODE_IDX)
            RECENT = []
            if MODE_IDX != prev and MODES[MODE_IDX]["name"] == "gear_shift":
                gear_reset()                      # fresh baseline each time it starts
            scores = ", ".join(f"{m['name']} {mode_score(m['name']):+.2f}" for m in MODES)
            what = "keeping it, still the best overall" if MODE_IDX == prev else \
                f"switching to {MODES[MODE_IDX]['name']}"
            print(f"hand {hn}: {mode} at {lost:.0f}bb over {WINDOW} hands -> {what} (bb/hand: {scores})")
        if not mode_allowed(MODES[MODE_IDX]):     # evidence now rules this mode out: leave at once
            prev = MODE_IDX
            MODE_IDX = pick_next_mode(MODE_IDX, force=True)
            if MODE_IDX != prev:
                RECENT = []
                if MODES[MODE_IDX]["name"] == "gear_shift":
                    gear_reset()
                print(f"hand {hn}: {MODES[prev]['name']} ruled out by the opponent's tendencies "
                      f"-> switching to {MODES[MODE_IDX]['name']}")
    LAST = (hn, start, c.bb)


# ---- opponent model (same as counter_nash) --------------------------------------
def est(key, prior, k=SHRINK):
    hits, n = STATS[key]
    return (hits + prior * k) / (n + k)


def _rec(key, happened):
    s = STATS[key]
    s[0] += 1 if happened else 0
    s[1] += 1


def update_model(gs):
    me, cur = gs["your_bot_id"], gs["hand_num"]
    hands = defaultdict(list)
    for e in gs.get("match_action_log") or []:
        hn = e.get("hand_num")
        if hn is None or hn >= cur or hn in _SEEN:
            continue
        hands[hn].append(e)
    for hn in sorted(hands):
        _SEEN.add(hn)
        ents = hands[hn]
        if any(e["action"] == "big_blind" for e in ents):
            _process_hand(hn, ents, me, _EFF_BB.get(hn, 30.0) <= PUSH_FOLD_BB)


# Honesty: when they bet big after the flop and the hand goes to showdown, do
# they turn up with the best hand? A bot that only bets big with strong hands
# (and only calls big bets with strong hands) should get no bluffs from us and
# no hero calls. Resolved once the hand's result is known (track_results).
PENDING_HONEST = []       # hands where they bet/raised big postflop and showed down
HAND_RESULT = {}          # hand_num -> big blinds we won
BIG_BET = 0.75            # a bet of at least this share of the pot counts as big


def resolve_honesty():
    for hn in list(PENDING_HONEST):
        if hn in HAND_RESULT:
            _rec("big_honest", HAND_RESULT[hn] < 0)     # their big bet beat us at showdown
            PENDING_HONEST.remove(hn)


def _process_hand(hn, ents, me, short):
    sb = next((e["bot_id"] for e in ents if e["action"] == "small_blind"), None)
    street, level, cur_bet, pending, first = None, 0, 0, [], True
    opp_aggr = opp_acts = 0
    pot, sbet = 0, {}
    opp_big, opp_shown = False, False
    for e in ents:
        a, who, amt = e["action"], e.get("bot_id"), e.get("amount", 0) or 0
        if a == "show":
            opp_shown = opp_shown or who != me
            continue
        if e["street"] != street:
            street, cur_bet, pending, sbet = e["street"], 0, [], {}
        if a in ("small_blind", "big_blind", "ante"):
            pot += amt
            if a != "ante":
                sbet[who] = sbet.get(who, 0) + amt
                cur_bet = max(cur_bet, sbet[who])
            continue
        if a not in ("fold", "check", "call", "raise", "all_in"):
            continue
        aggr = a == "raise" or (a == "all_in" and amt > cur_bet)
        size = (amt - cur_bet) / max(pot, 1) if aggr else 0.0
        big = aggr and (a == "all_in" or size >= BIG_BET or cur_bet > 0)   # jam, big bet or re-raise
        # chips into the pot
        if a == "call":
            pot += amt
            sbet[who] = sbet.get(who, 0) + amt
        elif a in ("raise", "all_in"):
            pot += max(amt - sbet.get(who, 0), 0)
            sbet[who] = max(sbet.get(who, 0), amt)
        if aggr:
            cur_bet = max(cur_bet, amt)
        if who != me and street != "preflop" and big:
            opp_big = True
        if who != me:
            for p in pending:
                _rec(p + "_fold", a == "fold")
                _rec(p + "_raise", aggr)
            pending = []
            if street == "preflop" and first and e["bot_id"] == sb:
                tag = "short" if short else "deep"
                _rec("sb_open_" + tag, aggr)
                _rec("sb_jam_" + tag, a == "all_in")
            if street != "preflop":
                _rec("post_aggr", aggr)
                opp_aggr += 1 if aggr else 0
                opp_acts += 1
        elif aggr:
            if street == "preflop":
                p = "jam" if a == "all_in" else ("open" if level == 0 else "3bet" if level == 1 else "4bet")
                if p == "jam":
                    p += "_short" if short else "_deep"
                pending = [p]
            else:
                pending = ["bet", "bigbet"] if big else ["bet"]
        elif street == "preflop" and first and a == "call" and who == sb:
            pending = ["limp"]                        # we limped: does the BB punish it?
        if street == "preflop":
            first = False
            if aggr:
                level += 1
    AGGR_HIST.append((opp_aggr, opp_acts))
    if opp_big and opp_shown:
        PENDING_HONEST.append(hn)


# ---- gear_shift: the counter to opponent-modelling bots ---------------------------
# A bot that models us adjusts to what it has seen, with a lag. gear_shift uses
# that lag. In "bait" it plays tight and passive after the flop (no bluffs, and
# it believes their bets, so it folds marginal hands). A modelling bot reads that
# as "folds too much" and starts bluffing more. Once their recent aggression is
# clearly above what it was when gear_shift started, it "springs": calls down
# light and traps strong hands while they bluff into it. When their aggression
# drops back (their model caught up), it goes back to baiting.
AGGR_HIST = []            # per finished hand: (their postflop bets/raises, their postflop actions)
GEAR = {"phase": "bait", "baseline": None, "start": 0}
GEAR_WINDOW = 12          # hands of recent aggression to compare with the baseline
GEAR_MIN_ACTS = 6         # postflop actions needed in that window before it acts on it
GEAR_SPRING_AT = 0.10     # spring when recent aggression >= baseline + this
GEAR_RESET_AT = 0.03      # back to bait when it falls to baseline + this
GEAR_PHASES = {
    "bait":   dict(fe_mult=0.60, bluff_keep=0.15, sizes=BET_SIZES, bluffs=False, trap=0.00),
    "spring": dict(fe_mult=0.80, bluff_keep=0.70, sizes=BET_SIZES, bluffs=False, trap=0.45),
}


def gear_reset():
    GEAR.update(phase="bait", baseline=None, start=len(AGGR_HIST))


def gear_mode(base):
    """The gear_shift mode's settings for this decision (bait or spring)."""
    if GEAR["baseline"] is None:
        GEAR["baseline"] = est("post_aggr", 0.35)
        GEAR["start"] = len(AGGR_HIST)
    recent = AGGR_HIST[max(GEAR["start"], len(AGGR_HIST) - GEAR_WINDOW):]
    a, n = sum(x for x, _ in recent), sum(y for _, y in recent)
    rate = a / n if n >= GEAR_MIN_ACTS else None
    if rate is not None:
        if GEAR["phase"] == "bait" and rate >= GEAR["baseline"] + GEAR_SPRING_AT:
            GEAR["phase"] = "spring"
        elif GEAR["phase"] == "spring" and rate <= GEAR["baseline"] + GEAR_RESET_AT:
            GEAR["phase"] = "bait"
    m = dict(base, **GEAR_PHASES[GEAR["phase"]])
    m["gear"] = {"phase": GEAR["phase"], "baseline": round(GEAR["baseline"], 2),
                 "recent": None if rate is None else round(rate, 2)}
    return m


# ---- per-mode ingredients -------------------------------------------------------
def _opp_range(c, gs, mode):
    """The opponent's range, with each line's width taken from what they've
    actually done (all modes: defending against their real ranges is never a
    mistake, however the mode chooses to attack)."""
    push, _ = pushfold_nash(c.eff_bb, c.ante / c.bb)
    push_range = top_range(est("sb_jam_short", range_share(push))) if c.eff_bb <= PUSH_FOLD_BB else None
    f_open = max(est("sb_open_deep", 0.80), 0.05)
    # share of ALL their hands that re-raise us, per spot
    f_3bet = min(max(est("open_raise", 0.15), 0.03), 0.95)            # BB vs our open
    f_iso = min(max(est("limp_raise", 0.35), 0.05), 0.95)             # BB vs our limp
    f_4bet = min(max(f_open * est("3bet_raise", 0.09), 0.02), 0.9)    # SB vs our 3-bet
    return opp_preflop_range(c, gs, f_open=f_open, f_iso=f_iso, f_3bet=f_3bet, f_4bet=f_4bet,
                             f_call_open=max(0.1, 1 - est("open_fold", 0.35)), push_range=push_range)


def _fold_prob(c, pot_before, call_amt, jam, mode):
    """Chance the opponent folds to our bet: equilibrium priors in gto mode,
    measured rates otherwise, then scaled by the mode's fe_mult."""
    get = est if mode["use_model"] else (lambda key, prior: prior)
    r = call_amt / max(pot_before, 1)
    mdf_fold = min(max(0.92 * r / (1 + r), 0.05), 0.85)
    if c.street == "preflop":
        if jam:
            if c.eff_bb <= PUSH_FOLD_BB:
                _, call = pushfold_nash(c.eff_bb, c.ante / c.bb)
                f = get("jam_short_fold", 1 - range_share(call))
            else:
                f = get("jam_deep_fold", 0.7)
        else:
            key = "open" if c.pf_raises == 0 else "3bet" if c.pf_raises == 1 else "4bet"
            f = get(key + "_fold", mdf_fold)
    elif jam or r >= BIG_BET:
        f = get("bigbet_fold", mdf_fold)      # big bets: what they actually fold to big bets
    else:
        f = get("bet_fold", 0.40) * mdf_fold / 0.37
    # A jam's fold equity is what they've shown; modes that assume extra folds
    # (pressure) don't get to inflate it, since a called jam risks the stack.
    mult = min(mode["fe_mult"], 1.0) if jam else mode["fe_mult"]
    return min(max(f * mult, 0.02), 0.95)


def candidate_raises(gs, c, mode, behind=False):
    bb, pot = c.bb, gs["pot"]
    out = []
    if c.street == "preflop":
        out += [2.5 * bb, 3.5 * bb] if gs["current_bet"] <= bb else [gs["current_bet"] * 3]
        if c.eff_bb <= 25 or gs["current_bet"] >= 0.25 * c.eff or (behind and c.eff_bb <= 40):
            out.append(c.my_max)
    else:
        out += [bet_pot_frac(gs, c, f) for f in mode["sizes"]]
        # Jam only when it isn't a huge overbet: deep stacks into a small pot get
        # called only by hands that beat us.
        if c.eff <= 1.6 * pot or (gs["current_bet"] > 0 and c.eff <= JAM_MAX_SPR * pot):
            out.append(c.my_max)
    seen, res = set(), []
    for t in out:
        t = int(min(max(t, gs["min_raise_to"]), c.my_max))
        if t not in seen and t > gs["current_bet"]:
            seen.add(t)
            res.append(t)
    return res


# ---- decisions --------------------------------------------------------------------
def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        presolve_pushfold(time.time() + 20)     # warmup gets 30 s; leave a margin
        return {"action": "check"}
    t0 = time.time()
    try:
        update_model(game_state)
        return _decide(game_state, t0)
    except Exception as e:
        print("error:", repr(e))
        return act_fold(game_state)


# Thinking budget. The tournament allows 2 s, but this bot holds itself to
# THINK_LIMIT. If a decision runs long, it drops the active mode and finishes
# with the main strategy (exploit, the default mode) instead.
THINK_LIMIT = 0.200       # seconds per decision
MC_SHARE = 0.35           # the Monte Carlo simulation gets what's left of 35% of it,
                          # leaving room for slow machines and the half-core sandbox
MAIN = MODES[0]           # the main strategy to fall back to
FALLBACKS = 0             # how many decisions fell back this match


RERAISE_BLUFF_KEEP = 0.5  # per re-raise: how much less often their range is a bluff
RAISE_MARGIN = 0.03       # a raise must beat the passive line by 3% of the extra chips it risks


def _elapsed(t0):
    return time.time() - t0


def _opp_reraises(c):
    """How many times this hand the opponent raised after we bet or raised on
    the same postflop street."""
    n = 0
    for st in ("flop", "turn", "river"):
        seq = [e for e in c.log if e["street"] == st]
        ours = False
        for e in seq:
            if _aggressive(e, seq):
                if e["bot_id"] == c.me:
                    ours = True
                elif ours:
                    n += 1
    return n


def _decide(gs, t0):
    global FALLBACKS
    c = context(gs)
    _EFF_BB.setdefault(gs["hand_num"], c.eff_bb)
    track_results(gs, c)
    mode = MODES[MODE_IDX]
    if mode["name"] == "gear_shift":
        mode = gear_mode(mode)
    owed, pot = gs["amount_owed"], gs["pot"]
    share = chip_share(gs, c)
    risk = risk_charge(share)                    # > 0 ahead (conservative), < 0 behind (aggressive)
    tag = {"mode": mode["name"], "window_bb": round(sum(RECENT), 1),
           "chip_share": round(share, 2), "risk": round(risk, 3)}
    if "gear" in mode:
        tag["gear"] = mode["gear"]
    fell_back = False
    if mode is not MAIN and _elapsed(t0) > 0.4 * THINK_LIMIT:   # already slow: go straight to main
        mode, fell_back = MAIN, True

    # Short-stack, first in from the SB: Nash push/fold in gto mode, best response otherwise
    if c.street == "preflop" and c.eff_bb <= PUSH_FOLD_BB and not c.pf and c.button:
        if not mode["use_model"] and risk == 0:
            push, _ = pushfold_nash(c.eff_bb, c.ante / c.bb)
            return {"action": "all_in" if random.random() < push[c.cls] else "fold", "debug": tag}
        a, S = c.ante / c.bb, c.eff_bb
        fold_p = _fold_prob(c, pot, c.eff - c.opp_bet, True, mode)
        eq = eq_vs_range(c.cls, top_range(max(1 - fold_p, 0.02)))
        ev_push = fold_p * (1 + a) + (1 - fold_p) * (eq * (2 * S + 2 * a) - S - a)
        ev_push -= risk * (1 - fold_p) * S          # chips at risk when called
        return {"action": "all_in" if ev_push > -(0.5 + a) else "fold", "debug": tag}

    if mode["bluff_keep"] is None:              # exploit mode: trust their bets as much as they've earned
        bluff_keep = min(max(0.1 + 0.9 * (est("post_aggr", 0.35) - 0.1), 0.08), 0.75)
    else:
        bluff_keep = mode["bluff_keep"]
    # Raise wars: each time they re-raise one of our postflop bets, narrow their
    # range twice as hard and believe it is a bluff half as often. A re-raise is
    # far stronger than a first bet, and draws must not survive it.
    reraises = _opp_reraises(c)
    if reraises:
        bluff_keep *= RERAISE_BLUFF_KEEP ** reraises
        tag = {**tag, "opp_reraises": reraises}
    # Honest opponents: if their big bets keep winning at showdown, believe their
    # bets more (whatever the mode assumes), so medium hands fold to them.
    honest = est("big_honest", 0.6)
    if honest > 0.6:
        bluff_keep *= max(0.25, (1 - honest) / 0.4)
        tag = {**tag, "honest": round(honest, 2)}
    res = mc_equity(gs["your_cards"], gs["community_cards"], _opp_range(c, gs, mode),
                    n_aggr=c.opp_post_aggr + reraises, n_calls=c.opp_post_calls, bluff_keep=bluff_keep,
                    budget=max(0.005, MC_SHARE * THINK_LIMIT - _elapsed(t0)))
    if mode is not MAIN and _elapsed(t0) > 0.8 * THINK_LIMIT:    # ran long: finish with main
        mode, fell_back = MAIN, True
    if fell_back:
        FALLBACKS += 1
        tag = {**tag, "fallback_to": MAIN["name"], "fallbacks": FALLBACKS}
    eq = res.eq
    river = c.street == "river"
    realize = 1.0 if river else (0.92 if c.button else 0.85)

    if gs["can_check"] and c.street != "preflop" and not river and eq > 0.8 and random.random() < mode["trap"]:
        return {"action": "check", "debug": {**tag, "eq": round(eq, 3), "trap": 1}}

    evs = {}
    if gs["can_check"]:
        evs["check"] = eq * pot * realize
    else:
        evs["fold"] = 0.0
        all_in_call = owed >= gs["your_stack"] or c.opp_stack == 0
        evs["call"] = eq * (pot + owed) * (1.0 if all_in_call else realize) - owed
        if c.street == "preflop" and c.button and not c.pf:
            # A limp only sees a flop if the BB lets it; when they raise, count it as lost.
            p_raised = est("limp_raise", 0.35)
            evs["call"] = (1 - p_raised) * evs["call"] - p_raised * owed
        evs["call"] -= risk * owed

    if c.can_raise:
        for total in candidate_raises(gs, c, mode, behind=risk < 0):
            add = total - gs["your_bet_this_street"]
            opp_add = min(total, c.opp_max) - c.opp_bet
            jam = total >= c.my_max or total >= c.opp_max
            fe = _fold_prob(c, pot + add, opp_add, jam, mode)
            eq_c = eq_vs_callers(res, fe)
            if not mode["bluffs"] and eq_c < 0.5:  # value-only modes never bet as a bluff
                continue
            r_c = 1.0 if (jam or river) else realize
            evs[("raise", total)] = fe * pot + (1 - fe) * (eq_c * (pot + add + opp_add) * r_c - add)
            evs[("raise", total)] -= risk * add * (1 - fe)    # chips are only at risk when called

    # Only build the pot when a raise clearly beats checking/calling: the EVs are
    # estimates, so a raise must win by RAISE_MARGIN per extra chip it risks
    # (more once they've re-raised), or we take the passive line.
    margin = RAISE_MARGIN * (1 + reraises)

    def score(k):
        if not isinstance(k, tuple):
            return evs[k]
        extra = k[1] - gs["your_bet_this_street"] - owed      # chips beyond a call
        return evs[k] - margin * max(extra, 0)

    best = max(evs, key=score)
    dbg = {**tag, "eq": round(eq, 3), "think_ms": round(_elapsed(t0) * 1000),
           "ev": {str(k): round(v) for k, v in evs.items()}}
    if best in ("check", "fold", "call"):
        return {"action": best, "debug": dbg}
    return {**act_raise_to(gs, c, best[1]), "debug": dbg}
