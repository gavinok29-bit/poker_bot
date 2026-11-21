
# ============================ STRATEGY HELPERS ==============================
# (shared by the strategy bots: opponent stats, board texture, GTO baseline)

# Hard limit: 200 ms per decision (the tournament allows 2 s). The Monte Carlo
# simulation gets what's left of 35% of it, which leaves room for slow
# machines, the half-core sandbox and everything else the bot does.
THINK_LIMIT = 0.200
MC_SHARE = 0.35
_T0 = [0.0]


def mc_budget():
    return max(0.005, MC_SHARE * THINK_LIMIT - (time.time() - _T0[0]))


def made_pct(hole, board):
    """Where `hole` ranks among every possible holding on this board right now
    (0 = weakest, 1 = the nuts). Made hands only, no draws."""
    dead = set(hole) | set(board)
    bc = [CARD[c] for c in board]
    me = eval7.evaluate([CARD[c] for c in hole] + bc)
    rest = [CARD[c] for c in DECK if c not in dead]
    below = total = 0
    for a in range(len(rest)):
        for b in range(a + 1, len(rest)):
            v = eval7.evaluate([rest[a], rest[b]] + bc)
            total += 1
            below += 1 if v < me else 0.5 if v == me else 0
    return below / max(total, 1)


class Rate:
    """A frequency that forgets: recent hands count more (decay per sample),
    shrunk toward a prior so a few samples don't swing it."""

    def __init__(self, decay=0.96):
        self.n = 0.0
        self.h = 0.0
        self.d = decay

    def add(self, hit):
        self.n = self.n * self.d + 1
        self.h = self.h * self.d + (1.0 if hit else 0.0)

    def get(self, prior, k=6.0):
        return (self.h + prior * k) / (self.n + k)


class OppStats:
    """Opponent tendencies, read from match_action_log once each hand is over."""

    def __init__(self):
        self.seen = set()
        self.boards = {}                       # hand_num -> {street: board we saw}
        self.hands = 0
        self.fold_to_bet = Rate()              # they fold when we bet/raise postflop
        self.raise_vs_bet = Rate()             # they raise when we bet/raise postflop
        self.aggr = Rate()                     # their postflop actions that are bets/raises
        self.cbet = Rate()                     # they bet the flop after raising preflop
        self.fold_to_flop_raise = Rate()       # they fold when we raise their flop bet
        self.we_fold_by_size = {"small": Rate(), "big": Rate(), "over": Rate()}
        self.opp_open = Rate()                 # they raise first in from the small blind
        self.fold_to_open = Rate()             # they fold when we open from the small blind
        self.fold_to_3bet = Rate()             # they fold when we 3-bet their open
        self.lead_sizes = []                   # their first bets on a postflop street, as pot fractions (recent)
        self.sizes = []                        # (size fraction, made_pct) of bets they showed down

    def update(self, gs):
        me, cur = gs["your_bot_id"], gs.get("hand_num", 0)
        if gs.get("community_cards"):
            self.boards.setdefault(cur, {})[gs["street"]] = list(gs["community_cards"])
        log = gs.get("match_action_log") or []
        by_hand = {}
        for e in log:
            h = e.get("hand_num")
            if h is None or h >= cur or h in self.seen:
                continue
            by_hand.setdefault(h, []).append(e)
        if len(log) >= 200 and by_hand:          # the oldest hand may be cut off
            self.seen.add(min(by_hand))
            by_hand.pop(min(by_hand))
        for h in sorted(by_hand):
            self.seen.add(h)
            try:
                self._hand(h, by_hand[h], me)
            except Exception as ex:
                print("stats error:", repr(ex))
            self.boards.pop(h, None)

    def _hand(self, h, entries, me):
        self.hands += 1
        pot = 0
        street, street_bet, cur_bet = None, {}, 0
        pf_aggr = None
        flop_first_opp = None
        opp_bets = []                           # (street, size fraction)
        last_aggr_by_us = {}                    # street -> True after our bet until opp answers
        opp_flop_bet = False
        we_raised_flop_bet = False
        shown = None
        pf_first, pf_pending, pf_level = None, None, 0
        for e in entries:
            a, who, amt, st = e["action"], e.get("bot_id"), e.get("amount", 0) or 0, e.get("street")
            opp = who != me
            if a == "show":
                if opp:
                    shown = e.get("cards")
                continue
            if st != street:
                street, street_bet, cur_bet = st, {}, 0
            if a in ("small_blind", "big_blind"):
                pot += amt
                street_bet[who] = street_bet.get(who, 0) + amt
                cur_bet = max(cur_bet, street_bet[who])
                continue
            if a == "ante":
                pot += amt
                continue
            post = st != "preflop"
            aggressive = a == "raise" or (a == "all_in" and amt > cur_bet)
            if not post:                                  # preflop tendencies
                if pf_first is None:
                    pf_first = who
                    if opp:
                        self.opp_open.add(aggressive)
                if opp and pf_pending:
                    (self.fold_to_open if pf_pending == "open" else self.fold_to_3bet).add(a == "fold")
                    pf_pending = None
                if not opp and aggressive:
                    pf_pending = ("open" if pf_first == me else None) if pf_level == 0 else \
                        "3bet" if pf_level == 1 else None
                if aggressive:
                    pf_level += 1
            if post and opp and last_aggr_by_us.get(st):
                self.fold_to_bet.add(a == "fold")
                self.raise_vs_bet.add(aggressive)
                last_aggr_by_us[st] = False
            if post and not opp and opp_bets and opp_bets[-1][0] == st and a in ("fold", "call"):
                frac = opp_bets[-1][1]
                b = "small" if frac < 0.55 else "big" if frac <= 1.05 else "over"
                self.we_fold_by_size[b].add(a == "fold")
            if post and opp and a != "fold":
                self.aggr.add(aggressive)
            if st == "flop" and opp and flop_first_opp is None:
                flop_first_opp = a
            if st == "flop" and opp and aggressive:
                opp_flop_bet = True
            if st == "flop" and opp and we_raised_flop_bet:
                self.fold_to_flop_raise.add(a == "fold")
                we_raised_flop_bet = False
            if st == "flop" and not opp and aggressive and opp_flop_bet:
                we_raised_flop_bet = True
            # chips
            if a == "call":
                pot += amt
                street_bet[who] = street_bet.get(who, 0) + amt
            elif a in ("raise", "all_in"):
                add = amt - street_bet.get(who, 0)
                if aggressive and post and opp:
                    opp_bets.append((st, (amt - cur_bet) / max(pot, 1)))
                    if cur_bet == 0 and a == "raise":
                        self.lead_sizes = (self.lead_sizes + [amt / max(pot, 1)])[-12:]
                pot += max(add, 0)
                street_bet[who] = amt
                cur_bet = max(cur_bet, amt)
            if aggressive:
                if st == "preflop":
                    pf_aggr = who
                elif not opp:
                    last_aggr_by_us[st] = True
        if pf_aggr is not None and pf_aggr != me and flop_first_opp is not None:
            self.cbet.add(flop_first_opp in ("raise", "all_in"))
        if shown and opp_bets:
            boards = self.boards.get(h, {})
            for st, frac in opp_bets:
                b = boards.get(st)
                if b and not (set(b) & set(shown)):
                    self.sizes.append((frac, made_pct(shown, b)))
            self.sizes = self.sizes[-40:]


STATS = OppStats()


def board_texture(board):
    """Rough texture: how high, how paired, how many draws."""
    ranks = sorted((RANKS.index(c[0]) for c in board), reverse=True)
    suits = [c[1] for c in board]
    t = Ctx()
    t.high = ranks[0]
    t.paired = len(set(ranks)) < len(ranks)
    t.flush_draw = max(suits.count(s) for s in SUITS) >= 2
    t.monotone = max(suits.count(s) for s in SUITS) >= 3
    u = sorted(set(ranks))
    conn = 0
    for i in range(len(u)):
        window = [r for r in u if u[i] <= r <= u[i] + 4]
        conn = max(conn, len(window))
    t.straighty = conn >= 3
    t.wet = t.monotone or (t.flush_draw and t.straighty) or conn >= 4
    t.dry = not t.flush_draw and not t.straighty
    # who the board favours: high, dry boards suit the preflop raiser's range
    t.raiser_favoured = t.high >= RANKS.index("Q") and not t.straighty
    t.caller_favoured = t.high <= RANKS.index("9") and t.straighty
    return t


def we_were_pf_aggressor(c):
    last = None
    for e in c.pf:
        if _aggressive(e, c.pf):
            last = e["bot_id"]
    return last == c.me


def postflop_equity(gs, c, bluff_keep=0.3):
    rng = opp_preflop_range(c, gs, push_range=_push_range(c) if c.eff_bb <= 15 else None)
    return mc_equity(gs["your_cards"], gs["community_cards"], rng, n_aggr=c.opp_post_aggr,
                     n_calls=c.opp_post_calls, bluff_keep=bluff_keep, budget=mc_budget())


def _push_range(c):
    push, _ = pushfold_nash(c.eff_bb, c.ante / c.bb)
    return push


PUSH_FOLD_BB = 12


def gto_preflop(gs, c, open_size=None, oop_flat=True, realize_oop=0.85):
    """The Nash bot's preflop: push/fold Nash when short, equilibrium-style
    ranges when deeper."""
    owed, pot, bb = gs["amount_owed"], gs["pot"], c.bb
    i, pct = c.cls, c.pct
    first_in = not c.pf and c.button
    short = c.eff_bb <= PUSH_FOLD_BB
    if short and first_in:
        push, _ = pushfold_nash(c.eff_bb, c.ante / bb)
        return {"action": "all_in"} if random.random() < push[i] else {"action": "fold"}
    if short and not c.button and c.opp_limped and owed == 0:
        if pct < 0.30 or (pct > 0.85 and random.random() < 0.15):
            return {"action": "all_in"}
        return {"action": "check"}
    if first_in:
        if pct < 0.78 or (pct < 0.88 and random.random() < 0.4):
            size = open_size() if open_size else (2.5 if c.eff_bb > 25 else 2.2)
            return act_raise_to(gs, c, size * bb)
        return {"action": "fold"}
    if not c.button and owed == 0 and c.opp_limped:
        if pct < 0.22 or (pct > 0.80 and random.random() < 0.18):
            return act_raise_to(gs, c, 4 * bb)
        return {"action": "check"}
    if owed == 0:
        return {"action": "check"}
    rng = opp_preflop_range(c, gs, push_range=_push_range(c) if c.eff_bb <= 15 else None)
    eq = eq_vs_range(i, rng)
    all_in_call = owed >= gs["your_stack"] or c.opp_stack == 0
    realize = 1.0 if all_in_call else (0.95 if c.button else realize_oop)
    need = owed / (pot + owed)
    level = c.pf_raises
    value_thr = {1: 0.60, 2: 0.57}.get(level, 0.55)
    dbg = {"eq_vs_range": round(eq, 3), "need": round(need, 3)}
    if c.can_raise and eq >= value_thr:
        mult = 3.2 if level == 1 else 2.3
        return {**act_raise_to(gs, c, gs["current_bet"] * mult), "debug": dbg}
    if c.can_raise and level == 1 and 0.12 < pct < 0.40 and random.random() < 0.22:
        return {**act_raise_to(gs, c, gs["current_bet"] * 3.2), "debug": {**dbg, "bluff3bet": 1}}
    if not oop_flat and not c.button and level >= 2 and not all_in_call:
        return {"action": "fold", "debug": {**dbg, "oop_no_flat": 1}}
    if eq * realize >= need:
        return {"action": "call", "debug": dbg}
    return {"action": "fold", "debug": dbg}


def baseline_postflop(gs, c, res=None, value_thr=0.68, bluff_scale=1.0, call_shift=0.0,
                      raise_thr=0.82, trap_p=0.2, trap_thr=0.9, bluff_raise_p=0.06, size=None, dbg=None):
    """The Nash bot's postflop, with knobs the exploit bots turn:
    bluff_scale  multiplies the balanced bluff frequency (0 = never bluff)
    call_shift   added to the pot-odds price we need to call (negative = call lighter)
    value_thr    equity needed to bet for value (lower = thinner value bets)
    size         pot fraction for our bets (default 0.6, 0.75 on the river)"""
    owed, pot = gs["amount_owed"], gs["pot"]
    res = res or postflop_equity(gs, c)
    eq = res.eq
    river = c.street == "river"
    r = random.random()
    dbg = {"eq": round(eq, 3), **(dbg or {})}
    if gs["can_check"]:
        s = size if size is not None else (0.75 if river else 0.6)
        if eq > value_thr:
            if eq > trap_thr and not river and r < trap_p:
                return {"action": "check", "debug": {**dbg, "trap": 1}}
            if c.eff / max(pot, 1) < 1.2:
                return {"action": "all_in", "debug": dbg}
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": dbg}
        bluff_p = (0.30 if river else 0.35) * bluff_scale
        weak = eq < 0.30 or (not river and eq < 0.48)
        if weak and r < min(bluff_p * (0.6 if eq < 0.3 and not river else 1), 0.95):
            return {**act_raise_to(gs, c, bet_pot_frac(gs, c, s)), "debug": {**dbg, "bluff": 1}}
        return {"action": "check", "debug": dbg}
    need = owed / (pot + owed) + call_shift
    if c.can_raise and eq > raise_thr:
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 0.8)), "debug": dbg}
    if c.can_raise and not river and eq < need * 0.7 and r < bluff_raise_p:
        return {**act_raise_to(gs, c, bet_pot_frac(gs, c, 0.8)), "debug": {**dbg, "bluffraise": 1}}
    if eq >= need - 0.02:
        return {"action": "call", "debug": dbg}
    return {"action": "fold", "debug": dbg}


def run(gs, preflop, postflop):
    """Shared decide(): warmup, stats, crash safety."""
    _T0[0] = time.time()
    if gs.get("type") == "warmup":
        presolve_pushfold(time.time() + 20)
        return {"action": "check"}
    try:
        STATS.update(gs)
        c = context(gs)
        return preflop(gs, c) if c.street == "preflop" else postflop(gs, c)
    except Exception as e:
        print("error:", repr(e))
        return act_fold(gs)
