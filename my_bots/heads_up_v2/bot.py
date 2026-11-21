"""Heads-up v2: bounded random-hand equity and explicit, explainable rules.
No persistent training, solver, LLM, network, writes or hidden-card access.
Read START_HERE.md for limitations and the reason behind each rule.
"""
import random
import time
import eval7

SAMPLES = 700
BUDGET = 0.12
DECK = [eval7.Card(r+s) for r in '23456789TJQKA' for s in 'shdc']
_CACHE = {}


def estimate_equity(hole, board, budget=BUDGET, rng=random):
    """Random-hand showdown equity, not equity versus a fitted betting range.
    Fixed sample cap normally makes seeded matches reproducible. The emergency
    wall-clock stop can change results on an overloaded machine. No hard-timeout
    promise is made. Incomplete estimates are deliberately not cached.
    """
    key = (tuple(sorted(hole)), tuple(sorted(board)))
    if key in _CACHE:
        return _CACHE[key]
    h, b = [eval7.Card(c) for c in hole], [eval7.Card(c) for c in board]
    known = set(map(str, h+b))
    deck = [c for c in DECK if str(c) not in known]
    score = 0.0
    n = 0
    start = time.perf_counter()
    for i in range(SAMPLES):
        if n and i % 16 == 0 and time.perf_counter()-start >= budget:
            break
        draw = rng.sample(deck, 7-len(b))
        full = b+draw[2:]
        a, c = eval7.evaluate(h+full), eval7.evaluate(draw[:2]+full)
        score += 1 if a > c else 0.5 if a == c else 0
        n += 1
    eq = score/max(1, n)
    if n == SAMPLES:
        if len(_CACHE) >= 2048:
            _CACHE.clear()
        _CACHE[key] = eq
    return eq


def opponent_stats(s):
    """Only the visible recent log. No cumulative double-counting across calls."""
    pending = {}
    folds = replies = aggressive = actions = 0
    highest = {}
    for e in s.get('match_action_log') or []:
        a = e.get('action')
        key = (e.get('hand_num'), e.get('street'))
        old_high = highest.get(key, 0)
        is_agg = a == 'raise' or (a == 'all_in' and e.get('amount', 0) > old_high)
        if a in ('small_blind', 'big_blind', 'raise', 'all_in'):
            highest[key] = max(old_high, e.get('amount', 0))
        if a not in ('raise', 'all_in', 'call', 'check', 'fold'):
            continue
        if e.get('bot_id') == s['your_bot_id']:
            pending[key] = is_agg
        else:
            actions += 1
            aggressive += is_agg
            if pending.get(key):
                replies += 1
                folds += a == 'fold'
                pending[key] = False
    return (folds/replies if replies else 0.5, replies,
            aggressive/actions if actions else 0, actions)


def opponent(s):
    return next((p for p in s.get('players', [])
                 if p.get('bot_id') != s['your_bot_id'] and not p.get('is_folded')
                 and p.get('state') != 'busted'), None)


def effective_bb(s):
    """Stack at start of this street, including live street commitments.
    Antes are dead money. Cap by the opponent, not just our own remaining chips.
    """
    total = s['your_stack']+s['your_bet_this_street']
    p = opponent(s)
    if p is not None:
        total = min(total, p['stack']+p.get('bet_this_street', 0))
    return total/max(1, s['blinds']['bb'])


def call_price(s):
    """Exclude an uncallable part of an oversized heads-up all-in from pot odds."""
    call = min(s['amount_owed'], s['your_stack'])
    uncalled = max(0, s['amount_owed']-call)
    return call/max(1, s['pot']-uncalled+call)


def safe(s):
    return {'action': 'check' if s.get('amount_owed', 0) <= 0 else 'fold'}


def raise_to(s, target):
    p = opponent(s)
    if p is not None and (p.get('is_all_in') or p['stack'] == 0):
        return {'action': 'call'} if s['amount_owed'] else safe(s)
    maximum = s['your_stack']+s['your_bet_this_street']
    target = max(s['min_raise_to'], int(target))
    if target >= maximum*0.90:
        return {'action': 'all_in'}
    return {'action': 'raise', 'amount': min(target, maximum)}


def unopened(s):
    """The small blind owes half a blind, but that is NOT an opponent raise."""
    if s['street'] != 'preflop' or s['current_bet'] > s['blinds']['bb']:
        return False
    return not any(e.get('street') == 'preflop' and
                   e.get('action') in ('raise', 'all_in')
                   for e in s.get('action_log') or [])


def has_draw(s):
    cards = s['your_cards']+s['community_cards']
    if len(s['community_cards']) not in (3, 4):
        return False
    # Require that our hole cards contribute. A wet board alone is not our draw.
    flush = any(sum(c[1] == suit for c in cards) == 4 and
                any(c[1] == suit for c in s['your_cards']) for suit in 'shdc')
    ranks = set('23456789TJQKA'.index(c[0])+2 for c in cards)
    if 14 in ranks:
        ranks.add(1)
    hole = set('23456789TJQKA'.index(c[0])+2 for c in s['your_cards'])
    if 14 in hole:
        hole.add(1)
    board_ranks = set('23456789TJQKA'.index(c[0])+2 for c in s['community_cards'])
    if 14 in board_ranks:
        board_ranks.add(1)
    straight = any(len(ranks & set(range(lo, lo+5))) == 4 and
                   bool((hole-board_ranks) & set(range(lo, lo+5))) for lo in range(1, 11))
    return flush or straight


def choose(s, eq, stats=(0.5, 0, 0, 0), rng=random):
    fold, n, aggression, na = stats
    bb = s['blinds']['bb']
    owed = s['amount_owed']
    pre = s['street'] == 'preflop'
    eff = effective_bb(s)
    p = opponent(s)
    can_raise = p is None or (p['stack'] > 0 and not p.get('is_all_in'))
    price = call_price(s)
    r = rng.random()

    # Deal with unopened pots BEFORE general facing-a-bet logic.
    if pre and unopened(s) and can_raise and owed > 0:
        if eff <= 12:
            # Heuristic push threshold, NOT a Nash push/fold chart.
            threshold = max(0.44, 0.57-(12-eff)*0.012)
            if eq >= threshold:
                return {'action': 'all_in'}
            return safe(s)
        if eq >= 0.48:
            return raise_to(s, 2.5*bb)
        return safe(s)

    if owed > 0:
        # Known limitation: a fixed discount approximates a stronger betting range.
        discount = (min(0.15, 0.035+0.012*s['current_bet']/bb) if pre
                    else min(0.17, 0.05+0.07*owed/max(1, s['pot']-owed)))
        if na >= 16 and aggression >= 0.60:
            discount *= 0.25
        adj = max(0, eq-discount)
        # If we cannot raise, only compare the callable pot to our actual call.
        if not can_raise or owed >= s['your_stack']:
            return {'action': 'call'} if adj > price+0.02 else {'action': 'fold'}
        if adj >= (0.68 if pre else 0.76):
            if eff <= 12:
                return {'action': 'all_in'}
            return raise_to(s, max(s['current_bet']*3,
                            s['current_bet']+0.7*(s['pot']+owed)))
        if adj > price+0.04:
            return {'action': 'call'}
        return {'action': 'fold'}

    if not can_raise:
        return safe(s)
    if pre:  # big blind checking option after a limp, or a completed short raise
        if eff <= 12:
            return {'action': 'all_in'} if eq >= 0.57 else safe(s)
        return raise_to(s, 3*bb) if eq > 0.61 else safe(s)
    if eq > 0.77:
        return raise_to(s, s['your_bet_this_street']+s['pot']*0.70)
    if eq > 0.63:
        return raise_to(s, s['your_bet_this_street']+s['pot']*0.50)
    # Small semi-bluff frequency, reduced versus observed calling stations.
    bluff = 0.08
    if n >= 12:
        bluff = max(0.01, min(0.16, 0.08+(fold-0.45)*0.25))
    if eq < 0.5 and has_draw(s) and r < bluff:
        return raise_to(s, s['your_bet_this_street']+s['pot']*0.50)
    return safe(s)


def decide(game_state):
    if game_state.get('type') == 'warmup':
        return {'action': 'check'}
    try:
        # The runner owns the hard timeout. Leave a large headroom for its IPC,
        # JSON, history handling and scheduling on a half-core container.
        budget = min(BUDGET, max(0.001, game_state.get('time_limit_ms', 2000)/1000*0.12))
        eq = estimate_equity(game_state['your_cards'], game_state['community_cards'], budget)
        act = choose(game_state, eq, opponent_stats(game_state))
        if act['action'] == 'check' and game_state['amount_owed'] > 0:
            act = safe(game_state)
        act['debug'] = {'random_hand_equity': round(eq, 3),
                        'effective_bb': round(effective_bb(game_state), 2)}
        return act
    except Exception as exc:
        act = safe(game_state)
        act['debug'] = {'fallback': type(exc).__name__}
        return act
