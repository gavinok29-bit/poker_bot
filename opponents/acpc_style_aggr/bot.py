"""Practice opponent acpc_style_aggr: ACPC-sample-shaped deep-stack preflop, simple postflop. Not a solver."""
import random
import eval7
CBET = 0.58
THREE_BET_EXTRA = 0.05   # extra 3-bet probability for suited-connector style bluffs
LIMP_SHARE = 0.05         # share of marginal button hands limped instead of raised
DECK = [eval7.Card(r+s) for r in '23456789TJQKA' for s in 'shdc']
RANKS = '23456789TJQKA'
BUTTON_FOLD = set('83o 72o 92o 32o 93o 42o 82o 62o 73o 52o T2o'.split())   # folded ~91-100% in sample
BUTTON_LIMP = set('94o 84o Q2o 63o T3o J2o'.split())                          # limp-heavy in sample
BB_FOLD = set('32o 82o 73o 92o 42o 72o 62o 94o 93o 83o 52o T2o 84o T3o 63o'.split())  # folded 92-100%
BB_MOSTLY_FOLD = set('J2o T4o T5o 74o Q2o'.split())                           # folded ~70-80%
PREMIUM = set('AA KK QQ JJ AKs AKo AQs KQs KJs'.split())
BLUFF3 = set('76s 65s A5s A4s'.split())

def cls(cards):
    a, b = sorted((cards[0][0], cards[1][0]), key=RANKS.index, reverse=True)
    if a == b:
        return a+b
    return a+b+('s' if cards[0][1] == cards[1][1] else 'o')

def equity(s):
    hole = [eval7.Card(c) for c in s['your_cards']]
    board = [eval7.Card(c) for c in s['community_cards']]
    known = set(map(str, hole+board))
    deck = [c for c in DECK if str(c) not in known]
    score = 0
    for i in range(100):
        d = random.sample(deck, 7-len(board))
        full = board+d[2:]
        a, b = eval7.evaluate(hole+full), eval7.evaluate(d[:2]+full)
        score += 1 if a > b else 0.5 if a == b else 0
    return score/100

def raise_to(s, amount):
    maximum = s['your_stack']+s['your_bet_this_street']
    target = max(s['min_raise_to'], int(amount))
    return {'action': 'all_in'} if target >= maximum else {'action': 'raise', 'amount': target}

def decide(s):
    if s.get('type') == 'warmup':
        return {'action': 'check'}
    owed, bb = s['amount_owed'], s['blinds']['bb']
    free = {'action': 'check'} if owed == 0 else {'action': 'fold'}
    other = [p for p in s['players'] if p['bot_id'] != s['your_bot_id']][0]
    mine = s['your_stack']+s['your_bet_this_street']
    depth = min(mine, other['stack']+other['bet_this_street'])/bb
    price = min(owed, s['your_stack'])/max(1, s['pot']+min(owed, s['your_stack']))
    if s['street'] == 'preflop':
        if depth <= 25:      # our own assumption: the sample has no short stacks
            e = equity(s)
            return {'action': 'all_in'} if e > (0.50 if owed else 0.53) else free
        c = cls(s['your_cards'])
        on_button = s['your_bot_id'] == [p for p in s['players'] if p['seat'] == s['sb_seat']][0]['bot_id']
        if on_button and s['current_bet'] <= bb:                       # first to act
            if c in BUTTON_FOLD:
                return {'action': 'fold'}
            if c in BUTTON_LIMP or (c not in PREMIUM and random.random() < LIMP_SHARE):
                return {'action': 'call'}
            return raise_to(s, bb*random.choice([2, 2.5, 2.5, 3]))
        if on_button:                                                    # facing a 3-bet
            e = equity(s)
            return {'action': 'call'} if e > price+0.08 else free
        if s['current_bet'] <= bb:                                       # BB facing a limp
            return raise_to(s, bb*3) if random.random() < 0.35 else {'action': 'check'}
        if s['current_bet'] > 4*bb:                                      # BB facing a 3-bet+ (rare)
            return {'action': 'call'} if c in PREMIUM else free
        if c in BB_FOLD or (c in BB_MOSTLY_FOLD and random.random() < 0.75):
            return {'action': 'fold'}
        if c in PREMIUM or (c in BLUFF3 and random.random() < 0.9) or random.random() < THREE_BET_EXTRA:
            return raise_to(s, s['current_bet']*3.5)
        return {'action': 'call'}
    e = equity(s)
    if owed == 0:
        first = s['street'] == 'flop' and any(a['bot_id'] == s['your_bot_id'] and a['action'] == 'raise'
                                              and a['street'] == 'preflop' for a in s['action_log'])
        if e > 0.68 or (first and random.random() < CBET) or (e > 0.5 and random.random() < 0.3):
            return raise_to(s, s['current_bet']+max(bb, s['pot']*0.6))
        return free
    if e > 0.82:
        return raise_to(s, s['current_bet']+max(bb, s['pot']*0.8))
    return {'action': 'call'} if e > price+0.05 else free
