"""Practice opponent: tight_value; parameter 0.72. Not a solver or GTO bot."""
import random
import eval7
STYLE = 'tight_value'
PARAMETER = 0.72
DECK = [eval7.Card(r+s) for r in '23456789TJQKA' for s in 'shdc']

def equity(s):
    hole = [eval7.Card(c) for c in s['your_cards']]
    board = [eval7.Card(c) for c in s['community_cards']]
    known = set(map(str, hole+board))
    deck = [c for c in DECK if str(c) not in known]
    score = 0
    for i in range(100):
        draw = random.sample(deck, 7-len(board))
        full = board+draw[2:]
        a, b = eval7.evaluate(hole+full), eval7.evaluate(draw[:2]+full)
        score += 1 if a > b else 0.5 if a == b else 0
    return score/100

def raise_to(s, amount):
    maximum = s['your_stack']+s['your_bet_this_street']
    target = max(s['min_raise_to'], int(amount))
    return {'action': 'all_in'} if target >= maximum else {'action': 'raise', 'amount': target}

def decide(s):
    if s.get('type') == 'warmup':
        return {'action': 'check'}
    owed = s['amount_owed']
    free = {'action': 'check'} if owed == 0 else {'action': 'fold'}
    if STYLE == 'always_fold':
        return free
    if STYLE == 'always_shove':
        return {'action': 'all_in'}
    e = equity(s)
    bb = s['blinds']['bb']
    price = min(owed, s['your_stack'])/max(1, s['pot']+min(owed, s['your_stack'])-max(0, owed-s['your_stack']))
    if STYLE == 'calling_station':
        return {'action': 'call'} if owed and (price < PARAMETER or e > 0.68) else free
    if STYLE == 'tight_value':
        if e > PARAMETER:
            return raise_to(s, s['current_bet']+max(bb, s['pot']*0.8))
        return {'action': 'call'} if owed and e > max(0.58, price+0.16) else free
    if STYLE == 'loose_aggressive':
        if random.random() < PARAMETER or e > 0.68:
            return raise_to(s, s['current_bet']+max(bb, s['pot']*0.75))
        return {'action': 'call'} if owed and e > price else free
    if STYLE == 'short_stack':
        total = s['your_stack']+s['your_bet_this_street']
        other = [p for p in s['players'] if p['bot_id'] != s['your_bot_id']][0]
        depth = min(total, other['stack']+other['bet_this_street'])/bb
        if depth <= PARAMETER and s['street'] == 'preflop':
            return {'action': 'all_in'} if e > 0.54 else free
        if e > 0.63:
            return raise_to(s, s['current_bet']+max(bb, s['pot']*0.65))
        return {'action': 'call'} if owed and e > price+0.08 else free
    if STYLE == 'trapper':
        if e > PARAMETER and s['street'] in ('turn', 'river'):
            return raise_to(s, s['current_bet']+max(bb, s['pot']))
        return {'action': 'call'} if owed and e > price+0.07 else free
    # Random mixed actions with equity guardrails. Deliberately NOT GTO.
    r = random.random()
    if e > 0.8 or r < PARAMETER:
        return raise_to(s, s['current_bet']+max(bb, s['pot']*0.5))
    if owed and (e > price+0.05 or r < 0.55):
        return {'action': 'call'}
    return free
