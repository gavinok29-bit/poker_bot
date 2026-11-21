########## analyze.py (per-bot action frequencies, c-bet) ##########
import re, os, glob
from collections import defaultdict, Counter

files = sorted(glob.glob('/tmp/acpc/*.txt'))
act_re = re.compile(r'^(.+?): (folds|checks|calls \$([\d.]+)|bets \$([\d.]+)|raises \$[\d.]+ to \$([\d.]+))')
seat_re = re.compile(r'^Seat (\d+): (.+?) \(\$([\d.]+) in chips\)')
btn_re = re.compile(r"Seat #(\d+) is the button")

players = ['Tartanian7','Prelude','Hyperborean_iro','Slumbot']
stats = {p: defaultdict(int) for p in players}
open_sizes = {p: Counter() for p in players}
stacks = Counter()
total_hands = 0
per_file_hands = Counter()

def bucket(to_amt):
    if to_amt <= 2.01: return 'minraise 2bb'
    if to_amt <= 2.51: return '2.5bb'
    if to_amt <= 3.01: return '3bb'
    if to_amt <= 4.01: return '4bb'
    if to_amt <= 10.01: return '5-10bb'
    return 'shove >10bb'

def process_hand(lines):
    global total_hands
    btn = None; seats = {}
    street = None
    preflop, flop = [], []
    for ln in lines:
        m = seat_re.match(ln)
        if m: seats[m.group(1)] = m.group(2); stacks[float(m.group(3))] += 1; continue
        m = btn_re.search(ln)
        if m: btn = m.group(1); continue
        if ln.startswith('*** HOLE CARDS'): street = 'pre'; continue
        if ln.startswith('*** FLOP'): street = 'flop'; continue
        if ln.startswith('*** TURN') or ln.startswith('*** RIVER') or ln.startswith('*** SUMMARY'): street = 'other'; continue
        m = act_re.match(ln)
        if m and street in ('pre','flop'):
            name = m.group(1)
            if m.group(3) is not None: a = ('call', float(m.group(3)))
            elif m.group(4) is not None: a = ('bet', float(m.group(4)))
            elif m.group(5) is not None: a = ('raise', float(m.group(5)))
            else: a = (m.group(2)[:4], None)  # folds/checks -> 'fold'/'chec'
            (preflop if street=='pre' else flop).append((name, a))
    if len(seats) != 2 or btn is None: return
    total_hands += 1
    names = list(seats.values())
    button = seats[btn]
    bb = [n for s,n in seats.items() if n != button][0]
    for n in names:
        if n in stats: stats[n]['hands'] += 1
    # preflop analysis
    if preflop:
        # button first action
        bacts = [a for n,a in preflop if n == button]
        bbacts = [a for n,a in preflop if n == bb]
        if bacts and button in stats:
            s = stats[button]
            first = bacts[0]
            if first[0] == 'fold': s['btn_fold'] += 1
            elif first[0] == 'call': s['btn_limp'] += 1
            elif first[0] == 'raise':
                s['btn_raise'] += 1
                open_sizes[button][bucket(first[1])] += 1
        # VPIP/PFR
        vpiped, pfrd = set(), set()
        for n,a in preflop:
            if a[0] in ('call','raise'):
                vpiped.add(n)
                if a[0]=='raise': pfrd.add(n)
        for n in vpiped:
            if n in stats: stats[n]['vpip'] += 1
        for n in pfrd:
            if n in stats: stats[n]['pfr'] += 1
        # BB response to button open raise
        if bacts and bacts[0][0]=='raise' and bbacts and bb in stats:
            s = stats[bb]; r = bbacts[0]
            if r[0]=='fold': s['bb_vs_open_fold'] += 1
            elif r[0]=='call': s['bb_vs_open_call'] += 1
            elif r[0]=='raise': s['bb_vs_open_3bet'] += 1
        # BB vs limp
        if bacts and bacts[0][0]=='call' and bbacts and bb in stats:
            s = stats[bb]
            if bbacts[0][0]=='chec': s['bb_vs_limp_check'] += 1
            elif bbacts[0][0]=='raise': s['bb_vs_limp_raise'] += 1
    # flop cbet: preflop aggressor = last raiser preflop
    raisers = [n for n,a in preflop if a[0]=='raise']
    if raisers and flop:
        agg = raisers[-1]
        if agg in stats:
            opp = bb if agg==button else button
            for n,a in flop:
                if n == opp and a[0] in ('bet','raise'):
                    stats[agg]['flop_faced_donk'] += 1; break
                if n == agg:
                    if a[0] in ('bet','raise'): stats[agg]['flop_cbet'] += 1
                    elif a[0]=='chec': stats[agg]['flop_check'] += 1
                    else: stats[agg]['flop_other'] += 1
                    break

for f in files:
    hand = []
    with open(f, errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('PokerStars Hand #'):
                if hand: process_hand(hand)
                hand = [line]
            elif hand is not None:
                hand.append(line)
    if hand: process_hand(hand)

print(f"TOTAL HANDS PARSED: {total_hands}")
print(f"STACK SIZES SEEN: {dict(stacks)}")
print()
for p in players:
    s = stats[p]; h = s['hands']
    if not h: continue
    print(f"=== {p} ({h} hands) ===")
    print(f"  VPIP {100*s['vpip']/h:.1f}%  PFR {100*s['pfr']/h:.1f}%")
    bt = s['btn_fold']+s['btn_limp']+s['btn_raise']
    if bt:
        print(f"  Button first action: fold {100*s['btn_fold']/bt:.1f}% | limp {100*s['btn_limp']/bt:.1f}% | raise {100*s['btn_raise']/bt:.1f}%")
        osz = open_sizes[p]; tot = sum(osz.values())
        print(f"  Button open sizes: " + ", ".join(f"{k}: {100*v/tot:.1f}%" for k,v in sorted(osz.items(), key=lambda x:-x[1])))
    bb = s['bb_vs_open_fold']+s['bb_vs_open_call']+s['bb_vs_open_3bet']
    if bb:
        print(f"  BB vs open: fold {100*s['bb_vs_open_fold']/bb:.1f}% | call {100*s['bb_vs_open_call']/bb:.1f}% | 3-bet {100*s['bb_vs_open_3bet']/bb:.1f}%")
    bl = s['bb_vs_limp_check']+s['bb_vs_limp_raise']
    if bl:
        print(f"  BB vs limp: check {100*s['bb_vs_limp_check']/bl:.1f}% | raise {100*s['bb_vs_limp_raise']/bl:.1f}%")
    fl = s['flop_cbet']+s['flop_check']
    if fl:
        print(f"  Flop c-bet (as preflop aggressor, no donk): {100*s['flop_cbet']/fl:.1f}%  (faced donk {s['flop_faced_donk']} times)")
    print()

########## analyze2.py (per-hand-class ranges) ##########
import re, glob
from collections import defaultdict, Counter

files = sorted(glob.glob('/tmp/acpc/*.txt'))
act_re = re.compile(r'^(.+?): (folds|checks|calls \$([\d.]+)|bets \$([\d.]+)|raises \$[\d.]+ to \$([\d.]+))')
seat_re = re.compile(r'^Seat (\d+): (.+?) \(\$([\d.]+) in chips\)')
btn_re = re.compile(r"Seat #(\d+) is the button")
show_re = re.compile(r'^(.+?): (?:shows|mucks) \[(\w\w) (\w\w)\]')
RANKS = '23456789TJQKA'

def canon(c1, c2):
    r1, s1, r2, s2 = c1[0], c1[1], c2[0], c2[1]
    if RANKS.index(r1) < RANKS.index(r2): r1, r2 = r2, r1
    if r1 == r2: return r1 + r2
    return r1 + r2 + ('s' if s1 == s2 else 'o')

# per hand class, aggregated over the 4 bots:
btn_act = defaultdict(Counter)      # hand class -> fold/limp/raise counts (button first action)
bb_vs_open = defaultdict(Counter)   # hand class -> fold/call/3bet when BB facing a button open

def process_hand(lines):
    btn = None; seats = {}; street = None
    preflop = []; shown = {}
    for ln in lines:
        m = seat_re.match(ln)
        if m: seats[m.group(1)] = m.group(2); continue
        m = btn_re.search(ln)
        if m: btn = m.group(1); continue
        if ln.startswith('*** HOLE CARDS'): street='pre'; continue
        if ln.startswith('*** FLOP') or ln.startswith('*** SUMMARY'): street='other'; continue
        m = show_re.match(ln)
        if m: shown[m.group(1)] = canon(m.group(2), m.group(3)); continue
        m = act_re.match(ln)
        if m and street=='pre':
            a = ('raise' if m.group(5) is not None else 'call' if m.group(3) is not None else m.group(2)[:4])
            preflop.append((m.group(1), a))
    if len(seats)!=2 or btn is None: return
    button = seats[btn]; bb = [n for s,n in seats.items() if n!=button][0]
    bacts = [a for n,a in preflop if n==button]
    bbacts = [a for n,a in preflop if n==bb]
    if button in shown and bacts:
        act = 'fold' if bacts[0]=='fold' else 'limp' if bacts[0]=='call' else 'raise'
        btn_act[shown[button]][act] += 1
    if bb in shown and bacts and bacts[0]=='raise' and bbacts:
        r = bbacts[0]
        bb_vs_open[shown[bb]]['fold' if r=='fold' else 'call' if r=='call' else '3bet'] += 1

for f in files:
    hand = []
    with open(f, errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('PokerStars Hand #'):
                if hand: process_hand(hand)
                hand=[line]
            elif hand is not None:
                hand.append(line)
    if hand: process_hand(hand)

print("== BUTTON strategy by hand class (all 4 bots pooled, classes with 300+ samples) ==")
rows=[]
for hc,c in btn_act.items():
    t=sum(c.values())
    if t>=300: rows.append((c['fold']/t, c['limp']/t, c['raise']/t, t, hc))
rows.sort()
print(f"{'hand':5} {'fold%':>6} {'limp%':>6} {'raise%':>6} {'n':>6}")
print("-- most-folded 25 --")
for f,l,r,t,hc in rows[-25:][::-1]: print(f"{hc:5} {100*f:6.1f} {100*l:6.1f} {100*r:6.1f} {t:6}")
print("-- never/rarely folded (15 lowest fold%) --")
for f,l,r,t,hc in rows[:15]: print(f"{hc:5} {100*f:6.1f} {100*l:6.1f} {100*r:6.1f} {t:6}")
print("-- most-limped 12 --")
rows.sort(key=lambda x:-x[1])
for f,l,r,t,hc in rows[:12]: print(f"{hc:5} {100*f:6.1f} {100*l:6.1f} {100*r:6.1f} {t:6}")
tot=sum(sum(c.values()) for c in btn_act.values())
print(f"total button decisions with cards: {tot}")

print()
print("== BB vs button open, by hand class (300+ samples) ==")
rows=[]
for hc,c in bb_vs_open.items():
    t=sum(c.values())
    if t>=300: rows.append((c['fold']/t, c['call']/t, c['3bet']/t, t, hc))
rows.sort()
print(f"{'hand':5} {'fold%':>6} {'call%':>6} {'3bet%':>6} {'n':>6}")
print("-- most-folded 20 --")
for f,c3,b,t,hc in rows[-20:][::-1]: print(f"{hc:5} {100*f:6.1f} {100*c3:6.1f} {100*b:6.1f} {t:6}")
print("-- most-3bet 15 --")
rows.sort(key=lambda x:-x[2])
for f,c3,b,t,hc in rows[:15]: print(f"{hc:5} {100*f:6.1f} {100*c3:6.1f} {100*b:6.1f} {t:6}")

########## audit.py (denominators, duplicates, per-bot robustness, spot checks) ##########
import re, glob, random
from collections import defaultdict, Counter

files = sorted(glob.glob('/tmp/acpc/*.txt'))
hand_re = re.compile(r'^PokerStars Hand #(\d+)')
act_re = re.compile(r'^(.+?): (folds|checks|calls \$([\d.]+)|bets \$([\d.]+)|raises \$[\d.]+ to \$([\d.]+))')
seat_re = re.compile(r'^Seat (\d+): (.+?) \(\$([\d.]+) in chips\)')
btn_re = re.compile(r"Seat #(\d+) is the button")
show_re = re.compile(r'^(.+?): (?:shows|mucks) \[(\w\w) (\w\w)\]')
blind_re = re.compile(r"Hold'em No Limit \(\$([\d.]+)/\$([\d.]+)")
RANKS='23456789TJQKA'
def canon(c1,c2):
    r1,s1,r2,s2 = c1[0],c1[1],c2[0],c2[1]
    if RANKS.index(r1)<RANKS.index(r2): r1,r2=r2,r1
    return r1+r2 if r1==r2 else r1+r2+('s' if s1==s2 else 'o')

players=['Tartanian7','Prelude','Hyperborean_iro','Slumbot']
ids_by_file={}
blinds=Counter()
parsed=[]  # (file, hid, button, bb, preflop actions, shown)

for f in files:
    fname=f.split('/')[-1]
    ids=[]; hand=[]
    def flush(hand):
        if not hand: return
        hid=None; btn=None; seats={}; street=None; preflop=[]; shown={}
        for ln in hand:
            m=hand_re.match(ln)
            if m: hid=m.group(1); continue
            m=blind_re.search(ln)
            if m: blinds[(m.group(1),m.group(2))]+=1; continue
            m=seat_re.match(ln)
            if m: seats[m.group(1)]=m.group(2); continue
            m=btn_re.search(ln)
            if m: btn=m.group(1); continue
            if ln.startswith('*** HOLE CARDS'): street='pre'; continue
            if ln.startswith('*** FLOP') or ln.startswith('*** SUMMARY'): street='other'; continue
            m=show_re.match(ln)
            if m: shown[m.group(1)]=canon(m.group(2),m.group(3)); continue
            m=act_re.match(ln)
            if m and street=='pre':
                if m.group(5) is not None: a=('raise',float(m.group(5)))
                elif m.group(3) is not None: a=('call',float(m.group(3)))
                else: a=(m.group(2)[:4],None)
                preflop.append((m.group(1),a))
        if len(seats)!=2 or btn is None: return
        button=seats[btn]; bb=[n for s,n in seats.items() if n!=button][0]
        ids.append(hid)
        parsed.append((fname,hid,button,bb,preflop,shown))
    with open(f,errors='replace') as fh:
        for line in fh:
            line=line.rstrip('\n')
            if line.startswith('PokerStars Hand #'):
                flush(hand); hand=[line]
            elif hand is not None:
                hand.append(line)
        flush(hand)
    ids_by_file[fname]=ids

print("== A. DUPLICATE / PERSPECTIVE ACCOUNTING ==")
all_ids=[i for ids in ids_by_file.values() for i in ids]
print(f"total hand records parsed: {len(all_ids)}; unique hand IDs: {len(set(all_ids))}")
# perspective-pair overlap test
pairs=[('Hyperborean_iro_vs_Prelude-15.txt','Prelude_vs_Hyperborean_iro-15.txt'),
       ('Slumbot_vs_Tartanian7-01.txt','Tartanian7_vs_Slumbot-01.txt')]
for a,b in pairs:
    if a in ids_by_file and b in ids_by_file:
        sa,sb=set(ids_by_file[a]),set(ids_by_file[b])
        print(f"  {a} ({len(sa)}) vs {b} ({len(sb)}): shared IDs={len(sa&sb)}")
# global ID collision count
c=Counter(all_ids)
dup=sum(1 for v in c.values() if v>1)
print(f"  IDs appearing in more than one record: {dup} (max occurrences: {max(c.values())})")
print(f"  blinds seen: {dict(blinds)}")

print()
print("== B. PER-BOT, PER-SEAT COUNTS (raw) ==")
st={p:defaultdict(int) for p in players}
btn_cls={p:defaultdict(Counter) for p in players}
bb3_cls={p:defaultdict(Counter) for p in players}
for fname,hid,button,bb,preflop,shown in parsed:
    for n in (button,bb):
        if n in st: st[n]['hands']+=1
    bacts=[a for n,a in preflop if n==button]
    bbacts=[a for n,a in preflop if n==bb]
    # button
    if button in st:
        st[button]['btn_hands']+=1
        if bacts:
            a=bacts[0]
            if a[0]=='fold': st[button]['btn_fold']+=1
            elif a[0]=='call': st[button]['btn_limp']+=1
            elif a[0]=='raise': st[button]['btn_raise']+=1
            if button in shown:
                cls=shown[button]
                act='fold' if a[0]=='fold' else 'limp' if a[0]=='call' else 'raise'
                btn_cls[button][cls][act]+=1
    # BB VPIP: any voluntary preflop call/raise by bb
    if bb in st:
        st[bb]['bb_hands']+=1
        if any(a[0] in ('call','raise') for a in bbacts): st[bb]['bb_vpip']+=1
        if bacts and bacts[0][0]=='raise' and bbacts:
            r=bbacts[0]
            cat='fold' if r[0]=='fold' else 'call' if r[0]=='call' else '3bet'
            st[bb]['bb_open_'+cat]+=1
            if bb in shown: bb3_cls[bb][shown[bb]][cat]+=1
        if bacts and bacts[0][0]=='call' and bbacts:
            st[bb]['bb_limp_' + ('raise' if bbacts[0][0]=='raise' else 'check' if bbacts[0][0]=='chec' else 'other')]+=1

print(f"{'bot':16} {'hands':>7} {'btnN':>7} {'btnR':>6} {'btnL':>6} {'btnF':>6} {'btnVPIP%':>8} {'bbN':>7} {'bbVPIP%':>8} {'overallVPIP%':>12}")
for p in players:
    s=st[p]; h=s['hands']
    btnv=s['btn_raise']+s['btn_limp']
    ov=100*(btnv+s['bb_vpip'])/h
    print(f"{p:16} {h:7} {s['btn_hands']:7} {s['btn_raise']:6} {s['btn_limp']:6} {s['btn_fold']:6} {100*btnv/s['btn_hands']:8.1f} {s['bb_hands']:7} {100*s['bb_vpip']/s['bb_hands']:8.1f} {ov:12.1f}")
    o=s['bb_open_fold']+s['bb_open_call']+s['bb_open_3bet']
    l=s['bb_limp_raise']+s['bb_limp_check']+s['bb_limp_other']
    print(f"   BB vs open n={o}: fold {s['bb_open_fold']} call {s['bb_open_call']} 3bet {s['bb_open_3bet']} | BB vs limp n={l}: raise {s['bb_limp_raise']} check {s['bb_limp_check']} other {s['bb_limp_other']}")

print()
print("== C. PER-BOT robustness of claimed ranges ==")
print("button always-fold classes, per bot (fold/n):")
for cls in ['83o','72o','92o','32o','93o','42o','82o','62o']:
    row=[]
    for p in players:
        c=btn_cls[p][cls]; t=sum(c.values()); row.append(f"{p.split('_')[0]}:{c['fold']}/{t}")
    print(f"  {cls:4} "+"  ".join(row))
print("BB 3-bet classes, per bot (3bet/n):")
for cls in ['AA','KK','AKs','AKo','AQs','76s','65s']:
    row=[]
    for p in players:
        c=bb3_cls[p][cls]; t=sum(c.values()); row.append(f"{p.split('_')[0]}:{c['3bet']}/{t}")
    print(f"  {cls:4} "+"  ".join(row))

print()
print("== D. MANUAL SPOT CHECKS (3 random hands: raw text vs parsed) ==")
random.seed(7)
byid={}
for rec in parsed: byid.setdefault(rec[1], rec)
for hid in random.sample(list(byid), 3):
    fname,h,button,bb,preflop,shown=byid[hid]
    print(f"--- hand {hid} in {fname}: button={button}, bb={bb}")
    print(f"    parsed preflop actions: {preflop}")
    print(f"    parsed shown cards: {shown}")
