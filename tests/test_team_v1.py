"""Run: python3 tests/test_team_v1.py   (from the repo root, Python 3.10 env)"""
import sys, time, random
sys.path.insert(0, "my_bots/team_v1")
import bot

def st(**kw):
    s = dict(your_cards=["As", "Ah"], community_cards=[], street="preflop", pot=150,
             your_stack=10000, your_bet_this_street=50, amount_owed=0, can_check=True,
             current_bet=100, min_raise_to=200, your_bot_id="me", blinds={"sb": 50, "bb": 100, "ante": 0},
             match_action_log=[], action_log=[], hand_num=0)
    s.update(kw); return s

def test_equity_known_values():
    random.seed(1)
    assert 0.83 < bot.estimate_equity(["As", "Ah"], [], budget=0.3) < 0.87   # AA vs random ~85%
    assert 0.30 < bot.estimate_equity(["7d", "2c"], [], budget=0.3) < 0.40   # 72o ~35%
    assert bot.estimate_equity(["As", "Ks"], ["Qs", "Js", "Ts"], budget=0.3) > 0.99  # royal flush

def test_never_checks_when_owing():
    for owed in (50, 400, 3000):
        a = bot.decide(st(amount_owed=owed, can_check=False, your_cards=["7d", "2c"]))
        assert a["action"] != "check"

def test_raise_is_a_total_and_legal():
    a = bot.decide(st(your_bet_this_street=0, amount_owed=100, can_check=False, current_bet=100))
    assert a["action"] in ("raise", "all_in")
    if a["action"] == "raise":
        assert a["amount"] >= 200

def test_folds_trash_to_big_bet():
    a = bot.decide(st(your_cards=["7d", "2c"], community_cards=["As", "Kd", "9h"], street="flop",
                      pot=2000, amount_owed=1500, can_check=False, current_bet=1500, min_raise_to=3000,
                      your_bet_this_street=0))
    assert a["action"] == "fold"

def test_speed():
    t = time.time(); bot.decide(st(your_cards=["Qh", "Jd"], community_cards=["2c", "9s", "Td"], street="flop")); 
    assert time.time() - t < 1.0

def test_fold_tracking():
    log = []
    for h in range(20):
        log += [{"hand_num": h, "bot_id": "me", "action": "raise"}, {"hand_num": h, "bot_id": "opp", "action": "fold"}]
    rate, n = bot.opponent_fold_rate(st(match_action_log=log))
    assert rate == 1.0 and n == 20

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
