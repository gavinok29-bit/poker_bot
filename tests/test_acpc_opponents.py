import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'opponents'/name/'bot.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def st(cards, sb_is_me, **kw):
    s = dict(your_cards=cards, community_cards=[], street='preflop', pot=75,
             your_stack=9975 if sb_is_me else 9950, your_bet_this_street=25 if sb_is_me else 50,
             amount_owed=25 if sb_is_me else 0, can_check=not sb_is_me, current_bet=50, min_raise_to=100,
             your_bot_id='me', blinds={'sb': 25, 'bb': 50, 'ante': 0}, action_log=[], sb_seat=0 if sb_is_me else 1,
             players=[{'bot_id': 'me', 'seat': 0, 'stack': 9975, 'bet_this_street': 25},
                      {'bot_id': 'other', 'seat': 1, 'stack': 9950, 'bet_this_street': 50}])
    s.update(kw)
    return s

class AcpcStyleTests(unittest.TestCase):
    def test_button_folds_sample_trash_and_raises_premium(self):
        for name in ('acpc_style_aggr', 'acpc_style_mid', 'acpc_style_passive'):
            m = load(name)
            self.assertEqual(m.decide(st(['8s', '3h'], True))['action'], 'fold')
            self.assertEqual(m.decide(st(['As', 'Ah'], True))['action'], 'raise')

    def test_bb_folds_trash_and_threebets_premium_vs_open(self):
        m = load('acpc_style_mid')
        open_raise = dict(current_bet=125, amount_owed=75, can_check=False, pot=175,
                          action_log=[], min_raise_to=200)
        self.assertEqual(m.decide(st(['7s', '2h'], False, **open_raise))['action'], 'fold')
        self.assertEqual(m.decide(st(['As', 'Ks'], False, **open_raise))['action'], 'raise')

    def test_shallow_stacks_use_push_fold_assumption(self):
        m = load('acpc_style_mid')
        s = st(['As', 'Ah'], True, your_stack=475, blinds={'sb': 25, 'bb': 50, 'ante': 0})
        s['players'][1]['stack'] = 450
        self.assertEqual(m.decide(s)['action'], 'all_in')

if __name__ == '__main__':
    unittest.main()
