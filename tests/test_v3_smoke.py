import importlib.util
from pathlib import Path
import unittest
from tests.test_heads_up import state

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v3', ROOT/'my_bots/heads_up_v3/bot.py')
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

class V3Smoke(unittest.TestCase):
    def test_returns_legal_action_types(self):
        for s in (state(), state(your_cards=['7s', '2h']), state(community_cards=['Ks', '7d', '2c'], street='flop',
                  amount_owed=0, can_check=True, current_bet=0, your_bet_this_street=0, pot=300)):
            self.assertIn(v3.decide(s)['action'], {'fold', 'check', 'call', 'raise', 'all_in'})
    def test_never_checks_when_owing(self):
        self.assertNotEqual(v3.decide(state(your_cards=['7s', '2h']))['action'], 'check')
if __name__ == '__main__':
    unittest.main()
