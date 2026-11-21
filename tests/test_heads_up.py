import importlib.util
from pathlib import Path
import random
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2', ROOT/'my_bots/heads_up_v2/bot.py')
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)


def state(**kwargs):
    s = dict(your_cards=['As', 'Ah'], community_cards=[], street='preflop',
             pot=150, your_stack=9950, your_bet_this_street=50, amount_owed=50,
             can_check=False, current_bet=100, min_raise_to=200, your_bot_id='me',
             blinds={'sb': 50, 'bb': 100, 'ante': 0}, match_action_log=[],
             action_log=[{'action': 'small_blind', 'street': 'preflop'},
                         {'action': 'big_blind', 'street': 'preflop'}],
             players=[{'bot_id': 'me', 'stack': 9950, 'bet_this_street': 50},
                      {'bot_id': 'other', 'stack': 9900, 'bet_this_street': 100}],
             hand_num=0, time_limit_ms=2000)
    s.update(kwargs)
    return s


class StrategyTests(unittest.TestCase):
    def test_small_blind_opens_owing_half_blind(self):
        self.assertEqual(bot.choose(state(), .60)['action'], 'raise')
    def test_short_stack_small_blind_shoves_premium(self):
        self.assertEqual(bot.choose(state(your_stack=950), .85)['action'], 'all_in')
    def test_short_stack_small_blind_folds_weak(self):
        self.assertEqual(bot.choose(state(your_stack=950), .36)['action'], 'fold')
    def test_unequal_stacks_use_opponent(self):
        s = state(players=[{'bot_id': 'other', 'stack': 900, 'bet_this_street': 100}])
        self.assertEqual(bot.effective_bb(s), 10)
        self.assertEqual(bot.choose(s, .85)['action'], 'all_in')
    def test_committed_blind_counts_in_depth(self):
        self.assertEqual(bot.effective_bb(state(your_stack=1151)), 12.01)
    def test_big_blind_medium_hand_checks_limped_pot(self):
        self.assertEqual(bot.choose(state(amount_owed=0, can_check=True), .53)['action'], 'check')
    def test_big_blind_short_stack_can_shove_limp(self):
        self.assertEqual(bot.choose(state(amount_owed=0, can_check=True, your_stack=900), .80)['action'], 'all_in')
    def test_all_in_call_not_aggression(self):
        log = [dict(bot_id='me', action='raise', amount=500, hand_num=0, street='flop'),
               dict(bot_id='other', action='all_in', amount=500, hand_num=0, street='flop')]
        self.assertEqual(bot.opponent_stats(state(match_action_log=log))[2], 0)
    def test_all_in_raise_is_aggression(self):
        log = [dict(bot_id='me', action='raise', amount=500, hand_num=0, street='flop'),
               dict(bot_id='other', action='all_in', amount=1000, hand_num=0, street='flop')]
        self.assertEqual(bot.opponent_stats(state(match_action_log=log))[2], 1)
    def test_raise_is_not_an_unopened_pot(self):
        self.assertFalse(bot.unopened(state(current_bet=300)))
        self.assertFalse(bot.unopened(state(action_log=[{'action': 'raise', 'street': 'preflop'}])))
    def test_big_blind_free_option_never_folds(self):
        self.assertEqual(bot.choose(state(amount_owed=0, can_check=True), .30)['action'], 'check')
    def test_unaffordable_call_price_excludes_unmatched_chips(self):
        s = state(your_stack=100, amount_owed=1000, pot=1100)
        self.assertAlmostEqual(bot.call_price(s), 1/3)
    def test_all_in_opponent_cannot_be_raised(self):
        s = state(players=[{'bot_id': 'other', 'stack': 0, 'bet_this_street': 200,
                            'is_all_in': True}], current_bet=200, amount_owed=150)
        self.assertEqual(bot.choose(s, .85)['action'], 'call')
        self.assertEqual(bot.raise_to(s, 800)['action'], 'call')
    def test_raise_total_is_legal(self):
        for target in (0, 99, 201, 500, 15000):
            a = bot.raise_to(state(), target)
            if a['action'] == 'raise':
                self.assertGreaterEqual(a['amount'], 200)
                self.assertLessEqual(a['amount'], 10000)
    def test_no_check_while_owing_property(self):
        rng = random.Random(5)
        for _ in range(1000):
            owed = rng.randint(1, 1000)
            s = state(amount_owed=owed, current_bet=owed+50, your_stack=rng.randint(1, 10000))
            self.assertNotEqual(bot.choose(s, rng.random(), rng=rng)['action'], 'check')
    def test_warmup(self):
        self.assertEqual(bot.decide({'type': 'warmup'})['action'], 'check')
    def test_fallback_visible_and_safe(self):
        s = state(your_cards=['bad'])
        act = bot.decide(s)
        self.assertEqual(act['action'], 'fold')
        self.assertIn('fallback', act['debug'])
    def test_recent_fold_tracking_no_posts(self):
        log = []
        for h in range(20):
            log += [dict(bot_id='me', action='raise', hand_num=h, street='flop'),
                    dict(bot_id='other', action='fold', hand_num=h, street='flop')]
        self.assertEqual(bot.opponent_stats(state(match_action_log=log))[:2], (1, 20))
    def test_no_cross_street_fold_attribution(self):
        log = [dict(bot_id='me', action='raise', hand_num=0, street='flop'),
               dict(bot_id='other', action='fold', hand_num=0, street='turn')]
        self.assertEqual(bot.opponent_stats(state(match_action_log=log))[1], 0)
    def test_draw_needs_hole_contribution(self):
        self.assertFalse(bot.has_draw(state(your_cards=['2c', '2d'], community_cards=['8s','9s','Ts','Js'],street='turn')))
        self.assertTrue(bot.has_draw(state(your_cards=['As','2d'], community_cards=['4s','8s','Ts'],street='flop')))
    def test_fixed_sample_replay(self):
        bot._CACHE.clear()
        a = bot.estimate_equity(['As','Kh'], [], budget=10, rng=random.Random(31))
        bot._CACHE.clear()
        b = bot.estimate_equity(['As','Kh'], [], budget=10, rng=random.Random(31))
        self.assertEqual(a, b)
    def test_equity_royal_flush(self):
        self.assertEqual(bot.estimate_equity(['As','Ks'], ['Qs','Js','Ts'], budget=10), 1)
    def test_board_royal_flush_is_tie(self):
        self.assertEqual(bot.estimate_equity(['2d','3h'], ['As','Ks','Qs','Js','Ts'], budget=10), .5)
    def test_aces_equity_reasonable(self):
        bot._CACHE.clear()
        self.assertTrue(.79 < bot.estimate_equity(['As','Ah'], [], budget=10, rng=random.Random(7)) < .91)
    def test_cache_bounded(self):
        bot._CACHE.clear()
        for i in range(2050):
            bot._CACHE[(i,)] = .5
        bot.estimate_equity(['4s','5d'], [], budget=10)
        self.assertLessEqual(len(bot._CACHE), 2048)
    def test_actual_engine_unopened_small_blind(self):
        from pokercore.engine import PokerEngine
        e = PokerEngine(hand_id='regression', bot_ids=['me','other'], dealer_seat=0,
                        starting_stacks={'me':1000,'other':19000}, seed=7,
                        small_blind=50,big_blind=100,ante=0,hand_num=0)
        s = e.start_hand()
        self.assertTrue(bot.unopened(s))
        self.assertEqual(s['amount_owed'], 50)
        self.assertEqual(bot.choose(s, .85)['action'], 'all_in')
    def test_harness_incomplete_never_counts_as_draw(self):
        from run_bench import run_pair
        with tempfile.TemporaryDirectory() as d:
            result = run_pair(('bots/caller','bots/caller',1,d,False,1,10000,'standard'))
        self.assertEqual(result['outcome'], 'invalid')
    def test_harness_crashes_are_invalid(self):
        from run_bench import run_pair
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d)/'bad'
            bad.mkdir()
            (bad/'bot.py').write_text('def decide(s):\n    raise ValueError("intentional test crash")\n')
            result = run_pair((str(bad),'bots/caller',1,d,False,1,10000,'standard'))
        self.assertEqual(result['outcome'], 'invalid')
        self.assertTrue(any(r['errors'] for r in result['runs']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
