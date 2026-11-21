# What top heads-up no-limit bots do (audited)
Measured from a converted sample of the official 2014 Annual Computer Poker Competition (ACPC) heads-up no-limit logs: 576,000 hand records (unique converted IDs, no double counting) among the top-4 instant-runoff finishers Tartanian7, Prelude, Hyperborean, Slumbot; 144,000 button + 144,000 big-blind decisions per bot. Source: unfoldpoker.com/acpc-hand-histories (sample ZIP), derived from computerpokercompetition.org official logs.

FORMAT LIMITS, read first: every one of the 576,000 hands is $0.50/$1 blinds with $200 stacks = exactly 200 big blinds deep, fixed blinds, bot-vs-bot. Our competition is rising-blind duplicate freezeouts with much shorter stacks. Everything below is deep-stack default play only. The sample contains zero short-stack or push/fold situations, which is exactly where our bot is weakest. These are frequencies of what four 2014 academic bots did, not proof of what beats our field.

## Denominator reconciliation (the audit point)
No contradiction: button VPIP and overall VPIP use different seat denominators. Each bot is button in exactly half its hands. Overall VPIP = (button voluntary + BB voluntary) / all hands.
- Button VPIP: 87.6-90.7% per bot (raise + limp out of 144,000 button hands each).
- BB VPIP: 53.9-60.5% per bot (any voluntary call/raise out of 144,000 BB hands each).
- Overall: 70.7-75.6% (matches the headline VPIP).

## Button first action (n = 144,000 per bot)
- Tartanian7: raise 76.3% / limp 14.0% / fold 9.8%
- Prelude: raise 72.0% / limp 18.7% / fold 9.3%
- Hyperborean: raise 74.0% / limp 14.9% / fold 11.2%
- Slumbot: raise 75.7% / limp 11.9% / fold 12.4%
- Open sizes (of raises): mostly 2-3bb, most common 2.5bb; effectively no shoves at 200bb.

## BB facing a button raise (conditional, n = 106,533 to 108,402 per bot)
- fold 27.9-33.9%, call 53.0-58.5%, 3-bet 13.0-17.3%.

## BB facing a button limp (conditional, n = 19,591 to 22,884 per bot)
- raise 31.2-41.2%, check the rest.

## Flop continuation bet by the preflop raiser (conditional, no donk bet)
- 42.8-58.4% depending on bot.

## Hand-class observations (per-bot counts, this sample only)
Button folds: 83o, 72o, 92o, 32o, 93o, 42o, 82o, 62o were folded in 100% of observed cases by EACH of the four bots (n = 1,221-1,398 per class per bot). 73o/52o/T2o folded 90%+. 94o, T3o, J2o mixed fold/limp. Everything else was always played.
BB 3-bets vs an open: AA, KK, AKs, AKo, AQs were 3-bet 100% by each bot (n = 320-973 per class per bot); QQ/JJ/KQs/KJs also ~100% pooled. 76s 96-99% per bot (bluff 3-bet; e.g. raw hand: Tartanian7 3-bet 76s to 12bb over a 3bb open). 65s varied: 100%/100%/100% for three bots but Slumbot only 73%.
Caution: all four are CFR-family academic bots with similar solution structure, so their agreement is not independent confirmation, and 100% here means "in every observed instance in this sample", not a universal rule.

## Data quality notes
- Parser verified against raw text on random hands: raise-to amounts ("raises $2 to $3" = 3bb), blind posts excluded from VPIP, conditional BB categories separated, all matched.
- All cards are revealed in these converted logs (that is how folded-hand ranges are knowable).
- Converted hand IDs are unique, but ACPC duplicate poker replays the same card deals with seats swapped, so underlying deals repeat by design; frequencies are unaffected.
- This sample is 2014 top-4 matches only. It is NOT equivalent to the Zenodo archive (which spans 2009-2017, ~279M HUNL hands). The official full-year NL archives at computerpokercompetition.org responded HTTP 200 to HEAD requests at check time (2014: 232MB, 2016: 152MB, 2017: 458MB, 2009: 56MB), so larger pulls are possible from a normal connection if ever wanted.

## Practical takeaway for our bot (deep-stack spots only)
- Button default: raise ~75% to 2.5bb, limp ~13-19%, fold the ~10% trash listed above.
- BB: fold only bottom offsuit trash vs a raise, 3-bet premiums + some suited connectors.
- C-bet roughly half of flops as preflop raiser.
- Below ~15bb effective, none of this applies: switch to push/fold logic, which this dataset cannot inform.
