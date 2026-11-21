# Christian's heads-up poker workspace

Start here. This package is for local development and review. Nothing has been pushed, merged, made public or submitted.

## What you have

- `my_bots/heads_up_v2/bot.py`: the corrected heads-up rules bot. Only this file is needed to run v2. The folder name is a development name, not your confirmed submission team name.
- `baselines/team_v1/bot.py`: the original v1, unchanged for comparison. Its original comments contain the short-stack and timeout overstatements corrected below.
- `RESULTS_BY_PLAYSTYLE.md`: per-opponent results for team_v1, v2 and the new v3 candidate on held-out seeds. Read this first. Honest answer: no version beat v1.
- `my_bots/heads_up_v3/`: candidate (v2 plus 3 changes), not shown stronger than v1.
- `IMPROVEMENT_LOOP.md`: read this first for how to use the data and tests to improve the bot (tests do not train it).
- `data/`: audited summary of the 2014 ACPC heads-up sample, aggregate output and parser.
- `generate_acpc_opponents.py`, `compare_reports.py`: build 3 sample-shaped opponents and compare two benchmark reports.
- `opponents/`: 17 practice bots (3 new `acpc_style_*` ones are synthetic, shaped by the sample, not reproductions of the ACPC bots), plus the original 14: with several settings for tight/value, loose/aggressive, short-stack, trapping, mixed and calling styles, plus always-fold and always-shove.
- `bots/caller` and `bots/template`: the two official examples, unchanged.
- `run_bench.py`: reliable paired-freezeout benchmark with structured reports and event logs.
- `tests/`: 42 tests (the original 37 plus 3 for the new opponents and 2 smoke tests for v3).
- `branch_guard.py`: optional LOCAL Git safeguard. It never pushes anything.
- `TEST_RESULTS.md`: exact release results, including losses and limits.
- `GEMINI_INSTRUCTIONS.md`: the publishing handoff, for your verified `chris-bot-v1` branch.
- `README.md`: the original competition kit instructions. Read them too.

## Safety: this is not Gavin's active checkout

The source was read from Gavin's `main` branch on November 21, 2026, around 11:18 AM Toronto time. Its 23 listed source files matched the official starter kit byte-for-byte at that check. This is a source snapshot, not a clone of every branch or Gavin's Git history. Other branches and unpushed PC work were not copied. Later, at about 11:33 AM, your `chris-bot-v1` branch was verified separately: it already contains team_v1 and three sparring folders, which must be preserved. The local development workspace has its own Git history and no remote. The ZIP intentionally omits that local `.git` directory, so extracting it gives ordinary files with no connection to any GitHub repository.

Extract into a NEW folder. Do not unzip over Gavin's repo or any existing work. Opening, editing and testing this standalone package does not change GitHub. A branch isolates commits, not file permissions or all possible mistakes. Never merge into or push to `main` while experimenting. Git hooks can be bypassed; only the repository owner can configure server-side branch protection.

## Windows setup

Use Ubuntu under WSL, not plain PowerShell for the Linux commands below. If WSL is not installed, open PowerShell as Administrator, run `wsl --install`, follow its prompts and restart if requested. Open Ubuntu and finish its first-use setup.

In Ubuntu, once:

```bash
sudo apt update
sudo apt install -y build-essential git make unzip
```

Save the ZIP from the email to Windows Downloads. In Ubuntu, find it under `/mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/`. Replace `YOUR_WINDOWS_USERNAME` with your actual Windows user folder. Do not paste it unchanged. Extract into a new Linux home folder:

```bash
mkdir -p ~/poker-work
cd ~/poker-work
unzip /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/poker_heads_up_workspace.zip
cd poker_bot_christian
```

Stop if a folder named `poker_bot_christian` already exists. Pick a fresh parent folder rather than overwrite it.

The kit needs Python 3.10. If `uv` is not installed, use the installation instructions at https://docs.astral.sh/uv/ and reopen Ubuntu afterward. This avoids changing the system Python. Then, from this package's root:

```bash
uv python install 3.10
uv venv --seed -p 3.10 .venv
source .venv/bin/activate
make install
```

`make install` handles the old eval7 build dependency, including `Cython<3`. Re-run `source .venv/bin/activate` whenever you open a new terminal in this project. Do not ship `.venv` in your bot folder.

## First checks

From the project root, environment active:

```bash
python3 -m unittest discover -s tests -v
make validate BOT=my_bots/heads_up_v2
python3 play.py my_bots/heads_up_v2 bots/caller --seed 7
python3 play.py bots/caller my_bots/heads_up_v2 --seed 7
```

The tests intentionally exercise a crashing fake bot. The official process code can emit `ResourceWarning` messages about closed process pipes in that test; all 42 tests still passed here. The official engine was not changed to suppress those warnings.

## What v2 actually does

1. It samples up to 700 possible opponent hands and board runouts, scores them with eval7, and estimates showdown equity against a RANDOM hand. It does not infer an accurate betting range or solve poker.
2. It compares that estimate with the price of calling. For an oversized all-in it caps the call by your remaining stack and removes the unmatched portion from the pot-odds denominator.
3. It applies explicit betting rules: open the small blind, value-raise strong hands, and use a small number of draw-based semi-bluffs. Those thresholds and bet sizes are heuristics, not trained weights or GTO charts.
4. It reads recent public action history to adjust for folds and aggression. It distinguishes all-in calls from all-in raises when prior amounts are visible. The history is truncated to the last 200 entries, so these are imperfect recent statistics, not a permanent opponent model.
5. Its effective stack is the smaller player's stack including current-street commitments, expressed in big blinds. It handles unopened small-blind decisions before treating money owed as an opponent bet. The big blind's free checking option is handled separately.
6. It keeps a bounded equity cache for the match. Fixed sample counts normally replay with seeded randomness; an emergency wall-clock cutoff can change a result on a heavily loaded machine. The equity budget is at most 0.12 seconds, checked every 16 samples. The runner owns the actual hard deadline. No code can promise a timeout is impossible on untested hardware.

There is no dataset training, persistent learning, runtime LLM, network call, background work or file write inside the bot. Its small history-based adjustments and cache disappear with the match process. Playing matches produces evidence; it does not permanently train this bot.

## What was fixed compared with v1

- Small-blind opening raises/shoves are reachable even while owing the other half of the big blind.
- Effective stack is capped by the opponent and counts live street commitments.
- Short-stack and big-blind checking options have explicit regression tests.
- Pot odds handle a call smaller than the announced bet.
- No attempted value raise against an opponent already all-in.
- All-in calls are not automatically counted as aggression.
- Bluffs favor draws that use your hole cards rather than random weak-hand reraises.
- Bounded cache and fixed sample cap improve resource use and normal seeded replay.
- Errors fall back to check only when free, otherwise fold; fallback use is visible in debug logs.
- Benchmark errors, crashes, restarts, warmup failures, fallback use and incomplete freezeouts are INVALID results, never hidden wins or draws.

This is a corrected candidate, not a proven strength upgrade. On ten fresh direct v2-versus-v1 pairs, v2 won 1, drew 7 and lost 2. That small sample does not justify a claim that v2 beats v1. Keep both and compare more before choosing an entry.

## Repeatable testing, not tuning to one toy bot

A quick check across all 16 default practice opponents:

```bash
python3 run_bench.py --seeds 1001 1002 1003 --out results/my_smoke
```

Each seed produces two freezeouts, with the same deck seed and seats swapped. A pairing is won only by winning both matches, drawn by winning one, and lost by losing both. This follows the kit's qualification description. The harness uses the unchanged engine's structured results, not a regex on printed winner names. Candidate/opponent identities and seeded bot randomness stay stable within a pair. It logs configuration, core hash, code hashes and per-match errors.

Use a NEW output directory each time. The harness refuses to overwrite old evidence. `--workers 1` is the default and is better for timing checks. `--workers 4` speeds up experiments but creates CPU contention, so do not treat parallel local timings as sandbox certification.

Direct comparison against v1:

```bash
python3 run_bench.py --opponents baselines/team_v1 --seeds 1101 1102 1103 1104 1105 --out results/my_v1_comparison
```

Short-stack stress, intentionally different from the standard 10,000-chip start:

```bash
python3 run_bench.py --stack 500 --opponents opponents/short_stack opponents/always_shove opponents/trapper --seeds 1201 1202 1203 --out results/my_short_stacks
```

Tune only on a development seed set. Freeze the candidate, then evaluate on unused seeds and new opponent settings. Look at each matchup, invalid runs and worst-case behavior, not just an average win rate. Ten seeds per opponent remains a small sample. Use substantially more when choosing between close candidates. Do not keep tuning to these published release seeds and still call them held-out tests.

These practice bots are simple synthetic styles, not a complete field and not GTO opponents. No finite suite covers every possible poker bot. Strong range-aware or adaptive opponents may exploit us. A future improvement would be range-aware equity and a tested heads-up preflop strategy, after the rules and engine checks remain green. A huge multi-variant dataset is not needed just to run these tests.

## Required before relying on tournament compatibility

Docker was not available here. The official validator and local guarded runner passed, but the actual half-core / 768 MB Docker environment is still UNVERIFIED.

On a machine with Docker working for WSL, run from the project root:

```bash
make sandbox
python3 play.py my_bots/heads_up_v2 bots/caller --docker --seed 7
python3 play.py bots/caller my_bots/heads_up_v2 --docker --seed 7
python3 run_bench.py --docker --opponents opponents/always_shove opponents/short_stack --seeds 1301 1302 --out results/my_docker_check
```

Check for errors and timeouts, not just who won. The actual rules are 2 seconds per decision, half a CPU core, 768 MB, no network and no background thinking. The bot itself uses only permitted imports. Development tools such as the benchmark and branch guard are NOT part of the submission folder.

## Before publishing or submitting

- Follow `GEMINI_INSTRUCTIONS.md` for the verified `chris-bot-v1` branch. Inspect the live checkout before copying anything.
- Preserve Gavin's bot and any existing team folder. Add v2 alongside them for testing. If a target path already exists, STOP rather than overwrite it.
- Your actual submission folder must be `my_bots/<registered_team_name>/`. `heads_up_v2` is not a confirmed registered name. The unit tests hardcode this development path, so leave it as a test candidate and copy only its `bot.py` into the chosen entry folder after team agreement.
- The kit collects `main`, not an experimental personal branch. A branch push alone is NOT a submission. Adoption/merge, repository visibility and submission are separate decisions for your team and organizers, not actions taken in this package.
- The current kit requires a public repo by the deadline and a submission form. Do not change visibility or submit until you decide to. Ask the organizer for the exact coding cutoff, which remains unverified here.
- Read `bot.py` and explain equity, pot odds, effective stacks, heuristic thresholds, opponent stats and limits in your own words. Coding assistance is allowed; runtime AI agents/LLM calls are not.

## Sources

Verified team source: https://github.com/gavinok29-bit/poker_bot

Official kit and rules checked November 21, 2026: https://github.com/FreddyyAndrews/poker-hackathon-kit

The original kit `README.md`, `CORE_VERSION` and engine files are included unchanged. The Python 3.10 / WSL instructions above follow that kit. No exact coding deadline, teammate adoption or remote branch contents are assumed.
