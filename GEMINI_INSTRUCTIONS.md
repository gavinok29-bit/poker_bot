# Paste this into Gemini

I am Christian. Help me add and test a poker bot candidate on my existing personal branch without changing Gavin's main branch. Explain each step and wait for terminal output when needed. Do not push, merge, delete, change visibility or submit without my separate confirmation.

## Verified destination and current context

- Repository: https://github.com/gavinok29-bit/poker_bot
- My exact personal branch: `chris-bot-v1`
- This branch was read-only verified on November 21, 2026, around 11:33 AM Toronto time.
- It already has `my_bots/team_v1`, `my_bots/spar_aggro`, `my_bots/spar_nit` and `my_bots/spar_shove`. Preserve these existing folders.
- The downloaded `poker_bot_christian` folder is a standalone source snapshot with no `.git` or remote, NOT my current checkout and NOT a clone of my branch.
- `my_bots/heads_up_v2` is a new test candidate. It is not proven stronger than v1. `my_bots/heads_up_v3` is a further candidate; held-out results (RESULTS_BY_PLAYSTYLE.md) show no version beating the original team_v1, so team_v1 remains the best-supported bot. The package includes 42 passing tests and release results. Docker is still unverified.
- I use Windows. Run project commands in Ubuntu/WSL with Python 3.10.

## Non-negotiable safety checks

1. First identify the exact folder we are in. Ask for `pwd`, `git status --short --branch`, `git branch --show-current`, `git remote -v`, and `git diff --stat` in the intended checkout. If it is not a Git repo, distinguish the extracted package from my checkout. Do not blindly run `git init`, replace origin or copy the snapshot over an existing repo.
2. If there are uncommitted changes, explain and stop before switching branches or copying files. Do not discard or stash them without asking me. Never run reset-hard, clean, force-pull, force-push or deletion commands.
3. Verify origin's fetch AND push URLs point exactly to `gavinok29-bit/poker_bot`. Stop on any different repository or an extra push destination. Inspect any existing push refspecs and hooks before changing Git configuration.
4. Fetch read-only with `git fetch origin`, then verify the existing branch with `git show-ref --verify refs/remotes/origin/chris-bot-v1`. If absent, stop and ask me, do not invent another branch.
5. If the local branch exists and the worktree is clean, use `git switch chris-bot-v1`. If it does not exist, use `git switch --track -c chris-bot-v1 origin/chris-bot-v1`. If local and remote histories differ, inspect the divergence and ask before integrating. No automatic rebase/merge.
6. Check that `git branch --show-current` prints exactly `chris-bot-v1`. Never edit or commit on `main`, `master`, a detached HEAD or a similarly named branch.
7. Never change `pokercore`, `play.py`, the official kit files, Gavin's existing bot or the existing team_v1/sparring folders. Do not copy the ZIP's full root into this checkout.

If I do not have a checkout yet, use a NEW folder that does not exist:

```bash
git clone --branch chris-bot-v1 --single-branch https://github.com/gavinok29-bit/poker_bot poker_bot_chris_branch
cd poker_bot_chris_branch
```

Only use that after checking the folder is new. Sign in to GitHub through its normal browser/credential-manager flow. Never ask me to paste passwords, tokens, cookies or recovery codes into chat. Stop if the branch cannot be cloned.

## Copy only the new work, with no overwrites

Keep the extracted package in a separate folder. Identify its actual absolute path instead of guessing a Windows username. Before copying, list every destination and verify none exists. Stop if any exists and compare it manually with me instead of overwriting.

Copy these items into my checked `chris-bot-v1` checkout, preserving their relative paths:

- `my_bots/heads_up_v2/`
- `my_bots/heads_up_v3/`
- `baselines/team_v1/` (the packaged original v1 comparison copy)
- `opponents/` (all 17 practice bots, including 3 new synthetic `acpc_style_*` ones)
- `tests/`
- `branch_guard.py`
- `generate_opponents.py`
- `summarize_heldout.py`
- `RESULTS_BY_PLAYSTYLE.md`
- `generate_acpc_opponents.py`
- `compare_reports.py`
- `data/` (audited 2014 ACPC sample summary, aggregate output, reference parser)
- `IMPROVEMENT_LOOP.md`
- `run_bench.py`
- `START_HERE.md`
- `TEST_RESULTS.md`
- `GEMINI_INSTRUCTIONS.md`
- `SOURCE_PROVENANCE.json`

`run_bench.py` or `tests/` may already exist on my branch. Do not replace them silently. If any selected target exists, STOP and compare; we can keep these tools under a new dedicated development folder or agree on a specific change after reviewing the diff. Do not delete existing tools to make copying easier. Results and logs can stay in the extracted package; they are not required for the bot to run.

Before installing the optional guard, explain that it is a local hook, not a guarantee and not GitHub branch protection. It can be bypassed, and remote URLs/refspecs still need inspection. It blocks main pushes and branch deletion and pins default origin pushes to my personal branch. If `.git/hooks/pre-push`, `core.hooksPath` or conflicting origin push settings already exist, the helper stops rather than overwrite them. Review with me rather than bypassing its checks.

From the checked branch root:

```bash
python3 branch_guard.py --branch chris-bot-v1
```

This installs a local safeguard only. It does not push. Git hooks and this configuration do not travel to another checkout, so repeat the checks there if I change computers.

## Set up and test

Follow `START_HERE.md` and the original kit README. Use Python 3.10 and WSL. Keep the virtual environment outside the submission folder. From the project root:

```bash
source .venv/bin/activate
python3 -m unittest discover -s tests -v
make validate BOT=my_bots/heads_up_v2
python3 play.py my_bots/heads_up_v2 bots/caller --seed 7
python3 play.py bots/caller my_bots/heads_up_v2 --seed 7
python3 run_bench.py --seeds 1001 1002 1003 --out results/my_smoke
```

If the existing branch's v1 differs from the archived baseline, also benchmark against `my_bots/team_v1`; do not assume the packaged baseline is the latest strategy. The default suite has 16 opponents; the 3 `acpc_style_*` opponents are run separately by listing them in `--opponents` (see `IMPROVEMENT_LOOP.md`). They are synthetic, shaped by a deep-stack 2014 ACPC sample, not reproductions of those bots, and their short-stack play is an assumption. Tests do not train the bot; changes are kept only if they help on held-out seeds. Paired results are wins/draws/losses over swapped-seat freezeouts, not fixed-hand chip profit. Use fresh seeds. Errors/incomplete runs must be invalid, never draws.

Docker must be tested before tournament reliance:

```bash
make sandbox
python3 play.py my_bots/heads_up_v2 bots/caller --docker --seed 7
python3 play.py bots/caller my_bots/heads_up_v2 --docker --seed 7
```

Do not say Docker passed unless it actually runs successfully. The bot has no runtime network/LLM calls. Do not add any, and do not edit the official engine. I need to understand how the code works, so explain equity, pot odds, effective stacks, short-stack heuristics and limitations rather than just giving commands.

## Commit locally before discussing a push

Show `git status`, `git diff --stat` and the actual source diff. For new untracked files inspect their contents too. Stage only the reviewed new candidate and development files, using explicit paths. Do NOT use `git add .`, include `.venv`, logs, credentials, extracted archives, unrelated edits or change the existing team folder. After staging, show `git diff --cached --stat` and `git diff --cached --name-only` and verify no original core or teammate files changed.

Help me choose an accurate commit message, then make only the reviewed LOCAL commit on `chris-bot-v1`. A commit is not a push and does not update GitHub by itself.

## Push only after asking me again

Immediately before proposing any push, verify all of:

```bash
git branch --show-current
git remote get-url --all origin
git remote get-url --push --all origin
git config --get-all remote.origin.push
git status --short --branch
```

The branch must be `chris-bot-v1`, URLs must be the expected shared repo, and the configured refspec must not target main. Check remote divergence with a fresh `git fetch origin`. Do not merge/rebase automatically if the remote branch changed. Ask me to confirm the exact repository, branch and commit to upload. After I approve those specifics, first use a dry-run:

```bash
git push --dry-run origin HEAD:refs/heads/chris-bot-v1
```

Read the dry-run destination. It must mention ONLY `chris-bot-v1`, no main, tags, deletion, mirror, all-branches or forced update. If safe and still approved, the explicit real command is:

```bash
git push origin HEAD:refs/heads/chris-bot-v1
```

Never use `--force`, `--all`, `--mirror`, `--tags`, `--no-verify`, or change/remove the guard to make a rejected push work. A rejected non-fast-forward push is a reason to inspect and ask, not override it. Verify the uploaded commit appears on `chris-bot-v1` afterward. Do not open/merge a pull request, touch main, change visibility or submit a form.

The competition kit collects MAIN at the deadline. My branch is an isolated experiment, not a submitted entry. Moving an agreed strategy into the registered team folder on main is a separate team decision. The exact coding cutoff and registered team folder are not verified in this package.
