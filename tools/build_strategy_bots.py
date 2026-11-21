#!/usr/bin/env python3
"""Build the strategy bots in my_bots/ from tools/strategies/.

Each bot must be one self-contained bot.py, so this stitches together:
  the bot's description + imports + the toolkit (copied from my_bots/nash/bot.py)
  + tools/strategies/_helpers.py + the bot's own strategy code,
and copies data/equity169.npy next to it. leaks.py holds the five sparring
bots, separated by "### <name>" lines.

    python3 tools/build_strategy_bots.py          # rebuild all of them
    python3 tools/build_strategy_bots.py tracker  # just one
"""
import os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "tools", "strategies")
nash = open(os.path.join(ROOT, "my_bots/nash/bot.py")).read()
toolkit = nash[nash.index("# ================================= TOOLKIT"):nash.index("# ============================== NASH STRATEGY")]
toolkit = toolkit.replace("(shared by the nash, monte_carlo and counter_nash bots, each a single file)",
                          "(shared by every bot in my_bots, each a single file)")
helpers = open(os.path.join(S, "_helpers.py")).read()
toolkit = toolkit.replace("if (k & 31) == 31 and time.time() - t0 > budget:", "if (k & 7) == 7 and time.time() - t0 > budget:")
assert "(k & 7) == 7" in toolkit
IMPORTS = "\nimport math  # noqa: F401\nimport os\nimport random\nimport time\n\nimport eval7\nimport numpy as np\n\n"
srcs = {}
for f in sorted(os.listdir(S)):
    if f.startswith("_") or not f.endswith(".py"):
        continue
    text = open(os.path.join(S, f)).read()
    if f == "leaks.py":
        for part in text.split("### ")[1:]:
            name, body = part.split("\n", 1)
            srcs[name.strip()] = body
    else:
        srcs[f[:-3]] = text
only = set(sys.argv[1:])                     # optional: build just these bots
for name, text in srcs.items():
    if only and name not in only:
        continue
    doc, code = text.split('"""(end of description)"""\n', 1)
    out = doc.rstrip() + "\n" + IMPORTS + toolkit + helpers.strip() + "\n\n\n# " + "=" * 74 + "\n# " + name.upper() + " STRATEGY\n\n" + code.strip() + "\n"
    d = os.path.join(ROOT, "my_bots", name)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    open(os.path.join(d, "bot.py"), "w", newline="\n").write(out)
    shutil.copy(os.path.join(ROOT, "my_bots/nash/data/equity169.npy"), os.path.join(d, "data/equity169.npy"))
    print("built", name, len(out.splitlines()), "lines")
