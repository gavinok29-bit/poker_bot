#!/usr/bin/env python3
"""Install LOCAL branch-only push safeguards in an existing checkout.
Development helper, never copy this file into the submitted bot folder.
Run only after choosing the exact branch you own. No fetch/push occurs here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ALLOWED = ('https://github.com/gavinok29-bit/poker_bot',
           'https://github.com/gavinok29-bit/poker_bot.git',
           'git@github.com:gavinok29-bit/poker_bot.git',
           'ssh://git@github.com/gavinok29-bit/poker_bot.git')


def git(*args, required=True):
    p = subprocess.run(['git', *args], capture_output=True, text=True)
    if required and p.returncode:
        raise RuntimeError(p.stderr.strip() or 'Git failed')
    return p.stdout.strip()


def install(branch):
    if branch in ('main', 'master', 'trunk') or not branch:
        raise ValueError('Choose your exact personal branch, not main/master/trunk.')
    git('check-ref-format', '--branch', branch)
    if git('branch', '--show-current') != branch:
        raise ValueError('STOP: current branch does not match requested personal branch.')
    for flags in ([], ['--push']):
        urls = git('remote', 'get-url', *flags, '--all', 'origin').splitlines()
        if len(urls) != 1 or urls[0] not in ALLOWED:
            raise ValueError('STOP: origin URL is not the expected shared repository.')
    custom = git('config', '--get', 'core.hooksPath', required=False)
    if custom:
        raise ValueError('Existing core.hooksPath requires manual review; not changing it.')
    hooks = Path(git('rev-parse', '--git-path', 'hooks'))
    hook = hooks/'pre-push'
    if hook.exists():
        raise ValueError('Existing pre-push hook requires review; not overwriting it.')
    refspec = 'HEAD:refs/heads/'+branch
    configured = git('config', '--get-all', 'remote.origin.push', required=False).splitlines()
    if configured and configured != [refspec]:
        raise ValueError('Existing origin push refspec requires review; not overwriting it.')
    header = '#!/usr/bin/env python3\nimport subprocess, sys\n'
    code = '''
BRANCH = %r
URLS = %r
remote = sys.argv[1] if len(sys.argv) > 1 else ''
url = sys.argv[2] if len(sys.argv) > 2 else ''
current = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
if current != BRANCH or remote != 'origin' or url not in URLS:
    sys.exit('BLOCKED: only the configured personal branch on origin may be pushed.')
updates = [line.split() for line in sys.stdin if line.strip()]
for row in updates:
    if len(row) != 4:
        sys.exit('BLOCKED: unknown push update.')
    local_ref, local_sha, remote_ref, remote_sha = row
    if local_ref not in ('HEAD', 'refs/heads/'+BRANCH) or remote_ref != 'refs/heads/'+BRANCH:
        sys.exit('BLOCKED: push destination/source is not the personal branch.')
    if set(local_sha) == {'0'}:
        sys.exit('BLOCKED: branch deletion is not allowed by this guard.')
print('Local guard: verified personal-branch-only destination.')
''' % (branch, ALLOWED)
    hooks.mkdir(parents=True, exist_ok=True)
    with hook.open('x') as f:
        f.write(header+code)
    hook.chmod(0o755)
    git('config', 'remote.origin.push', refspec)
    git('config', 'push.default', 'nothing')
    git('config', 'push.followTags', 'false')
    print('Installed local push guard for branch:', branch)
    print('No push occurred. This guard can be bypassed and is NOT server-side protection.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', required=True)
    args = p.parse_args()
    try:
        install(args.branch)
    except (ValueError, RuntimeError) as exc:
        p.exit(1, str(exc)+'\n')

if __name__ == '__main__':
    main()
