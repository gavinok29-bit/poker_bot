import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import branch_guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.getcwd()
        self.root = Path(self.tmp.name)/'work'
        self.root.mkdir()
        self.remote = Path(self.tmp.name)/'remote.git'
        subprocess.run(['git', 'init', '--bare', str(self.remote)], capture_output=True, check=True)
        os.chdir(self.root)
        for args in (['init', '-b', 'personal-test'], ['config','user.email','test@localhost'],
                     ['config','user.name','Local Test']):
            self.git(*args)
        Path('README.txt').write_text('Local guard test\n')
        self.git('add','README.txt')
        self.git('commit','-m','Local test')
        self.git('remote','add','origin', str(self.remote))
        self.original = branch_guard.ALLOWED
        branch_guard.ALLOWED = (str(self.remote),)
    def tearDown(self):
        branch_guard.ALLOWED = self.original
        os.chdir(self.old)
        self.tmp.cleanup()
    def git(self, *args, check=True):
        return subprocess.run(['git',*args], capture_output=True, text=True, check=check)
    def install(self):
        with contextlib.redirect_stdout(io.StringIO()):
            branch_guard.install('personal-test')
    def test_installs_without_push(self):
        self.install()
        self.assertTrue(Path('.git/hooks/pre-push').exists())
        self.assertEqual(self.git('ls-remote','origin').stdout, '')
    def test_rejects_main(self):
        with self.assertRaises(ValueError):
            branch_guard.install('main')
    def test_rejects_wrong_current_branch(self):
        self.git('switch','-c','another')
        with self.assertRaises(ValueError):
            branch_guard.install('personal-test')
    def test_rejects_unknown_destination(self):
        branch_guard.ALLOWED = ('https://example.invalid/repo',)
        with self.assertRaises(ValueError):
            branch_guard.install('personal-test')
    def test_preserves_existing_hook(self):
        hook = Path('.git/hooks/pre-push')
        hook.write_text('existing hook')
        with self.assertRaises(ValueError):
            branch_guard.install('personal-test')
        self.assertEqual(hook.read_text(), 'existing hook')
    def test_blocks_main_push(self):
        self.install()
        r = self.git('push','origin','HEAD:refs/heads/main', check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('BLOCKED', r.stderr)
        self.assertEqual(self.git('ls-remote','origin').stdout, '')
    def test_default_push_updates_only_personal_branch(self):
        self.install()
        self.git('push','origin')
        refs = self.git('ls-remote','--heads','origin').stdout
        self.assertIn('refs/heads/personal-test', refs)
        self.assertNotIn('refs/heads/main', refs)
    def test_switch_to_main_blocks_push(self):
        self.install()
        self.git('switch','-c','main')
        r = self.git('push','origin', check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.git('ls-remote','origin').stdout, '')
    def test_blocks_branch_deletion(self):
        self.install()
        self.git('push','origin')
        r = self.git('push','origin',':refs/heads/personal-test',check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('refs/heads/personal-test',self.git('ls-remote','--heads','origin').stdout)

if __name__ == '__main__':
    unittest.main(verbosity=2)
