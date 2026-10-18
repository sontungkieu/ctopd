"""Tiny archive fixtures; never download/install the production runtime locally."""
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'ensure_runtime.sh').read_text()
INLINE = SCRIPT.split("<<'CTOPD_ENSURE_PY'\n", 1)[1].rsplit('\nCTOPD_ENSURE_PY', 1)[0]


class EnsureRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.archive = self.root / 'runtime.tar.gz'
        self.prefix = self.root / 'node-local/runtime'
        self.code = {'__name__': 'ensure_fixture'}
        exec(compile(INLINE, 'ensure_runtime.sh:inline', 'exec'), self.code)
        # Fixture installation only: no real GPU/ML runtime or 45 GiB allocation.
        self.code['check_host'] = lambda: None
        self.code['MIN_FREE_BYTES'] = 0

    def tearDown(self):
        self.temp.cleanup()

    def archive_with(self, extra=None):
        import tarfile
        files = {'opt/ctopd-b200/python-b200.sh': b'#!/bin/bash\n',
                 'opt/ctopd-b200/python/cpython-3.12.12-fixture/bin/python3.12': b'fixture-python',
                 'opt/ctopd-b200/cuda/bin/nvcc': b'fixture-nvcc'}
        if extra:
            files.update(extra)
        with tarfile.open(self.archive, 'w:gz') as bundle:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        self.code['EXPECTED_SHA'] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.code['EXPECTED_BYTES'] = self.archive.stat().st_size

    def run_ensure(self, *options):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.code['main'](['--root', str(self.root), '--prefix', str(self.prefix), *options])
        return output.getvalue()

    def test_inline_and_probe_syntax(self):
        ast.parse(INLINE)
        ast.parse(self.code['PROBE'])

    def test_install_and_reuse_without_archive(self):
        self.archive_with()
        output = self.run_ensure('--install-only')
        self.assertIn('RUNTIME_INSTALLED_UNCHECKED=', output)
        self.assertNotIn('RUNTIME_READY=', output)
        receipt = self.prefix / self.code['RECEIPT']
        before = receipt.read_bytes()
        self.archive.unlink()
        output = self.run_ensure('--install-only')
        self.assertIn('REUSING_RUNTIME=', output)
        self.assertEqual(receipt.read_bytes(), before)

    def test_wrong_checksum_never_installs(self):
        self.archive_with()
        self.code['EXPECTED_SHA'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
            self.run_ensure('--install-only')
        self.assertFalse(self.prefix.exists())
        self.assertFalse(list(self.prefix.parent.glob('.runtime.install-*')))

    def test_unmanaged_existing_prefix_preserved(self):
        self.prefix.mkdir(parents=True)
        content = self.prefix / 'user-data'
        content.write_text('keep')
        with self.assertRaisesRegex(RuntimeError, 'unmanaged prefix preserved'):
            self.run_ensure('--install-only')
        self.assertEqual(content.read_text(), 'keep')

    def test_changed_runtime_preserved_and_rejected(self):
        self.archive_with()
        self.run_ensure('--install-only')
        wrapper = self.prefix / 'python-b200.sh'
        wrapper.write_text('user change')
        with self.assertRaisesRegex(RuntimeError, 'identity/integrity check failed'):
            self.run_ensure('--install-only')
        self.assertEqual(wrapper.read_text(), 'user change')

    def test_traversal_rejected_and_evidence_retained(self):
        self.archive_with({'opt/ctopd-b200/../escape': b'bad'})
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Unexpected archive member'):
                self.run_ensure('--install-only')
        self.assertFalse(self.prefix.exists())
        self.assertFalse((self.prefix.parent / 'escape').exists())
        self.assertEqual(len(list(self.prefix.parent.glob('.runtime.install-*'))), 1)

    def test_check_only_does_not_install_or_create_parent(self):
        with self.assertRaises(FileNotFoundError):
            self.run_ensure('--check-only')
        self.assertFalse(self.prefix.parent.exists())

    def test_escaping_symlink_rejected(self):
        import tarfile
        self.archive_with()
        with tarfile.open(self.archive, 'w:gz') as bundle:
            member = tarfile.TarInfo('opt/ctopd-b200/escape')
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp/outside-ctopd'
            bundle.addfile(member)
        self.code['EXPECTED_SHA'] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.code['EXPECTED_BYTES'] = self.archive.stat().st_size
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(tarfile.FilterError):
                self.run_ensure('--install-only')
        self.assertFalse(self.prefix.exists())


if __name__ == '__main__':
    unittest.main()
