"""Tiny archive fixtures; never download/install the production runtime locally."""
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import tempfile
import types
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
        self.code['check_host'] = lambda: (2, 39)
        self.code['MIN_FREE_BYTES'] = 0

    def make_overlay(self, forbidden=False):
        import tarfile
        self.archive_with({'opt/ctopd-b200/host-libs/old-incompatible.so': b'old-v4-host-library'})
        self.overlay = self.root / 'compat/glibc235-v2/overlay.tar.gz'
        self.overlay.parent.mkdir(parents=True)
        files = {'host-libs/libstdc++.so.6': b'compatible-library',
                 'site-packages/sglang_router/sglang_router_rs.abi3.so': b'compatible-router'}
        manifest = {'base_archive_sha256': self.code['EXPECTED_SHA'], 'minimum_glibc': '2.35',
                    'files': [{'path': path, 'sha256': hashlib.sha256(content).hexdigest()} for path, content in files.items()]}
        files['glibc235-overlay-manifest.json'] = json.dumps(manifest).encode()
        if forbidden:
            files['site-packages/torch/unrequested.py'] = b'forbidden'
        with tarfile.open(self.overlay, 'w:gz') as bundle:
            for name, content in files.items():
                member = tarfile.TarInfo('opt/ctopd-b200/' + name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        self.code.update(COMPAT_SHA=hashlib.sha256(self.overlay.read_bytes()).hexdigest(),
                         COMPAT_BYTES=self.overlay.stat().st_size, COMPAT_REVISION='fixture')

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

    def test_selected_gpu_reaches_check_and_forwarded_command(self):
        self.archive_with()
        self.run_ensure('--install-only')
        probes, forwarded = [], []

        def check_output(_command, *, env, text):
            probes.append(env.copy())
            return '{}\n'

        def execvpe(_program, _argv, env):
            forwarded.append(env.copy())

        self.code['subprocess'] = types.SimpleNamespace(check_output=check_output)
        self.code['os'] = types.SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': '3'}, execvpe=execvpe)
        for selection, expected in [(None, '3'), ('2', '2'), ('GPU-fixture-uuid', 'GPU-fixture-uuid')]:
            options = ['--check-only'] + (['--gpu', selection] if selection else [])
            self.run_ensure(*options, '--', '-c', 'pass')
            self.assertEqual(probes[-1]['CUDA_VISIBLE_DEVICES'], expected)
            self.assertEqual(forwarded[-1]['CUDA_VISIBLE_DEVICES'], expected)
        self.assertEqual(self.code['os'].environ['CUDA_VISIBLE_DEVICES'], '3')

    def test_gpu_selection_rejects_empty_or_multiple_devices_before_install(self):
        for selection in ['', '0,2', '-1', 'invalid']:
            with self.assertRaisesRegex(ValueError, 'exactly one allocated GPU'):
                self.run_ensure('--gpu=' + selection, '--install-only')
        self.assertFalse(self.prefix.parent.exists())

    def test_both_required_host_baselines_and_older_rejection(self):
        isolated = {'__name__': 'host_test'}
        exec(compile(INLINE, 'ensure_runtime.sh:inline', 'exec'), isolated)
        isolated['shutil'] = types.SimpleNamespace(which=lambda _name: '/usr/bin/g++')
        for version, expected in [('2.35', (2, 35)), ('2.39', (2, 39))]:
            isolated['platform'] = types.SimpleNamespace(libc_ver=lambda: ('glibc', version))
            self.assertEqual(isolated['check_host'](), expected)
        isolated['platform'] = types.SimpleNamespace(libc_ver=lambda: ('glibc', '2.34'))
        with self.assertRaises(RuntimeError):
            isolated['check_host']()

    def test_glibc235_does_not_silently_accept_unpatched_v4(self):
        self.archive_with()
        self.run_ensure('--install-only')
        before = (self.prefix / self.code['RECEIPT']).read_bytes()
        self.code['check_host'] = lambda: (2, 35)
        with self.assertRaisesRegex(RuntimeError, 'not glibc-2.35 compatible'):
            self.run_ensure('--install-only')
        self.assertEqual((self.prefix / self.code['RECEIPT']).read_bytes(), before)

    def test_glibc235_requires_real_pinned_overlay(self):
        self.archive_with()
        self.code['check_host'] = lambda: (2, 35)
        self.code.update(COMPAT_SHA=None, COMPAT_BYTES=None, COMPAT_REVISION=None)
        with self.assertRaisesRegex(RuntimeError, 'no published overlay pin'):
            self.run_ensure('--install-only')
        self.assertFalse(self.prefix.exists())

    def test_overlay_download_verifies_bytes_and_uses_selected_proxy(self):
        import os
        self.make_overlay()
        payload = self.overlay.read_bytes()
        self.overlay.unlink()
        for proxy in ['direct', 'http://fixture-proxy:80']:
            calls = []

            def download(command, **_kwargs):
                calls.append(command)
                pathlib.Path(command[command.index('-o') + 1]).write_bytes(payload)

            self.code['os'] = types.SimpleNamespace(environ={'CTOPD_PROXY': proxy}, close=os.close)
            self.code['subprocess'] = types.SimpleNamespace(run=download)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.code['compatibility_overlay'](self.root), self.overlay)
            self.assertEqual(self.overlay.read_bytes(), payload)
            self.assertEqual(calls[0][-2:], ['--noproxy', '*' if proxy == 'direct' else ''])
            self.assertEqual(list(self.overlay.parent.glob('.overlay-download-*')), [])
            self.overlay.unlink()

    def test_invalid_cached_overlay_is_preserved(self):
        self.make_overlay()
        self.overlay.write_bytes(b'invalid-existing-overlay')
        with self.assertRaisesRegex(RuntimeError, 'preserved'):
            self.code['compatibility_overlay'](self.root)
        self.assertEqual(self.overlay.read_bytes(), b'invalid-existing-overlay')

    def test_glibc235_installs_overlay_and_quarantines_old_host_libraries(self):
        self.make_overlay()
        self.code['check_host'] = lambda: (2, 35)
        self.run_ensure('--install-only')
        self.assertEqual((self.prefix / 'host-libs/libstdc++.so.6').read_bytes(), b'compatible-library')
        self.assertEqual((self.prefix / 'host-libs-v4-disabled/old-incompatible.so').read_bytes(), b'old-v4-host-library')
        receipt = json.loads((self.prefix / self.code['RECEIPT']).read_text())
        self.assertEqual(receipt['compatibility_overlay']['sha256'], self.code['COMPAT_SHA'])
        self.overlay.unlink()
        self.archive.unlink()
        self.assertIn('REUSING_RUNTIME=', self.run_ensure('--install-only'))

    def test_overlay_cannot_modify_training_package(self):
        self.make_overlay(forbidden=True)
        self.code['check_host'] = lambda: (2, 35)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Disallowed ABI overlay member'):
                self.run_ensure('--install-only')
        self.assertFalse(self.prefix.exists())

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
