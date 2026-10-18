#!/usr/bin/env bash
# Standalone node-local restore/check entry point. Safe to copy into ctopd_all/.
set -euo pipefail
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'CTOPD requires Linux x86_64; do not install this runtime on Windows.' >&2
  exit 2
fi
SYSTEM_PY=""
for candidate in python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then SYSTEM_PY="$(command -v "$candidate")"; break; fi
done
[[ -n "$SYSTEM_PY" ]] || { echo 'Need a system Python 3 interpreter.' >&2; exit 2; }
exec "$SYSTEM_PY" - "$@" <<'CTOPD_ENSURE_PY'
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

EXPECTED_SHA = '084c321c0b7af1cd3a36d12a595c623567d4a37525aa0eecbc3933aa7f9b15c3'
EXPECTED_BYTES = 16515065867
MIN_FREE_BYTES = 45 * 1024**3
RECEIPT = '.ctopd-runtime.json'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_host():
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 39):
        raise RuntimeError('Need glibc >= 2.39 (Ubuntu 24.04 baseline); change pod image, not system libc')
    if not shutil.which('g++'):
        raise RuntimeError('Host g++ is required for runtime JIT; use a prepared node image')


def fingerprints(prefix):
    interpreters = list(prefix.glob('python/cpython-3.12.12-*/bin/python3.12'))
    if len(interpreters) != 1:
        raise RuntimeError('Portable Python is missing or ambiguous in ' + str(prefix))
    paths = [prefix / 'python-b200.sh', interpreters[0], prefix / 'cuda/bin/nvcc']
    for path in paths:
        if not path.is_file():
            raise RuntimeError('Missing runtime component: ' + str(path))
    return {str(path.relative_to(prefix)): digest(path) for path in paths}


def restore(archive, prefix):
    if not hasattr(tarfile, 'data_filter'):
        raise RuntimeError('System Python needs safe tar extraction filters (Python 3.12 or patched 3.10/3.11)')
    if archive.stat().st_size != EXPECTED_BYTES or digest(archive) != EXPECTED_SHA:
        raise RuntimeError('Archive size/SHA-256 mismatch; reassemble numbered parts correctly before installing')
    print('ARCHIVE_SHA256_OK=' + EXPECTED_SHA, flush=True)
    if shutil.disk_usage(prefix.parent).free < MIN_FREE_BYTES:
        raise RuntimeError('Need at least 45 GiB free on the target filesystem for extraction')
    stage = pathlib.Path(tempfile.mkdtemp(prefix='.' + prefix.name + '.install-', dir=prefix.parent))
    print('EXTRACTING_TO=' + str(stage), flush=True)
    try:
        with tarfile.open(archive, 'r:gz') as bundle:
            for original in bundle:
                path = pathlib.PurePosixPath(original.name)
                if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('opt', 'ctopd-b200'):
                    raise ValueError('Unexpected archive member: ' + str(path))
                if len(path.parts) == 2:
                    continue
                member = copy.copy(original)
                member.name = str(pathlib.PurePosixPath(*path.parts[2:]))
                if member.islnk():
                    link = pathlib.PurePosixPath(member.linkname)
                    if link.is_absolute() or '..' in link.parts or link.parts[:2] != ('opt', 'ctopd-b200'):
                        raise ValueError('Unsafe hardlink: ' + member.linkname)
                    member.linkname = str(pathlib.PurePosixPath(*link.parts[2:]))
                bundle.extract(member, stage, filter='data')
        receipt = {'archive_sha256': EXPECTED_SHA, 'archive_bytes': EXPECTED_BYTES,
                   'installed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'critical_files_sha256': fingerprints(stage)}
        (stage / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')
        if prefix.exists() or prefix.is_symlink():
            raise FileExistsError(prefix)
        stage.rename(prefix)
    except Exception:
        print('INSTALL_FAILED: staging retained for inspection at ' + str(stage), file=sys.stderr)
        raise


PROBE = '''import importlib, importlib.metadata as md, json, os, pathlib, sys
import torch
assert torch.cuda.device_count() == 1, 'Expose exactly the B200 allocated to this check'
assert 'B200' in torch.cuda.get_device_name(0), 'Expected a B200'
expected = {'torch': '2.11.0+cu129', 'transformers': '5.6.2', 'sglang': '0.5.12.post1',
            'flash-attn': '2.8.3', 'flashinfer-python': '0.6.11.post1'}
versions = {name: md.version(name) for name in expected}
assert versions == expected, ('Runtime package mismatch', versions)
assert sys.version_info[:3] == (3, 12, 12), sys.version
cuda = pathlib.Path(os.environ['CUDA_HOME'])
assert (cuda / 'include/nvrtc.h').is_file(), 'Missing NVRTC headers'
for name in ['transformers', 'sglang.srt.entrypoints.engine', 'flash_attn', 'flashinfer']:
    importlib.import_module(name)
assert torch.version.cuda == '12.9', torch.version.cuda
value = torch.tensor([1.0], device='cuda')
assert value.cpu().item() == 1.0
print(json.dumps({'versions': versions, 'python': sys.version.split()[0],
                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'check': 'imports + CUDA memory roundtrip; no training or full kernel suite'}))
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description='Restore/check the pinned CTOPD runtime on this node; no pip, build or training')
    parser.add_argument('--root', type=pathlib.Path,
        default=pathlib.Path(os.environ.get('CTOPD_ROOT', '/workspace/storage-shared/nlp/tungks/ctopd_all')))
    parser.add_argument('--prefix', type=pathlib.Path,
        default=pathlib.Path(os.environ.get('CTOPD_RUNTIME_PREFIX', '/opt/ctopd-b200')))
    parser.add_argument('--archive', type=pathlib.Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--install-only', action='store_true', help='Restore without a GPU check; reports INSTALLED_UNCHECKED')
    mode.add_argument('--check-only', action='store_true', help='Check an existing managed runtime; never extract')
    parser.add_argument('python_args', nargs=argparse.REMAINDER, help='After --, execute these Python arguments through the runtime wrapper')
    args = parser.parse_args(argv)
    forwarded = args.python_args[1:] if args.python_args[:1] == ['--'] else args.python_args
    if args.install_only and forwarded:
        raise ValueError('--install-only cannot also execute a runtime command')
    raw = args.prefix.expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise ValueError('Prefix must be an absolute non-symlink directory')
    prefix = raw.resolve()
    if len(prefix.parts) < 3 or prefix == args.root.expanduser().resolve():
        raise ValueError('Refusing an unsafe runtime prefix: ' + str(prefix))
    check_host()
    if args.check_only and not prefix.exists():
        raise FileNotFoundError('Runtime is absent; run without --check-only to restore it')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock_path = prefix.parent / ('.' + prefix.name + '.ensure.lock')
    if lock_path.is_symlink():
        raise ValueError('Refusing symlink lock file')
    with lock_path.open('a') as lock:
        print('WAITING_FOR_RUNTIME_LOCK=' + str(lock_path), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if prefix.exists():
            if not prefix.is_dir() or not (prefix / RECEIPT).is_file():
                raise RuntimeError('Existing unmanaged prefix preserved; choose another --prefix or inspect it manually: ' + str(prefix))
            receipt = json.loads((prefix / RECEIPT).read_text())
            if receipt.get('archive_sha256') != EXPECTED_SHA or receipt.get('critical_files_sha256') != fingerprints(prefix):
                raise RuntimeError('Existing runtime identity/integrity check failed; preserved without replacement')
            print('REUSING_RUNTIME=' + str(prefix), flush=True)
        else:
            if args.check_only:
                raise FileNotFoundError('Runtime disappeared while waiting for its lock; no extraction performed')
            root = args.root.expanduser().resolve()
            candidates = [root / 'ctopd-b200-cu129-v4/runtime.tar.gz', root / 'runtime.tar.gz']
            archive = args.archive.expanduser().resolve() if args.archive else next((p for p in candidates if p.is_file()), None)
            if archive is None or not archive.is_file():
                raise FileNotFoundError('No runtime.tar.gz in ' + str(root) + '; download once to shared storage or pass --archive')
            restore(archive, prefix)
        if args.install_only:
            print('RUNTIME_INSTALLED_UNCHECKED=' + str(prefix), flush=True)
        else:
            env = os.environ.copy()
            env.update(PYTHONPATH='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1')
            output = subprocess.check_output(['bash', str(prefix / 'python-b200.sh'), '-c', PROBE], env=env, text=True)
            result = json.loads(output.strip().splitlines()[-1])
            result['checked_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (prefix / '.ctopd-last-check.json').write_text(json.dumps(result, indent=2) + '\n')
            print(output, end='', flush=True)
            print('RUNTIME_READY=' + str(prefix), flush=True)
        print('RUNTIME_PYTHON_WRAPPER=' + str(prefix / 'python-b200.sh'), flush=True)
    if forwarded:
        os.execvpe('bash', ['bash', str(prefix / 'python-b200.sh'), *forwarded], os.environ.copy())


if __name__ == '__main__':
    main()
CTOPD_ENSURE_PY
