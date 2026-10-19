"""Read-only runtime path selection shared by launchers; no CUDA imports."""
import json
import os
import pathlib
import platform

EXPECTED_SHA = '084c321c0b7af1cd3a36d12a595c623567d4a37525aa0eecbc3933aa7f9b15c3'


def default_prefix(host_glibc, base=pathlib.Path('/opt/ctopd-b200')):
    """Choose a node-local prefix without touching either existing install.

    Keep in sync with tools/runtime_paths.py (AST parity tested). This helper
    stays self-contained when copied into the shared ctopd_all directory.
    """
    legacy = base.with_name(base.name + '-u22')
    if base.exists() or base.is_symlink():
        if base.is_symlink():
            return base  # Let the caller reject an unsafe explicit target.
        try:
            receipt = json.loads((base / '.ctopd-runtime.json').read_text())
        except (OSError, ValueError):
            return base  # Preserve/reject an unmanaged or malformed install.
        if (isinstance(receipt, dict) and receipt.get('archive_sha256') == EXPECTED_SHA
                and host_glibc < (2, 39) and not receipt.get('compatibility_overlay')):
            return legacy  # Known original v4 on an older host: use a sibling.
        return base
    if legacy.exists() or legacy.is_symlink():
        return legacy  # Full identity/integrity checks still run before reuse.
    return base


def resolve_runtime(explicit=None, *, environ=None, host_glibc=None, base=pathlib.Path('/opt/ctopd-b200')):
    env = os.environ if environ is None else environ
    configured = explicit or env.get('CTOPD_RUNTIME_PREFIX')
    if configured:
        candidate = pathlib.Path(configured).expanduser()
    else:
        if host_glibc is None:
            libc, version = platform.libc_ver()
            host_glibc = tuple(map(int, version.split('.')[:2])) if libc == 'glibc' and version else (0, 0)
        candidate = default_prefix(host_glibc, base).expanduser()
    if candidate.is_symlink():
        raise ValueError('Runtime prefix must be a non-symlink directory')
    return candidate.resolve()
