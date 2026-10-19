"""Keep historical freeze manifests while auditing the explicitly patched adapter."""
import hashlib
import json


def assert_source_hash(case, root, row):
    expected = row['sha256']
    if row['path'] == 'kdflow/algorithms/mp_opd.py':
        patch = json.loads((root / 'docs/MP_OPD_INVALID_RESPONSES.json').read_text())
        case.assertEqual(expected, patch['baseline_sha256'])
        expected = patch['updated_sha256']
    case.assertEqual(hashlib.sha256((root / row['path']).read_bytes()).hexdigest(), expected)
