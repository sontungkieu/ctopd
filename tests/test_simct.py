"""Author-source oracle parity. Requires existing torch; never installs packages."""
import ast
import copy
import hashlib
import json
import logging
import pathlib
import types
import unittest
from mp_opd_source_checks import assert_source_hash

import test_mp_opd_port as fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
if fixtures.HAS_TORCH:
    import torch


def original_oracle_class():
    """Execute original numerical bodies, stripping only imports/decorators.

    The oracle does not import the port, adapter, or CTOPD RKL implementation.
    """
    source = ROOT / 'tests/fixtures/simct_upstream'
    rkl_tree = ast.parse((source / 'reverse_kl_div.py.txt').read_text())
    rkl = next(n for n in rkl_tree.body if isinstance(n, ast.FunctionDef))
    rkl.decorator_list = []
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[rkl], type_ignores=[]), 'original_rkl', 'exec'), scope)
    cls_tree = ast.parse((source / 'span_ctkd.py.txt').read_text())
    cls = next(n for n in cls_tree.body if isinstance(n, ast.ClassDef))
    cls.decorator_list = []
    scope.update(build_loss_fn=lambda *_: scope['compute_reverse_kl_div'],
                 logger=logging.getLogger('simct-oracle'))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'original_simct', 'exec'), scope)
    return scope['SpanCrossTokenizerKD']


@unittest.skipUnless(fixtures.HAS_TORCH, 'requires existing PyTorch; no bootstrap')
class SimCTTests(unittest.TestCase):
    parity_errors = []

    @classmethod
    def tearDownClass(cls):
        if cls.parity_errors:
            print('SIMCT_PARITY ' + json.dumps({
                'cases': len(cls.parity_errors),
                'max_loss_abs_error': max(x[0] for x in cls.parity_errors),
                'max_student_gradient_abs_error': max(x[1] for x in cls.parity_errors)}))

    def pair(self, *, threshold=0.0, cap=None):
        from kdflow.algorithms.simct import SimCT
        helper = fixtures.MPPortTests()
        mp, student, teacher = helper.fixture(cap=cap)
        args = mp.args
        args.kd.kd_algorithm = 'simct'
        args.kd.simct_gh_mask_threshold = args.kd.span_gh_mask_threshold = threshold
        actual = SimCT(mp.strategy, student, teacher, fixtures.Tokenizer(), fixtures.Tokenizer())
        ref_student, ref_teacher = copy.deepcopy(student), copy.deepcopy(teacher)

        class OriginalHFOutput:
            def __call__(self, ids, **kwargs):
                hidden = ref_student(ids, **kwargs)['hidden_states'][-1]
                logits = ref_student.model.lm_head(hidden, skip=False)
                if cap is not None:
                    logits = torch.tanh(logits / cap) * cap
                return {'logits': logits}

        reference = original_oracle_class()(mp.strategy, OriginalHFOutput(), ref_teacher,
                                            fixtures.Tokenizer(), fixtures.Tokenizer())
        reference._debug_count, reference._span_debug_count = 3, 1
        # Set iteration ordering is identical within this process. Assert explicitly.
        self.assertTrue(torch.equal(actual.upstream.student_overlap_token_ids,
                                    reference.student_overlap_token_ids))
        return helper, actual, reference, student, ref_student, teacher, ref_teacher

    def assert_pair(self, *, threshold=0.0, cap=None, copies=1, strict=False):
        helper, actual, reference, student, rs, teacher, rt = self.pair(threshold=threshold, cap=cap)
        batch = helper.batch(copies)
        if strict:
            batch['tea_input_ids'] = batch['stu_input_ids'].clone()
            batch['tea_loss_mask'] = batch['stu_loss_mask'].clone()
            batch['tea_attn_mask'] = batch['stu_attn_mask'].clone()
            batch['teacher_hiddens'] = torch.randn(int(batch['tea_loss_mask'].sum()), 4, requires_grad=True)
        a, b = actual.training_step(batch), reference.training_step(batch)
        torch.testing.assert_close(a['train/loss'], b['loss'], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a['distill/alignment_ratio'], b['align_ratio'])
        torch.testing.assert_close(a['distill/simct_gh_mask_ratio'], b['gh_mask_ratio'])
        ga = torch.autograd.grad(a['train/loss'], tuple(student.parameters()))
        gb = torch.autograd.grad(b['loss'], tuple(rs.parameters()))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        self.parity_errors.append((float((a['train/loss'] - b['loss']).detach().abs()),
                                   max(float((x - y).abs().max()) for x, y in zip(ga, gb))))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in rt.parameters()))
        # Assert teacher/hidden stop-gradient with autograd too, not only .grad buffers.
        grads = torch.autograd.grad(actual.training_step(batch)['train/loss'],
                                    (*teacher.parameters(), batch['teacher_hiddens']), allow_unused=True)
        self.assertTrue(all(g is None for g in grads))

    def test_mixed_spans_loss_and_all_student_gradients_match_original(self):
        self.assert_pair(copies=2)

    def test_strict_only_loss_and_gradients_match_original(self):
        self.assert_pair(strict=True)

    def test_softcap_matches_original_hf_student_output(self):
        self.assert_pair(cap=2.5)

    def test_optional_teacher_mass_mask_matches_original(self):
        for threshold in (0.5, 2.0, 100.0):
            with self.subTest(threshold=threshold):
                self.assert_pair(threshold=threshold)

    def test_virtual_logits_and_derivatives_match_independent_hand_calculation(self):
        _, algo, _, _, _, _, _ = self.pair()
        # First segment mismatches: teacher 'ab' vs student 'a','b'; next is EOS.
        segments = [(0, 1, 0, 2), (1, 2, 2, 3)]
        sl = torch.randn(3, 8, requires_grad=True)
        tl = torch.randn(2, 8)
        sv, tv = algo.upstream._build_virtual_vocab_logits(segments, sl, tl, [1, 2, 0], [4, 0])
        ids, tids = algo.upstream.student_overlap_token_ids, algo.upstream.teacher_overlap_token_ids
        manual_s = torch.stack([torch.cat([sl[0, ids], ((sl[0, 1] + sl[1, 2]) / 2)[None]]),
                                torch.cat([sl[2, ids], sl.new_tensor([-1e9])])])
        manual_t = torch.stack([torch.cat([tl[0, tids], tl[0, 4][None]]),
                                torch.cat([tl[1, tids], tl.new_tensor([-1e9])])])
        torch.testing.assert_close(sv, manual_s)
        torch.testing.assert_close(tv, manual_t)
        p, q = manual_s.log_softmax(-1), manual_t.log_softmax(-1)
        expected = (p.exp() * (p - q)).sum()
        result = algo.upstream.loss_fn(sv, tv, reduction='sum')
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(torch.autograd.grad(result, sl)[0],
                                   torch.autograd.grad(expected, sl)[0])

    def test_response_boundaries_and_microbatch_normalizer(self):
        helper, algo, _, student, _, _, _ = self.pair()
        full = helper.batch(copies=2)
        one = helper.batch()
        one['avg_micro_batch_token_num'] = full['avg_micro_batch_token_num']
        whole = algo.training_step(full)['train/loss']
        split = sum(algo.training_step(one)['train/loss'] for _ in range(2))
        torch.testing.assert_close(whole, split)
        for a, b in zip(torch.autograd.grad(whole, tuple(student.parameters())),
                        torch.autograd.grad(split, tuple(student.parameters()))):
            torch.testing.assert_close(a, b)

    def test_no_alignment_returns_connected_zero_and_bad_shapes_fail(self):
        helper, algo, _, student, _, _, _ = self.pair()
        batch = helper.batch()
        batch['tea_input_ids'][:] = 5
        loss = algo.training_step(batch)['train/loss']
        self.assertEqual(float(loss.detach()), 0.0)
        gradients = torch.autograd.grad(loss, tuple(student.parameters()), allow_unused=True)
        self.assertTrue(all(g is None or not bool(g.any()) for g in gradients))
        batch['teacher_hiddens'] = batch['teacher_hiddens'][:-1]
        with self.assertRaisesRegex(ValueError, 'hidden rows'):
            algo.training_step(batch)

    def test_mode_registration_arguments_and_no_legacy_alias(self):
        from kdflow.algorithms import ALGO_DICT
        from kdflow.arguments.distillation_args import DistillationArguments
        self.assertIn('simct', ALGO_DICT)
        self.assertNotIn('span_ctkd', ALGO_DICT)
        options = dict(kd_algorithm='simct', kd_ratio=1, kd_loss_fn='rkl')
        self.assertEqual(DistillationArguments(**options).simct_gh_mask_threshold, 0)
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                DistillationArguments(**options, simct_gh_mask_threshold=value)
        with self.assertRaises(ValueError):
            DistillationArguments(kd_algorithm='simct')

    def test_unsupported_topology_and_non_on_policy_fail(self):
        from kdflow.algorithms.simct import SimCT
        _, algo, _, student, _, teacher, _ = self.pair()
        algo.args.train.num_gpus_per_node = 2
        with self.assertRaisesRegex(ValueError, 'one student rank'):
            SimCT(algo.upstream.strategy, student, teacher, fixtures.Tokenizer(), fixtures.Tokenizer())
        algo.args.train.num_gpus_per_node = 1
        algo.args.scenario = 'off_policy_kd'
        with self.assertRaisesRegex(ValueError, 'on-policy'):
            SimCT(algo.upstream.strategy, student, teacher, fixtures.Tokenizer(), fixtures.Tokenizer())


class SimCTSourceTests(unittest.TestCase):
    def test_pinned_sources_and_frozen_existing_modes(self):
        manifest = json.loads((ROOT / 'docs/SIMCT_SOURCE.json').read_text())
        for record in manifest['files'] + manifest['unchanged_existing_files']:
            assert_source_hash(self, ROOT, record)
        raw = (ROOT / 'tests/fixtures/simct_upstream/span_ctkd.py.txt').read_bytes()
        port = (ROOT / 'kdflow/algorithms/_simct_upstream/span_ctkd.py').read_bytes()
        self.assertEqual(raw.replace(b'from kdflow.algorithms import register_algorithm',
                                     b'from . import register_algorithm'), port)


if __name__ == '__main__':
    unittest.main()
