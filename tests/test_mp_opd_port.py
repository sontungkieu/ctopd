"""CPU tensor/gradient tests; optional torch is never installed by this suite."""
import hashlib
import importlib.util
import json
import pathlib
import random
import subprocess
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    # Isolated namespace also lets this suite run in a torch-only environment.
    namespace = types.ModuleType('_ctopd_port_tests')
    namespace.__path__ = [str(ROOT / 'kdflow/algorithms')]
    registry = {}

    def register(name):
        def apply(cls):
            registry[name] = cls
            return cls
        return apply

    namespace.register_algorithm = register
    sys.modules[namespace.__name__] = namespace
    spec = importlib.util.spec_from_file_location('_ctopd_port_tests.mp_opd', ROOT / 'kdflow/algorithms/mp_opd.py')
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)


class Tokenizer:
    eos_token_id, pad_token_id = 0, 7
    eos_token, pad_token = '<eos>', '<pad>'
    all_special_ids = [0, 7]
    pieces = {0: '<eos>', 1: 'a', 2: 'b', 3: 'c', 4: 'ab', 5: 'd', 6: 'p', 7: '<pad>'}

    def decode(self, ids, **_kwargs):
        return ''.join(self.pieces[int(i)] for i in ids)

    def convert_ids_to_tokens(self, ids):
        return self.pieces[ids] if isinstance(ids, int) else [self.pieces[int(i)] for i in ids]

    def get_added_vocab(self):
        return {'<eos>': 0, '<pad>': 7}

    def get_vocab(self):
        return {text: index for index, text in self.pieces.items()}

    backend_tokenizer = types.SimpleNamespace(pre_tokenizer='ByteLevel', decoder='ByteLevel')


@unittest.skipUnless(HAS_TORCH, 'requires an existing PyTorch environment; no install/bootstrap')
class MPPortTests(unittest.TestCase):
    def fixture(self, algorithm='mp_opd_atomic', *, source='run', cap=None):
        from _ctopd_port_tests._mp_opd_port._mp_opd_atoms import SimCTAtomizer
        torch.manual_seed(51)

        class Head(torch.nn.Linear):
            def forward(self, value, skip=False):
                return value if skip else super().forward(value)

        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.model = torch.nn.Module()
                self.model.lm_head = Head(4, 8, bias=False)
                self.model.config = types.SimpleNamespace(final_logit_softcapping=cap)

            def forward(self, ids, **_kwargs):
                return {'hidden_states': [self.embedding(ids)]}

        kd = types.SimpleNamespace(kd_ratio=1.0, kd_temperature=1.0, kd_loss_fn='rkl',
            jsd_beta=.5, skew_lambda=.1, adaptive_alpha=.5, hrl_topk=5,
            mp_opd_fixed_span_length=2, mp_opd_random_min_length=1, mp_opd_random_max_length=2,
            mp_opd_random_seed=42, mp_opd_chunk_source=source, mp_opd_chunk_run_length=3,
            mp_opd_chunk_straddle='singleton', mp_opd_alignment_max_comb_len=4,
            mp_opd_grass_geometry='exact_head', mp_opd_grass_sigma_rho=.99,
            mp_opd_grass_sigma_min_pairs=1, mp_opd_align_eps_h=1e-12)
        args = types.SimpleNamespace(kd=kd, train=types.SimpleNamespace(chunked_loss_size=3,
            num_nodes=1, num_gpus_per_node=1), model=types.SimpleNamespace(ring_attn_size=1), scenario='on_policy_kd')
        strategy = types.SimpleNamespace(args=args, ring_attn_group=None)
        student, teacher = Student(), Head(4, 8, bias=False)
        teacher._mp_opd_softcap = cap
        instance = registry[algorithm](strategy, student, teacher, Tokenizer(), Tokenizer())
        return instance, student, teacher

    def batch(self, copies=1):
        # Response-only masks address positions predicting the labels, including EOS.
        sy, ty = [1, 2, 3, 1, 2, 3, 1, 2, 3, 0], [4, 3, 4, 3, 4, 3, 0]
        student_ids = torch.tensor([[6] + sy] * copies)
        teacher_ids = torch.tensor([[6] + ty] * copies)
        sm = torch.ones_like(student_ids, dtype=torch.bool)
        tm = torch.ones_like(teacher_ids, dtype=torch.bool)
        sm[:, -1], tm[:, -1] = False, False
        torch.manual_seed(9)
        th = torch.randn(int(tm.sum()) // copies, 4).repeat(copies, 1).requires_grad_()
        return {'stu_input_ids': student_ids, 'tea_input_ids': teacher_ids,
                'stu_attn_mask': torch.ones_like(student_ids), 'tea_attn_mask': torch.ones_like(teacher_ids),
                'stu_loss_mask': sm, 'tea_loss_mask': tm, 'teacher_hiddens': th,
                'avg_micro_batch_token_num': float(sm.sum())}

    def test_registry_and_lazy_vendor_imports(self):
        self.assertEqual(set(registry), {'mp_opd_atomic', 'mp_opd_fixed', 'mp_opd_random',
                                        'mp_opd_grass_chunk', 'mp_opd_align'})
        # Fresh process: registering classes must not eagerly load the donor kernels.
        script = """import sys,types,importlib.util
p=types.ModuleType('probe');p.__path__=[sys.argv[1]];p.register_algorithm=lambda n:lambda c:c;sys.modules['probe']=p
s=importlib.util.spec_from_file_location('probe.mp_opd',sys.argv[1]+'/mp_opd.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert not any(k.startswith('probe._mp_opd_port') for k in sys.modules)
"""
        subprocess.run([sys.executable, '-c', script, str(ROOT / 'kdflow/algorithms')], check=True)

    def test_fixed_and_random_cover_every_atom_with_short_tail(self):
        for count in [0, 1, 3, 13, 41]:
            for length in range(1, 6):
                part = adapter.fixed_partition(count, length)
                self.assertEqual([i for a, b in part for i in range(a, b)], list(range(count)))
                self.assertTrue(all(b-a == length for a, b in part[:-1]))
            for low, high in [(1, 2), (2, 5)]:
                part = adapter.random_partition(count, low, high, 17)
                self.assertEqual([i for a, b in part for i in range(a, b)], list(range(count)))
                self.assertTrue(all(low <= b-a <= high for a, b in part[:-1]))
                self.assertEqual(part, adapter.random_partition(count, low, high, 17))

    def test_random_seed_ignores_microbatch_layout_and_global_rng(self):
        state = random.getstate()
        seed = adapter.response_seed(42, [1, 2, 3], [4, 3])
        self.assertNotEqual(seed, adapter.response_seed(43, [1, 2, 3], [4, 3]))
        adapter.random_partition(27, 2, 5, seed)
        self.assertEqual(state, random.getstate())

    def test_atomic_matches_donor_loss_and_every_student_gradient(self):
        from _ctopd_port_tests._mp_opd_port._mp_opd_credit import build_atom_credits, soft_partition_loss
        algo, student, teacher = self.fixture()
        batch = self.batch()
        result = algo.training_step(batch)['train/loss']
        actual = torch.autograd.grad(result, tuple(student.parameters()))
        sy = batch['stu_input_ids'].roll(-1, 1)[batch['stu_loss_mask']]
        ty = batch['tea_input_ids'].roll(-1, 1)[batch['tea_loss_mask']]
        atoms = algo.atomizer.atomize(sy.tolist(), ty.tolist(), sample_id='reference').atoms
        sh = student(batch['stu_input_ids'])['hidden_states'][-1][batch['stu_loss_mask']]
        sl = student.model.lm_head(sh, skip=False)
        tl = teacher(batch['teacher_hiddens']).detach()
        credits = build_atom_credits(atoms, sl, sy, tl, ty)
        expected = soft_partition_loss(credits.current_nll, credits.rate) / batch['avg_micro_batch_token_num']
        reference = torch.autograd.grad(expected, tuple(student.parameters()))
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        self.assertIsNone(teacher.weight.grad)
        self.assertIsNone(batch['teacher_hiddens'].grad)

    def test_fixed1_matches_atomic_and_fixed_pooling_conserves_credit(self):
        atomic, _, _ = self.fixture()
        fixed, _, _ = self.fixture('mp_opd_fixed')
        fixed.kd.mp_opd_fixed_span_length = 1
        torch.testing.assert_close(atomic.training_step(self.batch())['train/loss'],
                                   fixed.training_step(self.batch())['train/loss'])
        from _ctopd_port_tests._mp_opd_port._mp_opd_grass_span import hard_pooled_credits
        base, weight = torch.tensor([2., -3., 7.]), torch.tensor([1., 2., 4.])
        pooled = hard_pooled_credits(base, weight, ((0, 2), (2, 3)))
        torch.testing.assert_close((pooled * weight).sum(), base.sum().double())

    def test_response_boundary_and_microbatch_gradient_normalization(self):
        for mode in registry:
            algo, student, _ = self.fixture(mode)
            full = algo.training_step(self.batch(2))['train/loss']
            grad_full = torch.autograd.grad(full, tuple(student.parameters()))
            algo, split_student, _ = self.fixture(mode)
            single = self.batch()
            single['avg_micro_batch_token_num'] *= 2
            # Same optimizer-window token denominator across both microbatches.
            split = algo.training_step(single)['train/loss'] + algo.training_step(single)['train/loss']
            grad_split = torch.autograd.grad(split, tuple(split_student.parameters()))
            torch.testing.assert_close(full, split, atol=2e-6, rtol=1e-5)
            for a, b in zip(grad_full, grad_split):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)

    def test_all_modes_backward_finite_with_run_and_native_alignment(self):
        for mode in registry:
            for source in ['run', 'xtoken'] if mode in {'mp_opd_align', 'mp_opd_grass_chunk'} else ['run']:
                algo, student, teacher = self.fixture(mode, source=source, cap=3.0)
                info = algo.training_step(self.batch())
                self.assertTrue(all(bool(torch.isfinite(value)) for value in info.values()))
                info['train/loss'].backward()
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in student.parameters()))
                self.assertIsNone(teacher.weight.grad)

    def test_gram_matches_explicit_lm_head_autograd_with_softcap(self):
        from _ctopd_port_tests._mp_opd_port._mp_opd_grass_chunk import chunk_head_gram
        torch.manual_seed(33)
        hidden = torch.randn(4, 3)
        head = torch.nn.Linear(3, 7, bias=False)
        labels = torch.tensor([0, 1, 2, 3])
        logits = torch.tanh(head(hidden) / 2) * 2
        nll = -logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
        spans = ((0, 2), (2, 3), (3, 4))
        grads = [torch.autograd.grad(nll[a:b].sum(), head.weight, retain_graph=True)[0].flatten() for a, b in spans]
        expected = torch.stack(grads) @ torch.stack(grads).T
        actual = chunk_head_gram(logits, hidden, labels, spans, ((0, 3),), selected_log_prob=-nll, softcap=2).gram
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)

    def test_numerical_token_tiling_does_not_change_loss_or_gradients(self):
        for width in [1, 3, 100]:
            algo, student, _ = self.fixture('mp_opd_grass_chunk')
            algo.args.train.chunked_loss_size = width
            loss = algo.training_step(self.batch())['train/loss']
            gradients = torch.autograd.grad(loss, tuple(student.parameters()))
            if width == 1:
                reference, reference_gradients = loss.detach(), gradients
            else:
                torch.testing.assert_close(loss, reference, atol=2e-6, rtol=1e-5)
                for actual, expected in zip(gradients, reference_gradients):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_align_projection_has_effect_and_is_not_credit_conserving_pooling(self):
        from _ctopd_port_tests._mp_opd_port._mp_opd_align import align_chunks
        rates, gram = torch.tensor([1., 2.]), torch.tensor([[1., -.5], [-.5, 1.]])
        actual = align_chunks(rates, gram, ((0, 2),))
        self.assertGreater(actual['projections'], 0)
        self.assertFalse(torch.equal(actual['rate'], rates))
        self.assertEqual(actual['conflict_fraction_post'], 0)

    def test_grass_singletons_and_zero_noise_are_identity_and_conserve_credit(self):
        from _ctopd_port_tests._mp_opd_port._mp_opd_grass_chunk import grass_chunk_tables, apply_chunk_shrinkage
        rate, weight = torch.tensor([2., -1., 4.]), torch.tensor([1., 3., 2.])
        for part, sigma in [(((0, 1), (1, 2), (2, 3)), 3.), (((0, 3),), 0.)]:
            tables = grass_chunk_tables(rate, weight, torch.eye(3), part, sigma)
            effective, residual = apply_chunk_shrinkage(rate, weight, tables)
            torch.testing.assert_close(effective.float(), rate)
            self.assertLess(float(residual), 1e-12)
        tables = grass_chunk_tables(rate, weight, torch.eye(3), ((0, 3),), 50.)
        effective, residual = apply_chunk_shrinkage(rate, weight, tables)
        torch.testing.assert_close((effective * weight).sum(), (rate * weight).sum().double())
        self.assertLess(float(residual), 1e-12)

    def test_atomization_mismatch_fails_instead_of_falling_back_to_atomic(self):
        algo, _, _ = self.fixture('mp_opd_align')
        batch = self.batch()
        batch['tea_input_ids'][0, 1] = 5
        with self.assertRaisesRegex(ValueError, 'atomization failed'):
            algo.training_step(batch)

    def test_noise_state_roundtrip_and_mode_guard(self):
        algo, _, _ = self.fixture('mp_opd_grass_chunk')
        algo.training_step(self.batch())
        other, _, _ = self.fixture('mp_opd_grass_chunk')
        other.load_state_dict(algo.state_dict())
        self.assertEqual(other.noise.state_dict(), algo.noise.state_dict())
        atomic, _, _ = self.fixture()
        with self.assertRaises(ValueError):
            atomic.load_state_dict(algo.state_dict())
        other.kd.mp_opd_chunk_run_length += 1
        with self.assertRaises(ValueError):
            other.load_state_dict(algo.state_dict())

    def test_argument_defaults_and_new_mode_validation(self):
        from kdflow.arguments.distillation_args import DistillationArguments
        legacy = DistillationArguments()
        self.assertEqual((legacy.kd_algorithm, legacy.kd_loss_fn, legacy.kd_ratio), ('byte_align_ctkd', 'kl', .5))
        for mode in registry:
            args = DistillationArguments(kd_algorithm=mode, kd_loss_fn='rkl', kd_ratio=1.)
            self.assertEqual(args.mp_opd_chunk_source, 'xtoken')
        for bad in [{'mp_opd_random_min_length': 0}, {'mp_opd_random_max_length': 0},
                    {'mp_opd_chunk_source': 'unknown'}, {'mp_opd_align_eps_h': float('nan')}]:
            with self.assertRaises(ValueError):
                DistillationArguments(kd_algorithm='mp_opd_align', kd_loss_fn='rkl', kd_ratio=1., **bad)

    def test_initial_adapter_rejects_unqualified_topologies(self):
        algo, student, teacher = self.fixture()
        algo.args.train.num_gpus_per_node = 2
        with self.assertRaisesRegex(ValueError, 'one student rank'):
            registry['mp_opd_atomic'](algo.strategy, student, teacher, Tokenizer(), Tokenizer())

    def test_head_precision_setting_is_restored_even_on_exception(self):
        from _ctopd_port_tests._mp_opd_port._mp_opd_grass_span import _exact_fp32
        backend = torch.backends.cuda.matmul
        setting = 'fp32_precision' if hasattr(backend, 'fp32_precision') else 'allow_tf32'
        before = getattr(backend, setting)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with _exact_fp32():
                raise RuntimeError('fixture')
        self.assertEqual(getattr(backend, setting), before)

    def test_existing_strict_losses_and_gradients_match_direct_kl_reference(self):
        from kdflow.algorithms import ALGO_DICT
        for name in ['simple_ctkd', 'byte_align_ctkd', 'topk_byte_align_ctkd']:
            initial, student, teacher = self.fixture()
            initial.args.kd.ctkd_topk = 2
            algo = ALGO_DICT[name](initial.strategy, student, teacher, Tokenizer(), Tokenizer())
            batch = self.batch()
            actual = algo.training_step(batch)['train/loss']
            actual_grad = torch.autograd.grad(actual, tuple(student.parameters()))
            sm, tm = batch['stu_loss_mask'], batch['tea_loss_mask']
            sy, ty = batch['stu_input_ids'].roll(-1, 1), batch['tea_input_ids'].roll(-1, 1)
            ti, si = algo._get_alignment_indices(ty, sy, tm, sm)
            sh = student(batch['stu_input_ids'])['hidden_states'][-1][sm][si]
            sl = student.model.lm_head(sh, skip=False)[:, algo.student_overlap_token_ids]
            tl = teacher(batch['teacher_hiddens'][ti]).detach()[:, algo.teacher_overlap_token_ids]
            if name == 'topk_byte_align_ctkd':
                indices = sl.topk(2, dim=-1).indices
                sl, tl = sl.gather(-1, indices), tl.gather(-1, indices)
            log_s, log_t = sl.log_softmax(-1), tl.log_softmax(-1)
            expected = (log_s.exp() * (log_s - log_t)).sum() / batch['avg_micro_batch_token_num']
            expected_grad = torch.autograd.grad(expected, tuple(student.parameters()))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


class PortSourceTests(unittest.TestCase):
    def test_vendored_source_hashes(self):
        manifest = json.loads((ROOT / 'docs/MP_OPD_SOURCE.json').read_text())
        for row in manifest['files']:
            self.assertEqual(hashlib.sha256((ROOT / row['destination']).read_bytes()).hexdigest(), row['ported_sha256'])

    def test_strict_sources_are_byte_identical_to_pre_port_checkpoint(self):
        manifest = json.loads((ROOT / 'docs/MP_OPD_SOURCE.json').read_text())
        for row in manifest['unchanged_strict_files']:
            self.assertEqual(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest(), row['sha256'])


if __name__ == '__main__':
    unittest.main()
