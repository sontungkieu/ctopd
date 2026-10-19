"""ALIGN-NEW donor-composition parity and independent output-head geometry checks."""
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import pathlib
import sys
import types
import unittest

import test_mp_opd_port as fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
if fixtures.HAS_TORCH:
    import torch


def original_scope():
    """Original guard/head-pair copies and donor method bodies, no adapter calls.

    Existing v1 Gram/credit/chunk helpers are frozen, source-hash checked dependencies.
    """
    from kdflow.algorithms._mp_opd_port import _mp_opd_grass_span as span
    from kdflow.algorithms._mp_opd_port import _mp_opd_grass_chunk as chunk
    from kdflow.algorithms._mp_opd_port import _mp_opd_align as align
    from kdflow.algorithms._mp_opd_port._mp_opd_credit import soft_partition_loss
    root = ROOT / 'tests/fixtures/align_new'
    package = types.ModuleType('_align_original')
    package.__path__ = [str(root)]
    sys.modules['_align_original'] = package
    sys.modules['_align_original._mp_opd_grass_span'] = span
    for name in ('_mp_opd_head_pair', '_mp_opd_align_new'):
        full = '_align_original.' + name
        loader = importlib.machinery.SourceFileLoader(full, str(root / (name + '.py.txt')))
        spec = importlib.util.spec_from_loader(full, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        loader.exec_module(module)
    pair = sys.modules['_align_original._mp_opd_head_pair']
    guard = sys.modules['_align_original._mp_opd_align_new']
    scope = dict(torch=torch, contextlib=contextlib, chunk_head_gram=chunk.chunk_head_gram,
        chunk_partition=chunk.chunk_partition, align_chunks=align.align_chunks,
        align_metrics=align.align_metrics, head_pair_geometry=pair.head_pair_geometry,
        align_new_guard=guard.align_new_guard, soft_partition_loss=soft_partition_loss,
        run_chunk_assignment=chunk.run_chunk_assignment,
        atom_chunk_ids_from_tokens=chunk.atom_chunk_ids_from_tokens)
    for name in ('_align_loss', '_grass_chunk_assignment'):
        exec(compile((root / (name + '.py.txt')).read_text(), 'donor_' + name, 'exec'), scope)
    return scope


@unittest.skipUnless(fixtures.HAS_TORCH, 'requires existing PyTorch; no bootstrap')
class AlignNewTests(unittest.TestCase):
    errors = []

    @classmethod
    def tearDownClass(cls):
        if cls.errors:
            print('ALIGN_NEW_PARITY ' + json.dumps({'cases': len(cls.errors),
                'max_loss_abs_error': max(v[0] for v in cls.errors),
                'max_gradient_abs_error': max(v[1] for v in cls.errors)}))

    def setup_pair(self, *, source='run', cap=None, rho=.1):
        from kdflow.algorithms.mp_opd_align_new import AlignNewMPOnPolicyKD
        helper = fixtures.MPPortTests()
        mp, student, teacher = helper.fixture(source=source, cap=cap)
        mp.kd.mp_opd_align_new_min_descent_fraction = rho
        mp.kd.mp_opd_align_new_eps_g = 1e-12
        algo = AlignNewMPOnPolicyKD(mp.strategy, student, teacher, fixtures.Tokenizer(), fixtures.Tokenizer())
        scope = original_scope()
        reference = types.SimpleNamespace(mode='align_new', grass_chunk_source=source,
            grass_chunk_run_length=mp.kd.mp_opd_chunk_run_length,
            grass_chunk_straddle=mp.kd.mp_opd_chunk_straddle, grass_aligner=algo.aligner,
            grass_geometry='exact_head', grass_softcap=cap, grass_head_bias=False,
            align_eps_h=mp.kd.mp_opd_align_eps_h, align_new_min_descent_fraction=rho, trust_eps_g=1e-12)
        reference._grass_chunk_assignment = types.MethodType(scope['_grass_chunk_assignment'], reference)
        return helper, algo, student, teacher, scope, reference

    def reference_loss(self, algo, student, teacher, batch, scope, reference):
        from kdflow.algorithms._mp_opd_port._mp_opd_credit import build_atom_credits
        sm, tm = batch['stu_loss_mask'], batch['tea_loss_mask']
        sh = student(batch['stu_input_ids'])['hidden_states'][-1][sm]
        th = batch['teacher_hiddens']
        sy, ty = batch['stu_input_ids'].roll(-1, 1), batch['tea_input_ids'].roll(-1, 1)
        total, so, to = sh.sum() * 0.0, 0, 0
        for row in range(sm.shape[0]):
            sid, tid = sy[row][sm[row]].tolist(), ty[row][tm[row]].tolist()
            atoms = algo.atomizer.atomize(sid, tid, sample_id='original')
            self.assertTrue(atoms.valid)
            ns, nt = atoms.covered_student_events, atoms.covered_teacher_events
            sl = student.model.lm_head(sh[so:so+ns], skip=False)
            tl = teacher(th[to:to+nt]).detach()
            if algo.student_cap is not None:
                sl = torch.tanh(sl / algo.student_cap) * algo.student_cap
                tl = torch.tanh(tl / algo.teacher_cap) * algo.teacher_cap
            labels, tea_labels = sy[row][sm[row]][:ns], ty[row][tm[row]][:nt]
            credits = build_atom_credits(atoms.atoms, sl, labels, tl, tea_labels)
            loss, _ = scope['_align_loss'](reference, credits, atoms.atoms,
                student_logits=sl, student_labels=labels, student_hidden=sh[so:so+ns],
                student_ids=sid[:ns], teacher_ids=tid[:nt])
            total = total + loss
            so, to = so + len(sid), to + len(tid)
        return total / batch['avg_micro_batch_token_num']

    def test_full_adapter_loss_and_gradients_match_original_composition(self):
        for source, cap, rho in [('run', None, .1), ('run', 3., .1), ('xtoken', None, .1),
                                  ('xtoken', 3., .1), ('run', None, 0.), ('run', None, 1.)]:
            with self.subTest(source=source, cap=cap, rho=rho):
                helper, algo, student, teacher, scope, reference = self.setup_pair(source=source, cap=cap, rho=rho)
                rs, rt = copy.deepcopy(student), copy.deepcopy(teacher)
                batch = helper.batch(copies=2)
                actual = algo.training_step(batch)['train/loss']
                expected = self.reference_loss(algo, rs, rt, batch, scope, reference)
                ga = torch.autograd.grad(actual, tuple(student.parameters()))
                gb = torch.autograd.grad(expected, tuple(rs.parameters()))
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                for a, b in zip(ga, gb):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                self.errors.append((float((actual-expected).detach().abs()),
                                    max(float((a-b).abs().max()) for a, b in zip(ga, gb))))
                teacher_grads = torch.autograd.grad(algo.training_step(batch)['train/loss'],
                    (*teacher.parameters(), batch['teacher_hiddens']), allow_unused=True)
                self.assertTrue(all(g is None for g in teacher_grads))

    def test_active_global_guard_matches_original_loss_and_head_gradient(self):
        from kdflow.algorithms._mp_opd_port._mp_opd_credit import AtomCreditTensors
        _, algo, student, _, scope, ref = self.setup_pair()
        ref.grass_chunk_run_length = 2
        with torch.no_grad():
            student.model.lm_head.weight.zero_()
        hidden = torch.tensor([[1.,0.,0.,0.],[-1.,1.,0.,0.],[-.4,-1.,0.,0.]], requires_grad=True)
        logits = student.model.lm_head(hidden, skip=False)
        labels = torch.ones(3, dtype=torch.long)
        nll = -logits.log_softmax(-1)[:, 1]
        rate = torch.ones(3)
        credits = AtomCreditTensors(-nll.detach()+1, -nll.detach(), rate, rate, rate, nll, nll)
        atoms = tuple(types.SimpleNamespace(student_start=i, student_end=i+1) for i in range(3))
        effective, metrics = algo._effective_rates(credits, atoms, ((0,2),(2,3)), hidden, labels)
        actual = (effective*nll).sum()
        expected, donor_metrics = scope['_align_loss'](ref, credits, atoms, student_logits=logits,
            student_labels=labels, student_hidden=hidden, student_ids=[1]*3, teacher_ids=[1]*3)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual, student.model.lm_head.weight, retain_graph=True)[0]
        gb = torch.autograd.grad(expected, student.model.lm_head.weight)[0]
        torch.testing.assert_close(ga, gb)
        beta = float(metrics['mp_opd_align_new_beta'])
        self.assertGreater(beta, 0)
        self.assertLess(beta, 1)
        self.assertGreaterEqual(float(metrics['mp_opd_align_new_retained_descent_fraction']), .1-1e-6)
        torch.testing.assert_close(metrics['mp_opd_align_new_beta'], donor_metrics['mp_opd_align_new_beta'])
        self.errors.append((float((actual-expected).detach().abs()), float((ga-gb).abs().max())))

    def test_full_head_pair_includes_cross_chunk_terms_bias_and_softcap(self):
        from kdflow.algorithms._mp_opd_port._mp_opd_head_pair import head_pair_geometry
        torch.manual_seed(123)
        hidden = torch.randn(5,3)
        weight = torch.randn(7,3, requires_grad=True)
        bias = torch.randn(7, requires_grad=True)
        labels = torch.tensor([0,1,2,3,4])
        ranges = ((0,2),(2,3),(3,5))
        first, second = torch.tensor([1.,-2.,.3]), torch.tensor([-.4,1.,2.])
        raw = hidden @ weight.T + bias
        logits = torch.tanh(raw / 2.) * 2.
        lp = logits.log_softmax(-1).gather(1, labels[:,None]).squeeze(1)
        sums = torch.stack([-lp[a:b].sum() for a,b in ranges])
        ga = torch.cat([g.flatten() for g in torch.autograd.grad((first*sums).sum(), (weight,bias), retain_graph=True)])
        gb = torch.cat([g.flatten() for g in torch.autograd.grad((second*sums).sum(), (weight,bias))])
        expected = (float(ga.double().square().sum()), float(gb.double().square().sum()), float((ga.double()*gb.double()).sum()))
        for tile in (1,3,4096):
            pair = head_pair_geometry(logits.detach(), hidden, labels, ranges, first, second,
                selected_log_prob=lp.detach(), softcap=2., head_bias=True, vocab_chunk=tile)
            torch.testing.assert_close(torch.tensor([pair.first_norm2,pair.second_norm2,pair.dot]), torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_microbatch_invariance_and_no_state_collision_with_v1(self):
        helper, algo, student, _, _, _ = self.setup_pair()
        full, one = helper.batch(2), helper.batch()
        one['avg_micro_batch_token_num'] *= 2
        a = algo.training_step(full)['train/loss']
        b = algo.training_step(one)['train/loss'] + algo.training_step(one)['train/loss']
        torch.testing.assert_close(a,b)
        for x,y in zip(torch.autograd.grad(a, tuple(student.parameters())), torch.autograd.grad(b, tuple(student.parameters()))):
            torch.testing.assert_close(x,y)
        old, _, _ = helper.fixture('mp_opd_align')
        with self.assertRaises(ValueError):
            algo.load_state_dict(old.state_dict())

    def test_ambient_autocast_uses_same_credit_logits_as_original(self):
        from kdflow.algorithms._mp_opd_port._mp_opd_credit import build_atom_credits
        helper, algo, student, teacher, scope, ref = self.setup_pair()
        batch = helper.batch()
        sy = batch['stu_input_ids'].roll(-1,1)[batch['stu_loss_mask']]
        ty = batch['tea_input_ids'].roll(-1,1)[batch['tea_loss_mask']]
        atomized = algo.atomizer.atomize(sy.tolist(), ty.tolist(), sample_id='autocast')
        ns,nt = atomized.covered_student_events,atomized.covered_teacher_events
        hidden = student(batch['stu_input_ids'])['hidden_states'][-1][batch['stu_loss_mask']][:ns]
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            logits = student.model.lm_head(hidden,skip=False)
            teacher_logits = teacher(batch['teacher_hiddens'][:nt])
            credits = build_atom_credits(atomized.atoms,logits,sy[:ns],teacher_logits,ty[:nt])
            partition,_,_ = algo._partition(atomized.atoms,sy[:ns].tolist(),ty[:nt].tolist())
            effective,metrics = algo._effective_rates(credits,atomized.atoms,partition,hidden,sy[:ns])
            actual = (credits.current_nll*effective).sum()
            expected,donor_metrics = scope['_align_loss'](ref,credits,atomized.atoms,
                student_logits=logits,student_labels=sy[:ns],student_hidden=hidden,
                student_ids=sy[:ns].tolist(),teacher_ids=ty[:nt].tolist())
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        for key in ('atomic_head_norm2','candidate_head_norm2','candidate_atomic_dot','beta'):
            name = 'mp_opd_align_new_'+key
            torch.testing.assert_close(metrics[name],donor_metrics[name],atol=1e-5,rtol=1e-5)

    def test_guard_edge_cases_and_invalid_geometry_fail_closed(self):
        from kdflow.algorithms._mp_opd_port._mp_opd_align_new import align_new_guard
        rate, candidate = torch.tensor([1.,2.]), torch.tensor([3.,4.])
        for a,b,d in [(0.,1.,0.), (1.,1.,1.), (1.,1.,-1.)]:
            actual, metrics = align_new_guard(rate,candidate,a,b,d)
            expected, ref_metrics = original_scope()['align_new_guard'](rate,candidate,a,b,d)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(metrics['mp_opd_align_new_beta'],ref_metrics['mp_opd_align_new_beta'])
        for a,b,d in [(1.,1.,2.), (-1.,1.,0.), (float('nan'),1.,0.)]:
            with self.assertRaises((ValueError,FloatingPointError)):
                align_new_guard(rate,candidate,a,b,d)

    def test_argument_guards_and_registry(self):
        from kdflow.arguments.distillation_args import DistillationArguments
        from kdflow.algorithms import ALGO_DICT
        opts = dict(kd_algorithm='mp_opd_align_new', kd_ratio=1., kd_loss_fn='rkl')
        self.assertIn('mp_opd_align_new', ALGO_DICT)
        self.assertEqual(DistillationArguments(**opts).mp_opd_align_new_min_descent_fraction, .1)
        for kwargs in ({'mp_opd_grass_geometry':'diag'}, {'mp_opd_align_new_eps_g':-1},
                       {'mp_opd_align_new_min_descent_fraction':1.1}, {'mp_opd_align_new_eps_g':float('nan')}):
            with self.assertRaises(ValueError):
                DistillationArguments(**opts, **kwargs)


class AlignNewSourceTests(unittest.TestCase):
    def test_source_hashes_and_unchanged_existing_algorithms(self):
        manifest = json.loads((ROOT / 'docs/ALIGN_NEW_SOURCE.json').read_text())
        for row in manifest['files'] + manifest['unchanged_existing_files']:
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(), row['sha256'])


if __name__ == '__main__':
    unittest.main()
