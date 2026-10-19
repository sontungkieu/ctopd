"""Regression for company Fixed1/GRASS/ALIGN invalid-response crashes."""
import hashlib
import copy
import json
import pathlib
import unittest

import test_mp_opd_port as fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
if fixtures.HAS_TORCH:
    import torch


class ReplacementTokenizer(fixtures.Tokenizer):
    pieces = {**fixtures.Tokenizer.pieces, 5: '\ufffd'}


class TransientPrefixTokenizer(fixtures.Tokenizer):
    def decode(self, ids, **kwargs):
        if list(ids) == [1]:
            return '\ufffd'  # Incomplete byte prefix which recovers in full decode.
        return super().decode(ids, **kwargs)


@unittest.skipUnless(fixtures.HAS_TORCH, 'requires existing PyTorch; no bootstrap')
class InvalidResponseTests(unittest.TestCase):
    def fixture(self, mode='mp_opd_fixed'):
        helper = fixtures.MPPortTests()
        algo, student, teacher = helper.fixture(mode)
        algo.kd.mp_opd_fixed_span_length = 1
        algo.atomizer.student_tokenizer = ReplacementTokenizer()
        return helper, algo, student, teacher

    def test_fixed1_mixed_batch_loss_gradients_offsets_and_denominator(self):
        helper, algo, student, _ = self.fixture()
        batch = helper.batch(3)
        batch['stu_input_ids'][0,1] = 5
        # Give each response distinct teacher states to catch offset errors.
        width = int(batch['tea_loss_mask'][0].sum())
        batch['teacher_hiddens'] = (batch['teacher_hiddens'].detach()
            * torch.tensor([1.,2.,3.]).repeat_interleave(width)[:,None]).requires_grad_()
        actual = algo.training_step(batch)
        clean = helper.batch(2)
        clean['avg_micro_batch_token_num'] = batch['avg_micro_batch_token_num']
        clean['teacher_hiddens'] = batch['teacher_hiddens'][width:]
        expected = algo.training_step(clean)['train/loss']
        torch.testing.assert_close(actual['train/loss'], expected)
        actual_grads=torch.autograd.grad(actual['train/loss'],tuple(student.parameters()))
        expected_grads=torch.autograd.grad(expected,tuple(student.parameters()))
        for a,b in zip(actual_grads,expected_grads):
            torch.testing.assert_close(a,b)
        self.assertEqual(float(actual['distill/mp_opd_valid_responses']),2.)
        self.assertEqual(float(actual['distill/mp_opd_attempted_responses']),3.)
        self.assertEqual(float(actual['distill/mp_opd_skipped_replacement_responses']),1.)
        self.assertAlmostEqual(float(actual['distill/mp_opd_skipped_replacement_fraction']),1/3,places=6)
        self.assertEqual(float(actual['distill/mp_opd_skipped_student_loss_tokens']),10.)
        self.assertEqual(float(actual['distill/mp_opd_skipped_teacher_loss_tokens']),7.)
        self.assertAlmostEqual(float(actual['distill/mp_opd_skipped_student_token_fraction']),1/3,places=6)
        # A removed response has no surrogate gradient, including its unique token.
        self.assertFalse(bool(actual_grads[0][5].any()))

    def test_all_replacement_responses_produce_finite_zero_and_backward(self):
        helper,algo,student,_ = self.fixture()
        batch=helper.batch(2)
        batch['stu_input_ids'][:,1]=5
        result=algo.training_step(batch)
        self.assertEqual(float(result['train/loss'].detach()),0.)
        self.assertEqual(float(result['distill/mp_opd_valid_responses']),0.)
        self.assertEqual(float(result['distill/mp_opd_skipped_replacement_fraction']),1.)
        result['train/loss'].backward()
        self.assertTrue(all(p.grad is None or not bool(p.grad.any()) for p in student.parameters()))

    def test_existing_mp_modes_skip_but_other_alignment_errors_stay_fatal(self):
        for mode in fixtures.registry:
            helper,algo,student,_=self.fixture(mode)
            batch=helper.batch(2)
            batch['stu_input_ids'][0,1]=5
            result=algo.training_step(batch)
            self.assertEqual(float(result['distill/mp_opd_skipped_replacement_responses']),1.)
            self.assertTrue(all(bool(torch.isfinite(value)) for value in result.values()))
            result['train/loss'].backward()
            self.assertTrue(any(p.grad is not None for p in student.parameters()))
        helper,algo,*_=self.fixture()
        bad=helper.batch()
        bad['tea_input_ids'][0,1]=7  # Unsupported added token remains fatal.
        with self.assertRaisesRegex(ValueError,'unsupported_added_token'):
            algo.training_step(bad)

    def test_normalization_mismatch_loss_and_gradients_for_all_mp_modes(self):
        import test_align_new
        for mode in (*fixtures.registry, 'align_new'):
            with self.subTest(mode=mode):
                if mode == 'align_new':
                    _,algo,_,_,_,_=test_align_new.AlignNewTests().setup_pair()
                    helper=fixtures.MPPortTests()
                    student=algo.student
                else:
                    helper,algo,student,_=self.fixture(mode)
                batch=helper.batch(2)
                batch['tea_input_ids'][0,1]=6
                # Distinct hidden rows ensure a rejected first row cannot shift
                # the teacher states used by the next valid response.
                width=int(batch['tea_loss_mask'][0].sum())
                batch['teacher_hiddens']=batch['teacher_hiddens'].detach().clone()
                batch['teacher_hiddens'][:width]*=3
                state=copy.deepcopy(algo.state_dict())
                actual=algo.training_step(batch)
                clean=helper.batch()
                clean['avg_micro_batch_token_num']=batch['avg_micro_batch_token_num']
                clean['teacher_hiddens']=batch['teacher_hiddens'][width:]
                algo.load_state_dict(state)
                expected=algo.training_step(clean)['train/loss']
                torch.testing.assert_close(actual['train/loss'],expected)
                for a,b in zip(torch.autograd.grad(actual['train/loss'],tuple(student.parameters())),
                               torch.autograd.grad(expected,tuple(student.parameters()))):
                    torch.testing.assert_close(a,b)
                self.assertEqual(float(actual['distill/mp_opd_skipped_normalization_responses']),1.)
                self.assertEqual(float(actual['distill/mp_opd_skipped_replacement_responses']),0.)
                self.assertEqual(float(actual['distill/mp_opd_skipped_normalization_fraction']),.5)
                self.assertEqual(float(actual['distill/mp_opd_skipped_responses']),1.)

    def test_transient_replacement_prefix_is_not_excluded(self):
        helper,algo,*_=self.fixture()
        algo.atomizer.student_tokenizer=TransientPrefixTokenizer()
        result=algo.training_step(helper.batch())
        self.assertEqual(float(result['distill/mp_opd_valid_responses']),1.)
        self.assertEqual(float(result['distill/mp_opd_skipped_responses']),0.)

    def test_align_new_and_trust_full_window_use_same_exclusion(self):
        import test_align_new
        import test_trust_paper
        _,align,_,_,_,_=test_align_new.AlignNewTests().setup_pair()
        align.atomizer.student_tokenizer=ReplacementTokenizer()
        batch=fixtures.MPPortTests().batch(2)
        batch['stu_input_ids'][0,1]=5
        result=align.training_step(batch)
        self.assertEqual(float(result['distill/mp_opd_skipped_replacement_responses']),1.)
        for mode,reason in ((m,r) for m in ('trust_r','trust_b','trust_p')
                            for r in ('replacement','normalization')):
            helper,algo,student,_=test_trust_paper.TrustTests().fixture(mode)
            algo.atomizer.student_tokenizer=ReplacementTokenizer()
            bad,good=helper.batch(),helper.batch()
            if reason == 'replacement':
                bad['stu_input_ids'][0,1]=5
            else:
                bad['tea_input_ids'][0,1]=6
            for b in (bad,good): b['avg_micro_batch_token_num']*=2
            if algo.requires_optimizer_batch:
                algo.prepare_optimizer_batch([bad,good])
                self.assertEqual(float(algo._batch_metrics['mp_opd_'+mode+'_calibration_response_count']),1.)
            result=algo.training_step(bad)
            self.assertEqual(float(result['distill/mp_opd_skipped_'+reason+'_responses']),1.)
            loss=result['train/loss']+algo.training_step(good)['train/loss']
            full_grad=torch.autograd.grad(loss,tuple(student.parameters()))
            algo.finish_optimizer_batch()
            if algo.requires_optimizer_batch:
                algo.strategy.accumulated_gradient=1
                algo.prepare_optimizer_batch([good])
            expected=algo.training_step(good)['train/loss']
            torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-5)
            for a,b in zip(full_grad,torch.autograd.grad(expected,tuple(student.parameters()))):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            algo.finish_optimizer_batch()


class InvalidResponseSourceTests(unittest.TestCase):
    def test_adapter_patch_hash(self):
        policy=json.loads((ROOT/'docs/MP_OPD_INVALID_RESPONSES.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/policy['path']).read_bytes()).hexdigest(),policy['updated_sha256'])


if __name__=='__main__':
    unittest.main()
