"""TRUST donor parity, optimizer-window integration and SimCT paper-score checks."""
import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import pathlib
import random
import sys
import types
import unittest

import test_mp_opd_port as fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
if fixtures.HAS_TORCH:
    import numpy as np
    import torch


def original_trust():
    from kdflow.algorithms._mp_opd_port import _mp_opd_grass_span as span
    from kdflow.algorithms._mp_opd_port import _mp_opd_head_pair as pair
    from kdflow.algorithms._mp_opd_port._mp_opd_grass_chunk import chunk_head_gram
    package = types.ModuleType('_trust_original')
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules['_trust_original._mp_opd_grass_span'] = span
    sys.modules['_trust_original._mp_opd_head_pair'] = pair
    modules = []
    for name in ('_mp_opd_trust','_mp_opd_trust_batch','_mp_opd_trust_p'):
        full = '_trust_original.'+name
        loader = importlib.machinery.SourceFileLoader(full,str(ROOT/'tests/fixtures/trust'/(name+'.py.txt')))
        spec = importlib.util.spec_from_loader(full,loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        loader.exec_module(module)
        modules.append(module)
    scope = dict(torch=torch,chunk_head_gram=chunk_head_gram,
        **{name:getattr(modules[0],name) for name in ('strict_mask','gram_affordable','trust_calibrate','trust_response_loss','trust_response_metrics')})
    exec(compile((ROOT/'tests/fixtures/trust/_trust_r_loss.py.txt').read_text(),'donor_trust_r','exec'),scope)
    return (*modules,scope['_trust_r_loss'])


@unittest.skipUnless(fixtures.HAS_TORCH,'requires existing PyTorch; no bootstrap')
class TrustTests(unittest.TestCase):
    errors = []

    @classmethod
    def tearDownClass(cls):
        if cls.errors:
            print('TRUST_PARITY '+json.dumps({'cases':len(cls.errors),
                'max_loss_abs_error':max(x[0] for x in cls.errors),
                'max_gradient_abs_error':max(x[1] for x in cls.errors)}))

    def fixture(self, mode='trust_b',cap=None):
        from kdflow.algorithms import ALGO_DICT
        helper = fixtures.MPPortTests()
        mp,student,teacher = helper.fixture(cap=cap)
        mp.kd.mp_opd_trust_eps_g = 1e-12
        mp.kd.mp_opd_trust_vocab_chunk = 3
        mp.kd.use_ema_teacher = False
        mp.args.train.use_dynamic_bsz = False
        mp.strategy.accumulated_gradient,mp.strategy.step = 2,0
        algo = ALGO_DICT['mp_opd_'+mode](mp.strategy,student,teacher,fixtures.Tokenizer(),fixtures.Tokenizer())
        return helper,algo,student,teacher

    def rows(self,algo,student,teacher,batch):
        from kdflow.algorithms._mp_opd_port._mp_opd_credit import build_atom_credits
        sm,tm = batch['stu_loss_mask'],batch['tea_loss_mask']
        hidden = student(batch['stu_input_ids'])['hidden_states'][-1][sm]
        sy,ty = batch['stu_input_ids'].roll(-1,1),batch['tea_input_ids'].roll(-1,1)
        so=to=0
        for row in range(sm.shape[0]):
            sid,tid = sy[row][sm[row]].tolist(),ty[row][tm[row]].tolist()
            atomized = algo.atomizer.atomize(sid,tid,sample_id='donor')
            self.assertTrue(atomized.valid)
            ns,nt = atomized.covered_student_events,atomized.covered_teacher_events
            sh = hidden[so:so+ns]
            sl = student.model.lm_head(sh,skip=False)
            tl = teacher(batch['teacher_hiddens'][to:to+nt]).detach()
            if algo.student_cap is not None:
                sl = torch.tanh(sl/algo.student_cap)*algo.student_cap
                tl = torch.tanh(tl/algo.teacher_cap)*algo.teacher_cap
            labels,tlabels = sy[row][sm[row]][:ns],ty[row][tm[row]][:nt]
            credits = build_atom_credits(atomized.atoms,sl,labels,tl,tlabels)
            yield credits,atomized.atoms,sl,labels,sh
            so,to = so+len(sid),to+len(tid)

    def reference(self,algo,student,teacher,batches):
        trust,batch_module,p,r_loss = original_trust()
        if algo.mode == 'trust_r':
            class IdentityMap(dict):
                def __missing__(self,key):
                    return key
            ref = types.SimpleNamespace(grass_softcap=algo.student_cap,grass_head_bias=False,
                trust_eps_g=algo.eps_g,_TRUST_RESPONSE_METRIC_NAMES=IdentityMap(),
                _trust_static_metrics=lambda *args:{})
            total = student.embedding.weight.sum()*0
            for batch in batches:
                for credits,atoms,logits,labels,hidden in self.rows(algo,student,teacher,batch):
                    loss,_ = r_loss(ref,credits,atoms,student_logits=logits,student_labels=labels,student_hidden=hidden)
                    total = total+loss/batch['avg_micro_batch_token_num']
            return total,None
        collector = batch_module.TrustHeadAccumulator(softcap=algo.student_cap,
            head_bias=False,vocab_chunk=3,force_fp32=algo.mode=='trust_p')
        with torch.no_grad():
            for batch in batches:
                for credits,atoms,logits,labels,hidden in self.rows(algo,student,teacher,batch):
                    mask = trust.strict_mask([a.boundary_type for a in atoms])
                    collector.add_response(logits,hidden,labels,[(a.student_start,a.student_end) for a in atoms],
                        credits.rate,mask,-credits.student_token_nll)
        result = collector.finalize(eps_g=algo.eps_g,**({'calibrate':p.trust_p_calibrate_scalars} if algo.mode=='trust_p' else {}))
        fn = p.trust_p_response_loss if algo.mode=='trust_p' else trust.trust_response_loss
        total = student.embedding.weight.sum()*0
        for batch in batches:
            for credits,atoms,*_ in self.rows(algo,student,teacher,batch):
                mask = trust.strict_mask([a.boundary_type for a in atoms])
                total = total+fn(credits.current_nll,credits.rate,mask,~mask,result)/batch['avg_micro_batch_token_num']
        return total,result

    def test_full_loss_and_all_student_gradients_match_donor(self):
        from kdflow.algorithms.mp_opd_trust import restore_rng
        for mode in ('trust_r','trust_b','trust_p'):
            for cap in (None,3.):
                with self.subTest(mode=mode,cap=cap):
                    helper,algo,student,teacher = self.fixture(mode,cap)
                    batches = [helper.batch(),helper.batch()]
                    for b in batches:
                        b['avg_micro_batch_token_num'] *= 2
                    batches[1]['teacher_hiddens'] = (batches[1]['teacher_hiddens']*1.7).detach().requires_grad_()
                    rs,rt = copy.deepcopy(student),copy.deepcopy(teacher)
                    prepared = algo.prepare_optimizer_batch(batches) if algo.requires_optimizer_batch else None
                    if prepared:
                        parts=[]
                        for i,b in enumerate(batches):
                            restore_rng(prepared['rng_states'][i])
                            parts.append(algo.training_step(b)['train/loss'])
                        actual = sum(parts)
                    else:
                        actual = sum(algo.training_step(b)['train/loss'] for b in batches)
                    expected,result = self.reference(algo,rs,rt,batches)
                    ga = torch.autograd.grad(actual,tuple(student.parameters()))
                    gb = torch.autograd.grad(expected,tuple(rs.parameters()))
                    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                    for a,b in zip(ga,gb):
                        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
                    if result is not None:
                        field = 'strict_scale' if mode=='trust_p' else 'lam'
                        self.assertAlmostEqual(getattr(algo._result,field),getattr(result,field),places=6)
                    self.errors.append((float((actual-expected).detach().abs()),max(float((a-b).abs().max()) for a,b in zip(ga,gb))))
                    teacher_grad = torch.autograd.grad(algo.training_step(batches[0])['train/loss'],
                        (*teacher.parameters(),batches[0]['teacher_hiddens']),allow_unused=True)
                    self.assertTrue(all(g is None for g in teacher_grad))
                    algo.finish_optimizer_batch()

    def test_full_batch_cross_response_terms_and_projected_p_match_explicit_autograd(self):
        from kdflow.algorithms._mp_opd_port._mp_opd_trust_batch import TrustHeadAccumulator
        from kdflow.algorithms._mp_opd_port._mp_opd_trust_p import trust_p_calibrate_scalars
        weight = torch.zeros(2,1,requires_grad=True)
        rows = [torch.tensor([[1.],[2.]]),torch.tensor([[-.5],[-3.]])]
        losses=[]
        collector = TrustHeadAccumulator(force_fp32=True,vocab_chunk=1)
        for hidden in rows:
            logits = hidden@weight.T
            labels = torch.ones(2,dtype=torch.long)
            nll = -logits.log_softmax(-1)[:,1]
            losses.append(nll)
            collector.add_response(logits,hidden,labels,((0,1),(1,2)),torch.ones(2),
                torch.tensor([True,False]),-nll.detach())
        strict = sum(loss[0] for loss in losses)
        mismatch = sum(loss[1] for loss in losses)
        gs = torch.autograd.grad(strict,weight,retain_graph=True)[0]
        gm = torch.autograd.grad(mismatch,weight,retain_graph=True)[0]
        args = (float(gs.square().sum()),float(gm.square().sum()),float((gs*gm).sum()),2,2)
        result = collector.finalize(calibrate=trust_p_calibrate_scalars)
        original = original_trust()[2].trust_p_calibrate_scalars(*args)
        self.assertAlmostEqual(result.strict_scale,3.,places=5)
        self.assertEqual(result.strict_scale,original.strict_scale)
        projected = result.strict_scale*strict+mismatch
        gradient = torch.autograd.grad(projected,weight)[0]
        torch.testing.assert_close(gradient,result.strict_scale*gs+gm)
        self.assertGreaterEqual(float((gs*gradient).sum()),0.)
        r = original_trust()[0].trust_calibrate_scalars(*args)
        self.assertEqual(r.lam,0.)
        # Per-response averaging would give (1/2+1/6)/2 = 1/3 instead.
        self.assertNotAlmostEqual(r.lam,1/3)

    def test_missing_window_wrong_boundary_and_normalizers_fail(self):
        helper,algo,*_ = self.fixture()
        with self.assertRaisesRegex(RuntimeError,'prepare_optimizer_batch'):
            algo.training_step(helper.batch())
        with self.assertRaisesRegex(ValueError,'complete optimizer'):
            algo.prepare_optimizer_batch([helper.batch()])
        batches=[helper.batch(),helper.batch()]
        batches[1]['avg_micro_batch_token_num'] *= 2
        with self.assertRaisesRegex(ValueError,'common normalizer'):
            algo.prepare_optimizer_batch(batches)
        self.assertIsNone(algo._result)
        self.assertIsNone(algo._collector)

    def test_actor_two_pass_rng_replay_and_one_optimizer_update(self):
        from kdflow.algorithms.mp_opd_trust import capture_rng
        # Execute the actual actor dispatch methods without importing Ray/CUDA.
        tree=ast.parse((ROOT/'kdflow/ray/train/student_actor.py').read_text())
        methods={}
        for name in ('fit','evaluate'):
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            scope={}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_actor_'+name,'exec'),scope)
            methods[name]=scope[name]
        helper,algo,student,teacher = self.fixture('trust_p')
        records=[]
        original_forward=student.forward
        def forward(_self,ids,**kwargs):
            random.random(); np.random.random()
            hidden=original_forward(ids,**kwargs)['hidden_states'][-1]
            hidden=torch.nn.functional.dropout(hidden,p=.2,training=student.training)
            records.append(hidden.detach().clone())
            return {'hidden_states':[hidden]}
        student.forward=types.MethodType(forward,student)
        optimizer=torch.optim.SGD(student.parameters(),lr=.01)
        scheduler=types.SimpleNamespace(get_last_lr=lambda:[.01])
        strategy=algo.strategy
        updates=[]
        def backward(loss,*_):
            strategy.step=(strategy.step+1)%strategy.accumulated_gradient
            (loss/strategy.accumulated_gradient).backward()
        def step(*_):
            if strategy.step==0:
                optimizer.step();optimizer.zero_grad();updates.append(1)
        strategy.backward=backward
        strategy.optimizer_step=step
        strategy.all_reduce=lambda value,**_:value
        actor=types.SimpleNamespace(student=student,args=algo.args,strategy=strategy,kd_algorithm=algo,
            optim=optimizer,scheduler=scheduler,_prepare_micro_batch=lambda batch:dict(batch))
        batches=[helper.batch(),helper.batch()]
        for b in batches:
            b['avg_micro_batch_token_num'] *= 2
        observed={}
        original_prepare=algo.prepare_optimizer_batch
        def prepare(*args,**kwargs):
            result=original_prepare(*args,**kwargs)
            observed['after']=result['after_rng']
            self.assertEqual(updates,[])
            return result
        algo.prepare_optimizer_batch=prepare
        status=methods['fit'](actor,batches)
        self.assertTrue(np.isfinite(status['train/loss']))
        self.assertEqual(len(updates),1)
        self.assertEqual(len(records),4)
        for a,b in zip(records[:2],records[2:]):
            torch.testing.assert_close(a,b)
        after=capture_rng()
        self.assertEqual(after[0],observed['after'][0])
        self.assertTrue(np.array_equal(after[1][1],observed['after'][1][1]))
        self.assertTrue(torch.equal(after[2],observed['after'][2]))
        self.assertIsNone(algo._result)
        algo.prepare_optimizer_batch=original_prepare
        before=[p.detach().clone() for p in student.parameters()]
        methods['evaluate'](actor,batches)
        self.assertTrue(student.training)
        self.assertIsNone(algo._result)
        for a,b in zip(before,student.parameters()):
            torch.testing.assert_close(a,b)

    def test_only_strict_and_only_mismatch_rules(self):
        trust,_,p,_=original_trust()
        from kdflow.algorithms._mp_opd_port._mp_opd_trust import trust_calibrate_scalars
        from kdflow.algorithms._mp_opd_port._mp_opd_trust_p import trust_p_calibrate_scalars
        for args in ((1.,0.,0.,2,0),(0.,1.,0.,0,2),(0.,0.,0.,0,0),(1.,4.,-2.,1,1)):
            self.assertEqual(trust_calibrate_scalars(*args).lam,trust.trust_calibrate_scalars(*args).lam)
            self.assertEqual(trust_p_calibrate_scalars(*args).strict_scale,p.trust_p_calibrate_scalars(*args).strict_scale)

    def test_batch_microbatch_regrouping_keeps_loss_and_gradients(self):
        for mode in ('trust_b','trust_p'):
            helper,algo,student,_=self.fixture(mode)
            full=helper.batch(2)
            one=helper.batch()
            one['avg_micro_batch_token_num']*=2
            algo.strategy.accumulated_gradient=1
            algo.prepare_optimizer_batch([full])
            whole=algo.training_step(full)['train/loss']
            gw=torch.autograd.grad(whole,tuple(student.parameters()))
            algo.finish_optimizer_batch()
            algo.strategy.accumulated_gradient=2
            algo.prepare_optimizer_batch([one,one])
            split=algo.training_step(one)['train/loss']+algo.training_step(one)['train/loss']
            gs=torch.autograd.grad(split,tuple(student.parameters()))
            algo.finish_optimizer_batch()
            torch.testing.assert_close(whole,split,atol=1e-6,rtol=1e-5)
            for a,b in zip(gw,gs):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_trust_and_paper_argument_guards(self):
        from kdflow.arguments.distillation_args import DistillationArguments
        for mode in ('mp_opd_trust_r','mp_opd_trust_b','mp_opd_trust_p','simct_paper'):
            options=dict(kd_algorithm=mode,kd_ratio=1.,kd_loss_fn='rkl')
            self.assertEqual(DistillationArguments(**options).kd_algorithm,mode)
            with self.assertRaises(ValueError):
                DistillationArguments(kd_algorithm=mode)
            if mode.startswith('mp_opd_trust_'):
                for extra in ({'mp_opd_grass_geometry':'diag'}, {'mp_opd_trust_eps_g':float('nan')},
                              {'mp_opd_trust_vocab_chunk':0}):
                    with self.assertRaises(ValueError):DistillationArguments(**options,**extra)


@unittest.skipUnless(fixtures.HAS_TORCH,'requires existing PyTorch; no bootstrap')
class SimCTPaperTests(unittest.TestCase):
    def pair(self):
        from kdflow.algorithms.simct_paper import SimCTPaper
        helper=fixtures.MPPortTests()
        mp,student,teacher=helper.fixture()
        mp.kd.simct_gh_mask_threshold=0.
        return helper,SimCTPaper(mp.strategy,student,teacher,fixtures.Tokenizer(),fixtures.Tokenizer()),student,teacher

    @staticmethod
    def manual(upstream,segments,sl,tl,sid,tid):
        rows=[[],[]]
        spans=[i for i,(ts,te,ss,se) in enumerate(segments) if te-ts>1 or se-ss>1]
        for logits,ids,overlap,out in ((sl,sid,upstream.student_overlap_token_ids,rows[0]),
                                      (tl,tid,upstream.teacher_overlap_token_ids,rows[1])):
            logp=logits.float().log_softmax(-1)
            student=out is rows[0]
            for index,(ts,te,ss,se) in enumerate(segments):
                lo,hi=(ss,se) if student else (ts,te)
                extras=[logp[lo:hi].gather(1,torch.tensor(ids[lo:hi])[:,None]).mean()
                        if index==span else logp.new_tensor(-1e9) for span in spans]
                out.append(torch.cat([logp[lo,overlap],torch.stack(extras)]) if spans else logp[lo,overlap])
        return torch.stack(rows[0]),torch.stack(rows[1])

    def test_virtual_scores_loss_gradients_and_logit_shift_invariance(self):
        _,algo,*_=self.pair()
        torch.manual_seed(7)
        sl=torch.randn(3,8,requires_grad=True);tl=torch.randn(2,8)
        segments=[(0,1,0,2),(1,2,2,3)]
        sid,tid=[1,2,0],[4,0]
        actual=algo.upstream._build_virtual_vocab_logits(segments,sl,tl,sid,tid)
        expected=self.manual(algo.upstream,segments,sl,tl,sid,tid)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b)
        loss=algo.upstream.loss_fn(*actual,reduction='sum')
        p,q=expected[0].log_softmax(-1),expected[1].log_softmax(-1)
        ref=(p.exp()*(p-q)).sum()
        ga=torch.autograd.grad(loss,sl,retain_graph=True)[0]
        gb=torch.autograd.grad(ref,sl)[0]
        torch.testing.assert_close(loss,ref)
        torch.testing.assert_close(ga,gb)
        shifted_sl=(sl.detach()+torch.tensor([0.,10.,-3.])[:,None]).requires_grad_()
        shifted_tl=tl+torch.tensor([-8.,12.])[:,None]
        shifted=algo.upstream._build_virtual_vocab_logits(segments,shifted_sl,shifted_tl,sid,tid)
        shifted_loss=algo.upstream.loss_fn(*shifted,reduction='sum')
        torch.testing.assert_close(loss,shifted_loss,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(ga,torch.autograd.grad(shifted_loss,shifted_sl)[0],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(ga.sum(-1),torch.zeros(3),atol=1e-6,rtol=1e-5)

    def test_complete_adapter_loss_and_gradient_match_manual_equations(self):
        helper,algo,student,teacher=self.pair()
        rs,rt=copy.deepcopy(student),copy.deepcopy(teacher)
        batch=helper.batch(2)
        actual=algo.training_step(batch)['train/loss']
        sm,tm=batch['stu_loss_mask'],batch['tea_loss_mask']
        sh=rs(batch['stu_input_ids'])['hidden_states'][-1][sm]
        th=batch['teacher_hiddens']
        sy,ty=batch['stu_input_ids'].roll(-1,1),batch['tea_input_ids'].roll(-1,1)
        so=to=0;expected=sh.sum()*0
        for row in range(sm.shape[0]):
            sid,tid=sy[row][sm[row]].tolist(),ty[row][tm[row]].tolist()
            segments,tid,sid=algo.upstream._align_sequences_with_spans(tid,sid)
            sl=rs.model.lm_head(sh[so:so+len(sid)],skip=False)
            tl=rt(th[to:to+len(tid)]).detach()
            sv,tv=self.manual(algo.upstream,segments,sl,tl,sid,tid)
            p,q=sv.log_softmax(-1),tv.log_softmax(-1)
            expected=expected+(p.exp()*(p-q)).sum()/batch['avg_micro_batch_token_num']
            so,to=so+len(sid),to+len(tid)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        for a,b in zip(torch.autograd.grad(actual,tuple(student.parameters())),torch.autograd.grad(expected,tuple(rs.parameters()))):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


class TrustPaperSourceTests(unittest.TestCase):
    def test_source_hashes_and_frozen_existing_modes(self):
        manifest=json.loads((ROOT/'docs/TRUST_PAPER_SOURCE.json').read_text())
        for row in manifest['files']+manifest['unchanged_existing_files']:
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'])


if __name__=='__main__':
    unittest.main()
