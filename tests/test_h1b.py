"""Offline H1b contracts. Real tokenizer/rollout/B200 gates are separate tools."""
import copy
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
import test_mp_opd_port as fixtures

ROOT=Path(__file__).resolve().parents[1]
if fixtures.HAS_TORCH:
    import torch
    # Test-only namespace: keep the real numerical/logging modules, bypass the
    # utility package's eager tokenizer factory and optional HF dependencies.
    if 'kdflow.utils' not in sys.modules:
        package=types.ModuleType('kdflow.utils'); package.__path__=[str(ROOT/'kdflow/utils')]
        sys.modules['kdflow.utils']=package


@unittest.skipUnless(fixtures.HAS_TORCH,'requires existing torch; no bootstrap')
class H1bTests(unittest.TestCase):
    def fixture(self,arm='a'):
        from kdflow.algorithms.h1b_span import H1bSpanCTKD
        helper=fixtures.MPPortTests(); mp,student,teacher=helper.fixture()
        kd=mp.args.kd
        kd.h1b_arm=arm; kd.h1b_span_lambda=.5; kd.h1b_log_sample_rate=.05; kd.h1b_dump_every='0,20,60,100'
        mp.args.train.save_path=None
        mp.strategy.step=0; mp.strategy.accumulated_gradient=1
        algo=H1bSpanCTKD(mp.strategy,student,teacher,fixtures.Tokenizer(),fixtures.Tokenizer())
        return helper,algo,student,teacher

    def test_T2_cover_matches_bruteforce_and_ff_edges(self):
        from kdflow.h1b.vocab_index import VocabIndex,next_prefix
        vi=VocabIndex(fixtures.Tokenizer())
        rng=random.Random(42)
        for _ in range(2000):
            rem=bytes(rng.choice([97,98,99,100,255]) for _ in range(rng.randint(1,5)))
            self.assertEqual(set(vi.cover(rem)),{i for i,b in vi._bytes.items() if b.startswith(rem) and b})
        self.assertIsNone(next_prefix(b'\xff\xff'))
        self.assertEqual(next_prefix(b'a\xff'),b'b')
        with self.assertRaises(ValueError): vi.cover(b'')

    def test_T3_strict_groups_match_author_fixture_eos_and_offsets(self):
        from kdflow.h1b.groups import align_byte_span_groups_batch
        from kdflow.token_alignment import align_byte_span_batch_1to1,get_tokenizer_byte_codec
        tok=fixtures.Tokenizer(); codec=get_tokenizer_byte_codec(tok).piece_to_bytes
        for tea,stu in [([['ab','c','<eos>'],['a','b']],[['a','b','c','<eos>'],['ab']]),
                        ([['a'],[]],[['a'],[]]),([['<eos>','ab','<eos>']],[['<eos>','a','b','<eos>']])]:
            rows=align_byte_span_groups_batch(tea,stu,tok.eos_token,tok.eos_token,codec,codec)
            groups=[g for r in rows for g in r.groups if g.kind=='strict']
            a=([g.t_idx[0] for g in groups],[g.s_idx[0] for g in groups])
            self.assertEqual(a,align_byte_span_batch_1to1(tea,stu,tok.eos_token,tok.eos_token,codec,codec))

    def test_T9b_row_groups_rejects_unsupported_special_whole_response(self):
        import torch
        from tools.h1b.common import row_groups
        # A sampled reserved special must invalidate the whole response with
        # reason 'unsupported_special', matching the training_step post-filter,
        # instead of reaching the byte vocabulary index (KeyError there).
        class Tok(fixtures.Tokenizer):
            pieces={0:'<eos>',1:'a',2:'b',7:'<pad>',8:'<rsvd>'}
            all_special_ids=[0,7,8]
        tok=Tok()
        ids=torch.tensor([[1,2,0],[1,8,0]])
        mask=torch.tensor([[True,True,False],[True,True,False]])
        batch={'stu_input_ids':ids,'stu_loss_mask':mask,'stu_attn_mask':torch.ones_like(ids),
               'tea_input_ids':ids.clone(),'tea_loss_mask':mask.clone(),'tea_attn_mask':torch.ones_like(ids)}
        rows=row_groups(batch,tok,tok)
        self.assertTrue(rows[0].valid)
        self.assertFalse(rows[1].valid)
        self.assertEqual(rows[1].reason,'unsupported_special')

    def test_T4_trunk_and_exact_enumeration_with_duplicate_byte_ids(self):
        from kdflow.h1b.vocab_index import VocabIndex
        from kdflow.h1b.strprob import chunk_logprobs
        from kdflow.h1b.exact import exact_chunk_logprob
        # 'A' and 'a' map to the same bytes. Exact proper-prefix enumeration
        # must branch on both IDs rather than taking only id_of(...).
        class Tok(fixtures.Tokenizer):
            pieces={0:'<eos>',1:'a',2:'b',3:'ab',4:'A',5:'abc',6:'ba',7:'<pad>'}
        codec=types.SimpleNamespace(piece_to_bytes=lambda s:s.lower().encode())
        vi=VocabIndex(Tok(),codec)
        torch.manual_seed(19)
        table={():torch.randn(8).log_softmax(0),(1,):torch.randn(8).log_softmax(0),(4,):torch.randn(8).log_softmax(0)}
        lp=chunk_logprobs(torch.stack([table[()],table[(1,)]]),[1,2],b'ab',vi)
        expected=torch.logsumexp(torch.stack([table[()][3],table[()][5],table[()][1]+torch.logsumexp(table[(1,)][[2,6]],0)]),0)
        torch.testing.assert_close(lp.log_p_trunk,expected)
        result=exact_chunk_logprob(lambda ids:table[ids],b'ab',vi)
        exact=math.exp(float(expected))+math.exp(float(table[()][4]+torch.logsumexp(table[(4,)][[2,6]],0)))
        self.assertAlmostEqual(result['log_probability_lower'],math.log(exact),places=6)
        self.assertTrue(result['exact'])
        self.assertGreaterEqual(result['log_probability_lower']+1e-6,float(lp.log_p_trunk))
        bounded=exact_chunk_logprob(lambda ids:table[ids],b'ab',vi,max_nodes=1)
        self.assertLessEqual(bounded['log_probability_lower'],math.log(exact)+1e-6)
        self.assertGreaterEqual(bounded['log_probability_upper']+1e-6,math.log(exact))

    def test_T4_thirty_token_lm_fifty_strings(self):
        import itertools
        from kdflow.h1b.vocab_index import VocabIndex
        from kdflow.h1b.strprob import chunk_logprobs
        from kdflow.h1b.exact import exact_chunk_logprob
        words=[''.join(x) for size in range(1,5) for x in itertools.product('ab',repeat=size)]
        class Tok(fixtures.Tokenizer):
            pieces={0:'<eos>',31:'<pad>',**{i+1:w for i,w in enumerate(words)}}
            all_special_ids=[0,31]; pad_token_id=31
        vi=VocabIndex(Tok()); rng=random.Random(17)
        def oracle(path):
            seed=int.from_bytes(hashlib.sha256(bytes(path)).digest()[:4],'big')
            return torch.randn(32,generator=torch.Generator().manual_seed(seed),dtype=torch.float64).log_softmax(0)
        for _ in range(50):
            chunk=''.join(rng.choice('ab') for _ in range(rng.randint(1,4))).encode()
            path=[vi.id_of(bytes([b])) for b in chunk]
            lp=chunk_logprobs(torch.stack([oracle(tuple(path[:j])) for j in range(len(path))]),path,chunk,vi)
            acc=1.; brute=0.
            for j,token in enumerate(path):
                row=oracle(tuple(path[:j])).exp(); rem=chunk[j:]
                brute+=acc*sum(float(row[i]) for i,b in vi._bytes.items() if b.startswith(rem))
                acc*=float(row[token])
            self.assertAlmostEqual(float(lp.log_p_trunk),math.log(brute),places=6)
            exact=exact_chunk_logprob(oracle,chunk,vi)
            self.assertGreaterEqual(exact['log_probability_lower']+1e-6,float(lp.log_p_trunk))

    def test_T5_same_tokenizer_all_arms_have_zero_span(self):
        reference=None
        for arm in ('a','b','b_bl','c_str','c'):
            helper,algo,_,_=self.fixture(arm); batch=helper.batch()
            batch['tea_input_ids']=batch['stu_input_ids'].clone()
            batch['tea_loss_mask']=batch['stu_loss_mask'].clone()
            batch['tea_attn_mask']=batch['stu_attn_mask'].clone()
            batch['teacher_hiddens']=torch.randn(int(batch['tea_loss_mask'].sum()),4)
            info=algo.training_step(batch)
            self.assertEqual(float(info['train/h1b_span_loss'].detach()),0.)
            self.assertEqual(float(info['distill/h1b_mis_groups']),0.)
            value=info['train/loss'].detach()
            if reference is None: reference=value
            else: torch.testing.assert_close(value,reference)

    def test_T6_baseline_previous_window_and_evaluation_immutability(self):
        helper,algo,student,_=self.fixture('c')
        before=algo.baseline.prev.clone()
        algo.training_step(helper.batch())
        torch.testing.assert_close(algo.baseline.prev,before)
        algo.on_optimizer_window_end()
        self.assertTrue(bool(algo.baseline.prev.any()))
        state=copy.deepcopy(algo.state_dict()); counts=algo.baseline.counts.clone()
        student.eval()
        with torch.no_grad(): algo.training_step(helper.batch())
        torch.testing.assert_close(algo.baseline.prev,state['baseline']['prev'])
        torch.testing.assert_close(algo.baseline.counts,counts)
        self.assertEqual(algo.chunk_logger.rows,0)
        algo.load_state_dict(state)

    def test_T7_span_surrogate_gradient_equals_mse(self):
        helper,algo,student,_=self.fixture('b')
        algo.replay_enabled=True
        info=algo.training_step(helper.batch())
        expected=sum(algo.lam*(r['log_q']-(r['log_q'].detach()-r['path']))**2 for r in algo.replay_records)/helper.batch()['avg_micro_batch_token_num']
        a=torch.autograd.grad(info['train/h1b_span_loss'],tuple(student.parameters()),retain_graph=True)
        b=torch.autograd.grad(expected,tuple(student.parameters()))
        for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)

    def test_T8_arm_a_loss_and_all_gradients_equal_strict(self):
        from kdflow.algorithms.byte_align_ctkd import ByteAlignCrossTokenizerKD
        helper,algo,student,teacher=self.fixture('a')
        strict=ByteAlignCrossTokenizerKD(algo.strategy,student,teacher,fixtures.Tokenizer(),fixtures.Tokenizer())
        batch=helper.batch(2)
        a=algo.training_step(batch)['train/loss']; b=strict.training_step(batch)['train/loss']
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        for x,y in zip(torch.autograd.grad(a,tuple(student.parameters())),torch.autograd.grad(b,tuple(student.parameters()))):
            torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
        self.assertIsNone(teacher.weight.grad)

    def test_byte_mismatch_drops_complete_response_including_eos(self):
        helper,algo,student,_=self.fixture('b')
        batch=helper.batch(); batch['tea_input_ids'][0,1]=5
        info=algo.training_step(batch)
        self.assertEqual(float(info['train/loss'].detach()),0.)
        self.assertEqual(float(info['distill/h1b_dropped_responses']),1.)
        info['train/loss'].backward()
        self.assertTrue(all(p.grad is None or not bool(p.grad.any()) for p in student.parameters()))

    def test_tile_recomputation_preserves_path_gradient_and_statistics(self):
        helper,algo,student,_=self.fixture('c_str')
        batch=helper.batch(); a=algo.training_step(batch)
        ga=torch.autograd.grad(a['train/loss'],tuple(student.parameters()))
        algo.args.train.chunked_loss_size=1
        b=algo.training_step(batch); gb=torch.autograd.grad(b['train/loss'],tuple(student.parameters()))
        torch.testing.assert_close(a['train/loss'],b['train/loss'])
        for x,y in zip(ga,gb): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)

    def test_window_statistics_use_chunk_counts_not_microbatch_means(self):
        from kdflow.h1b.chunklog import ChunkLogger
        log=ChunkLogger(sample_rate=0,dump_steps=())
        for v in (1.,2.,9.):
            row=dict(path=v,str=v/2,path_bl=v-1,str_bl=v/2-2,gap=v/2,slack_stu=1.,slack_tea=1+v/2,
                     overhang_stu=0.,overhang_tea=0.,bucket=0,response_hash=str(v),b0=0,b1=1)
            log.add(row)
        out=log.end_window(0)
        self.assertEqual(out['distill/h1b_gap_mean'],2.)
        self.assertAlmostEqual(out['distill/h1b_var_path'],(1+4+81)/3-16)

    def test_actual_actor_hook_runs_after_one_update_and_not_during_eval(self):
        from kdflow.utils.logging_utils import normalize_eval_metrics
        helper,algo,student,_=self.fixture('c')
        tree=ast.parse((ROOT/'kdflow/ray/train/student_actor.py').read_text())
        scope=dict(torch=torch,defaultdict=defaultdict,normalize_eval_metrics=normalize_eval_metrics)
        methods={}
        for name in ('fit','evaluate'):
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_h1b_actor','exec'),scope)
            methods[name]=scope[name]
        strategy=algo.strategy; strategy.accumulated_gradient=2
        algo.args.train.use_dynamic_bsz=False; algo.args.kd.use_ema_teacher=False
        optimizer=torch.optim.SGD(student.parameters(),lr=.01); updates=[]
        def backward(loss,*_):
            strategy.step=(strategy.step+1)%2
            (loss/2).backward()
        def step(*_):
            if strategy.step==0: optimizer.step(); optimizer.zero_grad(); updates.append(1)
        strategy.backward=backward; strategy.optimizer_step=step
        strategy.all_reduce=lambda x,**_:x
        actor=types.SimpleNamespace(student=student,kd_algorithm=algo,args=algo.args,strategy=strategy,
            optim=optimizer,scheduler=types.SimpleNamespace(get_last_lr=lambda:[.01]),_prepare_micro_batch=lambda b:dict(b))
        original=algo.on_optimizer_window_end
        def end():
            self.assertEqual(len(updates),1)
            return original()
        algo.on_optimizer_window_end=end
        batches=[helper.batch(),helper.batch(2)]
        result=methods['fit'](actor,batches)
        self.assertEqual(result['distill/h1b_window_responses'],3.)
        self.assertEqual(algo.window,1)
        previous=algo.baseline.prev.clone()
        with torch.no_grad(): methods['evaluate'](actor,batches)
        self.assertEqual(algo.window,1)
        torch.testing.assert_close(algo.baseline.prev,previous)

    def test_math_eval_override_never_changes_rollout_top_p(self):
        from kdflow.arguments.rollout_args import RolloutArguments
        tree=ast.parse((ROOT/'kdflow/evaluation/math_evaluator.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        assignments=[n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'eval_top_p','sampling_params'} for t in n.targets)]
        for override,expected in ((None,1.),(.95,.95)):
            actor=types.SimpleNamespace(args=types.SimpleNamespace(rollout=RolloutArguments(top_p=1.,eval_math_top_p=override)),
                max_tokens=8192,n_samples=32,generate_kwargs={'top_p':1.,'temperature':1.})
            scope={'self':actor}
            exec(compile(ast.Module(body=assignments,type_ignores=[]),'actual_math_sampling','exec'),scope)
            self.assertEqual(scope['sampling_params']['top_p'],expected)
            self.assertEqual(actor.generate_kwargs['top_p'],1.)
        with self.assertRaises(ValueError): RolloutArguments(eval_math_top_p=0)

    def test_fit_curve_and_analysis_handle_known_direction_and_zero_vectors(self):
        import numpy as np
        from tools.h1b.fit_cb import fit_curve,analyze_matrices
        sizes=[64,128,256,512,1024,2048,4096]
        fit=fit_curve(sizes,[.7/math.sqrt(1+300/b) for b in sizes])
        self.assertAlmostEqual(fit['c_infinity'],.7,places=2)
        x=np.tile(np.array([1.,2.,3.]),(8,1))
        report=analyze_matrices(x,x,x,x)
        self.assertTrue(all(abs(r['cosine']-1)<1e-9 for r in report['curve']))
        zero=analyze_matrices(x,x*0,x,x)
        self.assertTrue(all(r['cosine'] is None for r in zero['curve']))

    def test_replay_analysis_cli_on_complete_small_saved_sketches(self):
        from tools.h1b.fit_cb import main
        torch.manual_seed(3)
        keys=['strict','path','str','base_path','base_str','strict_0','strict_1']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            for i in range(8):
                files={}
                for key in keys:
                    file=f'{i}_{key}.pt'; torch.save(torch.randn(16),root/file); files[key]=file
                rows.append(dict(files=files,tokens=20+i))
            (root/'manifest.json').write_text(json.dumps(dict(records=rows,groups=8,group_size=64,bucket_classes=[])))
            main(['--replay',tmp,'--output',str(root/'curves.json'),'--bootstrap','2'])
            result=json.loads((root/'curves.json').read_text())
            self.assertEqual(set(result['results']),{'path','str','path_bl','str_bl'})
            self.assertIsNotNone(result['cos_mean_path_str'])

    def test_h1b_rollout_preserves_eos_policy_and_reports_raw_reencoding(self):
        from typing import Any,Dict,List,Optional
        tree=ast.parse((ROOT/'kdflow/trainer/data_processor/rollout_data_processor.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        scope=dict(torch=torch,Any=Any,Dict=Dict,List=List,Optional=Optional)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_h1b_processor','exec'),scope)
        class Tok:
            eos_token_id=0
            def __call__(self,text,return_tensors=None,**kwargs):
                ids=[{'p':6,'a':1,'b':2,'c':3}[c] for c in text]
                return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)} if return_tensors else {'input_ids':ids}
        obj=object.__new__(scope['RolloutDataProcessor'])
        obj.args=types.SimpleNamespace(kd=types.SimpleNamespace(kd_algorithm='h1b_span'))
        obj.student_processor=Tok(); obj.teacher_processors={}; obj.is_same_tokenizer=True
        sample=obj._build_rollout_sample('p','p',dict(text='abc',output_ids=[1,2,3]),None,append_eos=False)
        self.assertEqual(float(sample['stu_reencode_match']),1.)
        self.assertEqual(sample['stu_input_ids'].tolist(),[6,1,2,3])
        sample=obj._build_rollout_sample('p','p',dict(text='abc',output_ids=[2,3]),None,append_eos=True)
        self.assertEqual(float(sample['stu_reencode_match']),0.)
        self.assertEqual(sample['stu_input_ids'][-1],0)

    def test_KV_oracle_keeps_sibling_prefix_cache_immutable(self):
        from tools.h1b.common import HFPrefixOracle
        class Base(torch.nn.Module):
            def forward(self,ids,past_key_values=None,use_cache=False,**kwargs):
                cache={'ids':[]} if past_key_values is None else past_key_values
                cache['ids'].extend(ids[0].tolist())  # Deliberate in-place cache mutation.
                x=torch.tensor([sum(cache['ids']),len(cache['ids']),cache['ids'][-1],1.],dtype=torch.float32)
                return types.SimpleNamespace(last_hidden_state=x[None,None,:],past_key_values=cache if use_cache else None)
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.base_model=Base(); self.head=torch.nn.Linear(4,8)
            def get_output_embeddings(self): return self.head
        model=Model(); cached=HFPrefixOracle(model,[6,1,2],cache_depth=2); direct=HFPrefixOracle(model,[6,1,2],use_cache=False)
        for branch in ((),(1,),(1,2),(1,2,3),(2,),(2,3),(1,)):
            torch.testing.assert_close(cached(branch),direct(branch))
        self.assertEqual(cached.stack[0][2]['ids'],[6,1,2])

    def test_old_archive_prompt_recovery_requires_exact_unambiguous_match(self):
        from tools.h1b.enrich_archive import recover
        records=[dict(prompt='S:x',output='abc',finish_reason='stop')]
        rows=[dict(s='S:x',t='T:x')]
        result=recover(records,rows,lambda r:r['s'],lambda r:r['t'])
        self.assertEqual(result[0]['tea_prompt'],'T:x')
        self.assertEqual(result[0]['output'],'abc')
        self.assertNotIn('output_ids',result[0])
        with self.assertRaises(ValueError): recover(records,[],lambda r:r['s'],lambda r:r['t'])
        with self.assertRaises(ValueError): recover(records,rows+[dict(s='S:x',t='T:y')],lambda r:r['s'],lambda r:r['t'])

    def test_T11_sketch_cosine_norm_and_chunk_invariance(self):
        from tools.h1b.sketch import count_sketch
        torch.manual_seed(9); a=torch.randn(2000); b=a+.2*torch.randn_like(a)
        sa,na=count_sketch([('x',a)],chunk_size=17)
        sb,nb=count_sketch([('x',b)],chunk_size=99)
        sa2,_=count_sketch([('x',a)],chunk_size=33)
        torch.testing.assert_close(sa,sa2)
        self.assertLess(abs(float(sa.norm())/na-1),.01)
        self.assertLess(abs(float(sb.norm())/nb-1),.01)
        self.assertLess(abs(float(torch.nn.functional.cosine_similarity(sa,sb,dim=0)-torch.nn.functional.cosine_similarity(a,b,dim=0))),.01)

    def test_presets_share_full_workload_and_separate_eval_top_p(self):
        from tools.launch_single_b200 import parser,build_plan,H1B_ARMS
        original=build_plan(parser().parse_args(['--algorithm','strict-full','--dry-run']))
        allowed={'kd_algorithm','h1b_arm','h1b_span_lambda','h1b_log_sample_rate','h1b_dump_every','top_p','eval_math_top_p','save_path','ckpt_path'}
        for name,arm in H1B_ARMS.items():
            plan=build_plan(parser().parse_args(['--algorithm',name,'--dry-run']))
            self.assertEqual(plan['flags']['h1b_arm'],arm)
            self.assertEqual(plan['flags']['top_p'],1.)
            self.assertEqual(plan['flags']['eval_math_top_p'],.95)
            for key,value in original['flags'].items():
                if key not in allowed: self.assertEqual(plan['flags'][key],value)

    def test_source_manifest_frozen_strict(self):
        manifest=json.loads((ROOT/'docs/H1B_SOURCE.json').read_text())
        for row in manifest['files']+manifest['frozen_strict']:
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),row['sha256'])


if __name__=='__main__': unittest.main()
