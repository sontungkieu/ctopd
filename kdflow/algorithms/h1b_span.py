"""H1b Strict + detached path/string span coefficients, opt-in only."""
import hashlib
from dataclasses import replace
import math
import torch
from kdflow.algorithms import register_algorithm
from kdflow.algorithms.byte_align_ctkd import ByteAlignCrossTokenizerKD
from kdflow.loss.chunked_loss import chunked_loss
from kdflow.h1b.groups import align_byte_span_groups_batch
from kdflow.h1b.vocab_index import VocabIndex
from kdflow.h1b.strprob import score_hidden_chunks
from kdflow.h1b.baseline import BucketBaseline
from kdflow.h1b.buckets import bucket_of
from kdflow.h1b.chunklog import ChunkLogger


@register_algorithm('h1b_span')
class H1bSpanCTKD(ByteAlignCrossTokenizerKD):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        kd=self.args.kd
        self.arm=kd.h1b_arm; self.lam=kd.h1b_span_lambda
        if self.arm not in {'a','b','b_bl','c_str','c'} or not math.isfinite(self.lam) or self.lam<0:
            raise ValueError('Invalid H1b arm or span lambda')
        if kd.kd_ratio!=1 or kd.kd_temperature!=1 or kd.kd_loss_fn!='rkl':
            raise ValueError('H1b requires kd_ratio=1, temperature=1 and rkl')
        if (self.args.train.num_nodes*self.args.train.num_gpus_per_node!=1
                or self.args.model.ring_attn_size!=1):
            raise ValueError('H1b supports one student rank and ring size 1')
        config=getattr(self.student.model,'config',None)
        config=getattr(config,'text_config',None) or config
        if getattr(config,'final_logit_softcapping',None) or getattr(self.teacher_lm_head,'_mp_opd_softcap',None):
            raise ValueError('H1b softcapped heads are not qualified')
        self.svi=VocabIndex(self.student_tokenizer,self.student_byte_codec)
        self.tvi=VocabIndex(self.teacher_tokenizer,self.teacher_byte_codec)
        self.baseline=BucketBaseline()
        self.window=0
        self.chunk_logger=ChunkLogger(getattr(self.args.train,'save_path',None),
            kd.h1b_log_sample_rate,tuple(int(s) for s in kd.h1b_dump_every.split(',') if s.strip()))

    def validate_optimizer_window(self,batches):
        if self.strategy.step!=0 or len(batches)!=self.strategy.accumulated_gradient:
            raise ValueError('H1b requires one complete optimizer window per fit')

    def state_dict(self):
        return dict(version=1,arm=self.arm,span_lambda=self.lam,window=self.window,baseline=self.baseline.state_dict())

    def load_state_dict(self,state):
        if state['version']!=1 or state['arm']!=self.arm or state['span_lambda']!=self.lam:
            raise ValueError('Changed H1b state contract')
        self.baseline.load_state_dict(state['baseline']); self.window=int(state['window'])
        self.chunk_logger.reset(); self.chunk_logger.step=self.window

    def on_optimizer_window_end(self):
        if self.strategy.step!=0: raise ValueError('Cannot close partial H1b optimizer window')
        metrics=self.chunk_logger.end_window(self.window)
        self.baseline.end_window(); self.window+=1
        return metrics

    def training_step(self,batch):
        replay=getattr(self,'replay_enabled',False)
        # Offline replay intentionally retains graphs; normal training must not
        # keep the previous microbatch graph alive during the next forward.
        self.replay_terms={}; self.replay_records=[]
        sm,tm=batch['stu_loss_mask'].bool(),batch['tea_loss_mask'].bool()
        sy,ty=batch['stu_input_ids'].roll(-1,1),batch['tea_input_ids'].roll(-1,1)
        st=[self.student_tokenizer.convert_ids_to_tokens(sy[i][sm[i]].cpu().tolist()) for i in range(len(sm))]
        tt=[self.teacher_tokenizer.convert_ids_to_tokens(ty[i][tm[i]].cpu().tolist()) for i in range(len(tm))]
        rows=align_byte_span_groups_batch(tt,st,self.teacher_tokenizer.eos_token,self.student_tokenizer.eos_token,
            self.teacher_byte_codec.piece_to_bytes,self.student_byte_codec.piece_to_bytes)
        for i,row in enumerate(rows):
            sids=sy[i][sm[i]].cpu().tolist(); tids=ty[i][tm[i]].cpu().tolist()
            if (any(t in self.svi.special and t!=self.student_tokenizer.eos_token_id for t in sids)
                    or any(t in self.tvi.special and t!=self.teacher_tokenizer.eos_token_id for t in tids)):
                rows[i]=replace(row,groups=(),valid=False,reason='unsupported_special')
        sh=self.student(batch['stu_input_ids'],attention_mask=batch['stu_attn_mask'],allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,**(batch.get('stu_multi_modal_inputs') or {}))['hidden_states'][-1][sm]
        th=batch['teacher_hiddens']
        if isinstance(th,list): th=torch.cat(th,0)
        if len(th)!=int(tm.sum()): raise ValueError('H1b teacher hidden/mask mismatch')
        th=th.detach().to(self.teacher_lm_head.weight)
        sl,tl=sy[sm],ty[tm]
        groups=[g for row in rows for g in row.groups]
        strict=[g for g in groups if g.kind=='strict']
        mis=[g for g in groups if g.kind=='mis']
        si=[g.s_idx[0] for g in strict]; ti=[g.t_idx[0] for g in strict]
        size=self.args.train.chunked_loss_size or max(len(sh),1)
        denom=float(batch['avg_micro_batch_token_num'])
        if not math.isfinite(denom) or denom<=0: raise ValueError('Positive finite common normalizer required')
        zero=sh.sum()*0.
        strict_sum=zero; metric_sums={}
        if si:
            ash,ath=sh[si],th[ti]
            def slog(h,skip=False): return self.student.model.lm_head(h,skip=skip)[:,self.student_overlap_token_ids]
            def tlog(a,b):
                with torch.no_grad():
                    return self.teacher_lm_head(ath[a:b])[:,self.teacher_overlap_token_ids]
            strict_sum,metric_sums=chunked_loss(ash,self.student.model.lm_head,self.loss_fn,
                student_logits_fn=slog,teacher_logits_fn=tlog,chunk_size=size,reduction='sum',
                metric_fns=self.metric_fns,return_metrics=True)
        sprob=score_hidden_chunks(self.student.model.lm_head,sh,sl,[(g.s_idx,g.content) for g in mis],self.svi,size,student=True)
        with torch.no_grad():
            tprob=score_hidden_chunks(self.teacher_lm_head,th,tl,[(g.t_idx,g.content) for g in mis],self.tvi,size,student=False)
        terms={k:zero for k in ('path','str','base_path','base_str')}
        training=self.student.training and torch.is_grad_enabled()
        log=self.chunk_logger
        if training:
            log.rows+=len(rows); log.dropped+=sum(not r.valid for r in rows)
            log.byte_dropped+=sum(r.reason=='byte_mismatch' for r in rows)
            log.stu+=sum(r.student_tokens for r in rows); log.tea+=sum(r.teacher_tokens for r in rows)
            log.strict_stu+=sum(len(g.s_idx) for g in strict if not g.eos)
            log.strict_tea+=sum(len(g.t_idx) for g in strict if not g.eos)
            log.mis_stu+=sum(len(g.s_idx) for g in mis)
            if 'stu_reencode_match' in batch:
                values=torch.as_tensor(batch['stu_reencode_match']).detach()
                log.reencode_sum+=float(values.sum()); log.reencode_n+=values.numel()
        # Response identity excludes prompt text and does not retain raw content.
        response_hashes={i:hashlib.sha256(bytes(str((st[i],tt[i])),'utf8')).hexdigest() for i in range(len(rows))}
        owner={id(g):i for i,r in enumerate(rows) for g in r.groups}
        for g,s,t in zip(mis,sprob,tprob):
            dp=(s.log_q_path-t.log_q_path).detach(); ds=(s.log_p_trunk-t.log_p_trunk).detach()
            bucket=bucket_of(g.content)
            bp=self.baseline.lookup(bucket,'path'); bs=self.baseline.lookup(bucket,'str')
            if replay:
                self.replay_records.append(dict(bucket=bucket,path=dp,str=ds,log_q=s.log_q_path))
            for k,v in (('path',dp),('str',ds),('base_path',bp),('base_str',bs)):
                terms[k]=terms[k]+2*self.lam*v*s.log_q_path
            if training:
                p,v=float(dp),float(ds)
                coefficient=0. if self.arm=='a' else 2*self.lam*(p-bp if self.arm=='b_bl' else v-bs if self.arm=='c' else p if self.arm=='b' else v)
                log.add(dict(response_hash=response_hashes[owner[id(g)]],step=self.window,bucket=bucket,
                    b0=g.b0,b1=g.b1,chunk_hex=g.content.hex(),s_tokens=len(g.s_idx),t_tokens=len(g.t_idx),
                    path=p,str=v,path_bl=p-bp,str_bl=v-bs,gap=p-v,baseline_path=bp,baseline_str=bs,coefficient=coefficient,
                    stu_log_q_path=float(s.log_q_path.detach()),tea_log_q_path=float(t.log_q_path),
                    stu_log_p_trunk=float(s.log_p_trunk),tea_log_p_trunk=float(t.log_p_trunk),
                    stu_log_overhang=float(s.log_overhang),tea_log_overhang=float(t.log_overhang),
                    slack_stu=float(s.log_p_trunk-s.log_q_path.detach()),slack_tea=float(t.log_p_trunk-t.log_q_path),
                    overhang_stu=float((s.log_overhang-s.log_p_trunk).exp()),overhang_tea=float((t.log_overhang-t.log_p_trunk).exp())))
                self.baseline.accumulate(bucket,p,'path'); self.baseline.accumulate(bucket,v,'str')
        span=zero if self.arm=='a' else (terms['path'] if self.arm=='b' else terms['path']-terms['base_path'] if self.arm=='b_bl'
                                       else terms['str'] if self.arm=='c_str' else terms['str']-terms['base_str'])
        loss=(strict_sum+span)/denom
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite H1b loss')
        if replay:
            self.replay_terms={k:v/denom for k,v in terms.items()}
            self.replay_terms['strict']=strict_sum/denom
        if getattr(self,'replay_control',False):
            for parity in (0,1):
                x,y=sh[si[parity::2]],th[ti[parity::2]]
                if len(x):
                    def replay_teacher(a,b,y=y):
                        with torch.no_grad():
                            return self.teacher_lm_head(y[a:b])[:,self.teacher_overlap_token_ids]
                    val=chunked_loss(x,self.student.model.lm_head,self.loss_fn,
                        student_logits_fn=slog,
                        teacher_logits_fn=replay_teacher,
                        chunk_size=size,reduction='sum')
                else: val=zero
                self.replay_terms['strict_'+str(parity)]=val/denom
        info={'train/loss':loss,'train/kd_loss':loss,'train/h1b_span_loss':span/denom,
              'distill/alignment_ratio':torch.tensor(len(si)/max(len(sl),1)),
              'distill/h1b_mis_groups':loss.detach().new_tensor(float(len(mis))),
              'distill/h1b_dropped_responses':loss.detach().new_tensor(float(sum(not r.valid for r in rows)))}
        info.update({k:v/denom for k,v in metric_sums.items()})
        return info
