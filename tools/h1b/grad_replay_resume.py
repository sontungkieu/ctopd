"""Replay fixed archived responses through H1b; resume-capable copy; never generate new responses."""
import argparse
import hashlib
import json
from pathlib import Path
import types
import torch
from kdflow.algorithms.h1b_span import H1bSpanCTKD
from kdflow.arguments.distillation_args import DistillationArguments
from .common import archive_rows,load_pair,tokenize_record,forward_hidden,StudentWrapper,snapshot_identity,source_identity
from .sketch import count_sketch


def collate(batches):
    from torch.nn.utils.rnn import pad_sequence
    return {k:pad_sequence([b[k][0] for b in batches],batch_first=True) for k in batches[0]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','student','teacher','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--limit',type=int,default=4096); p.add_argument('--group-size',type=int,default=64)
    p.add_argument('--micro-batch',type=int,default=4); p.add_argument('--tile-size',type=int,default=128)
    p.add_argument('--dimension',type=int,default=2**22); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda'); p.add_argument('--buffer-device',default='cuda')
    p.add_argument('--exact-groups',type=int,default=2)
    p.add_argument('--buckets',default='',help='Optional byte class IDs 0=digits,3=letters; adds gradient buffers')
    p.add_argument('--assume-shared-prompt',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume into existing output dir; skip groups already done')
    p.add_argument('--start-group',type=int,default=0)
    p.add_argument('--end-group',type=int,default=None)
    args=p.parse_args(argv)
    if min(args.limit,args.group_size,args.micro_batch,args.tile_size,args.dimension)<1:
        p.error('Sizes must be positive')
    if args.limit%args.group_size or args.group_size%args.micro_batch:
        p.error('Require complete equal groups and microbatches')
    root=Path(args.output)
    loaded=None
    if args.resume:
        if not root.is_dir(): p.error('Resume requires existing output dir')
        mfile=root/'manifest.json'
        if not mfile.exists(): p.error('Resume requires manifest.json')
        loaded=json.loads(mfile.read_text())
    else:
        if root.exists(): raise FileExistsError(root)
    records=list(archive_rows(args.archive,args.limit))
    if len(records)!=args.limit: raise ValueError('Archive has fewer responses than requested')
    selected=[int(s) for s in args.buckets.split(',') if s.strip()]
    if any(not 0<=s<=5 for s in selected): raise ValueError('Byte class must be in 0..5')
    ngroups=args.limit//args.group_size
    end=args.end_group if args.end_group is not None else ngroups
    if not 0<=args.start_group<=end<=ngroups: p.error('Require 0 <= start <= end <= ngroups')
    if loaded is not None:
        if loaded.get('groups')!=ngroups or loaded.get('group_size')!=args.group_size:
            raise ValueError('Resume manifest groups/group_size mismatch')
        if loaded.get('dimension')!=args.dimension or loaded.get('seed')!=args.seed:
            raise ValueError('Resume manifest dimension/seed mismatch')
        if list(loaded.get('bucket_classes',[]))!=selected:
            raise ValueError('Resume manifest buckets mismatch')
    models,toks=load_pair(args.student,args.teacher,args.device)
    student,teacher=models
    teacher.requires_grad_(False)
    wrapper=StudentWrapper(student)
    strategy=types.SimpleNamespace(args=types.SimpleNamespace(
        kd=DistillationArguments(kd_algorithm='h1b_span',kd_ratio=1,kd_temperature=1,kd_loss_fn='rkl'),
        model=types.SimpleNamespace(ring_attn_size=1),
        train=types.SimpleNamespace(chunked_loss_size=args.tile_size,num_nodes=1,num_gpus_per_node=1,save_path=None),
        scenario='on_policy_kd'),ring_attn_group=None)
    algo=H1bSpanCTKD(strategy,wrapper,teacher.get_output_embeddings(),*toks)
    algo.replay_enabled=True
    algo.replay_control=False
    wrapper.eval()
    config=student.config
    if any(getattr(config,k,0) for k in ('attention_dropout','hidden_dropout','hidden_dropout_prob')):
        raise ValueError('Deterministic checkpointed replay requires zero model dropout')
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    student.base_model.train()
    encoded=[tokenize_record(r,*toks,args.assume_shared_prompt) for r in records]
    sums=torch.zeros(ngroups,2,24,dtype=torch.float64); counts=torch.zeros_like(sums)
    def batch_at(start,denominator):
        batch=collate(encoded[start:start+args.micro_batch])
        with torch.no_grad(): batch['teacher_hiddens']=forward_hidden(teacher,batch,'tea')
        for k,v in batch.items(): batch[k]=v.to(args.device)
        batch['avg_micro_batch_token_num']=denominator
        return batch
    with torch.no_grad():
        for i in range(0,args.limit,args.micro_batch):
            algo.training_step(batch_at(i,1.))
            gi=i//args.group_size
            for record in algo.replay_records:
                b=record['bucket']
                for kind,key in enumerate(('path','str')):
                    sums[gi,kind,b]+=float(record[key]); counts[gi,kind,b]+=1
    if ngroups<2: raise ValueError('Leave-group-out replay needs at least two groups')
    algo.replay_control=True
    root.mkdir(parents=True,exist_ok=bool(args.resume))
    named=list(wrapper.named_parameters()); params=tuple(v for _,v in named)
    offsets=[0]
    for _,param in named: offsets.append(offsets[-1]+param.numel())
    keys=['strict','path','str','base_path','base_str','strict_0','strict_1']
    for b in selected:
        keys.extend(f'class{b}_{term}' for term in ('path','str','base_path','base_str'))
    if loaded is not None:
        manifest=loaded
    else:
        manifest=dict(version=1,source=source_identity(),archive_sha256=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
                      student=snapshot_identity(args.student),teacher=snapshot_identity(args.teacher),group_size=args.group_size,groups=ngroups,
                      dimension=args.dimension,seed=args.seed,normalization='sum token-normalized microbatch gradients per group',
                      baseline='leave-group-out bucket means; outcome-dependent bucket caveat applies',
                      preconditioner='raw gradients only',bucket_classes=selected,records=[])
    done={r['group']:r for r in manifest.get('records',[])}
    for gi in range(args.start_group,end):
        if gi in done and all((root/f).exists() for f in done[gi].get('files',{}).values()):
            continue
        manifest['records']=[r for r in manifest['records'] if r['group']!=gi]
        other_s=sums.sum(0)-sums[gi]; other_n=counts.sum(0)-counts[gi]
        algo.baseline.prev=torch.where(other_n>0,other_s/other_n.clamp_min(1),torch.zeros_like(other_s))
        buffers={k:torch.zeros(offsets[-1],dtype=torch.float32,device=args.buffer_device) for k in keys}
        lo=gi*args.group_size
        tokens=sum(int(b['stu_loss_mask'].sum()) for b in encoded[lo:lo+args.group_size])
        for i in range(lo,lo+args.group_size,args.micro_batch):
            algo.training_step(batch_at(i,max(tokens,1)))
            terms=dict(algo.replay_terms)
            for b in selected:
                chunk_records=[r for r in algo.replay_records if r['bucket']//4==b]
                for term in ('path','str','base_path','base_str'):
                    zero=terms['strict']*0
                    key=term.removeprefix('base_')
                    terms[f'class{b}_{term}']=sum((2*algo.lam*(algo.baseline.lookup(r['bucket'],key)
                        if term.startswith('base_') else r[term])*r['log_q']/max(tokens,1) for r in chunk_records),zero)
            for ki,key in enumerate(keys):
                grads=torch.autograd.grad(terms[key],params,retain_graph=ki<len(keys)-1,allow_unused=True)
                for pi,grad in enumerate(grads):
                    if grad is not None: buffers[key][offsets[pi]:offsets[pi+1]].add_(grad.detach().flatten().to(args.buffer_device))
        files={}; norms={}
        for key,flat in buffers.items():
            tensors=((name,flat[offsets[j]:offsets[j+1]]) for j,(name,_) in enumerate(named))
            sketch,norm=count_sketch(tensors,args.dimension,args.seed)
            file=f'group_{gi:03d}_{key}.pt'
            torch.save(sketch,root/file); files[key]=file; norms[key]=norm
            if gi<args.exact_groups:
                torch.save(flat.cpu(),root/f'group_{gi:03d}_{key}_exact.pt')
        manifest['records'].append(dict(group=gi,tokens=tokens,files=files,exact_norms=norms,
                                       bucket_sums=sums[gi].tolist(),bucket_counts=counts[gi].tolist()))
        manifest['records'].sort(key=lambda r: r['group'])
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
        del buffers
    print(json.dumps(dict(groups=ngroups,output=str(root),qualification='raw fixed-text replay; GPU execution must be independently qualified')))


if __name__=='__main__': main()
