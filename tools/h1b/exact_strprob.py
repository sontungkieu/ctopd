"""Score an archived rollout; bound off-trunk mass without generating responses."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from kdflow.h1b.vocab_index import VocabIndex
from kdflow.h1b.strprob import score_hidden_chunks
from kdflow.h1b.buckets import bucket_of
from kdflow.h1b.exact import exact_chunk_logprob
from .common import archive_rows,load_pair,tokenize_record,row_groups,forward_hidden,snapshot_identity,HFPrefixOracle,source_identity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True); p.add_argument('--student',required=True); p.add_argument('--teacher',required=True)
    p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=200)
    p.add_argument('--device',default='cuda'); p.add_argument('--tile-size',type=int,default=128)
    p.add_argument('--prune',type=float,default=1e-6); p.add_argument('--max-nodes',type=int,default=10000)
    p.add_argument('--trunk-only',action='store_true'); p.add_argument('--assume-shared-prompt',action='store_true')
    p.add_argument('--full-prefix',action='store_true',help='Disable KV reuse for a correctness comparison')
    p.add_argument('--cache-depth',type=int,default=16)
    args=p.parse_args(argv)
    if args.limit<1 or args.tile_size<1: p.error('limit and tile size must be positive')
    target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    models,toks=load_pair(args.student,args.teacher,args.device)
    indices=[VocabIndex(t) for t in toks]
    summaries=[[] for _ in range(24)]; dropped=0
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream,torch.no_grad():
        for ri,record in enumerate(archive_rows(args.archive,args.limit)):
            batch=tokenize_record(record,*toks,args.assume_shared_prompt)
            grouped=row_groups(batch,*toks)[0]
            if not grouped.valid: dropped+=1; continue
            groups=[g for g in grouped.groups if g.kind=='mis']
            scored=[]
            for model,tok,vi,prefix in zip(models,toks,indices,('stu','tea')):
                hidden=forward_hidden(model,batch,prefix)
                labels=batch[prefix+'_input_ids'].roll(-1,1)[batch[prefix+'_loss_mask'].bool()].to(hidden.device)
                head=model.get_output_embeddings()
                chunks=[((g.s_idx if prefix=='stu' else g.t_idx),g.content) for g in groups]
                # A plain HF head has no skip keyword, so detached scoring uses teacher mode.
                scores=score_hidden_chunks(head,hidden,labels,chunks,vi,args.tile_size,student=False)
                results=[]
                positions=batch[prefix+'_loss_mask'][0].nonzero().flatten().tolist()
                for g,lp in zip(groups,scores):
                    data=dict(path=float(lp.log_q_path),trunk=float(lp.log_p_trunk),
                              overhang=float((lp.log_overhang-lp.log_p_trunk).exp()))
                    if not args.trunk_only:
                        flat=(g.s_idx if prefix=='stu' else g.t_idx)[0]
                        base=batch[prefix+'_input_ids'][0,:positions[flat]+1].tolist()
                        oracle=HFPrefixOracle(model,base,args.cache_depth,not args.full_prefix)
                        bounds=exact_chunk_logprob(oracle,g.content,vi,args.prune,args.max_nodes)
                        data.update(bounds)
                        data['error_upper']=max(0.,bounds['log_probability_upper']-data['trunk'])
                        data['error_lower']=max(0.,bounds['log_probability_lower']-data['trunk'])
                    results.append(data)
                scored.append(results)
            for g,s,t in zip(groups,*scored):
                bucket=bucket_of(g.content)
                row=dict(response_index=ri,response_hash=hashlib.sha256(record['output'].encode()).hexdigest(),
                         bucket=bucket,b0=g.b0,b1=g.b1,chunk_hex=g.content.hex(),student=s,teacher=t)
                row['gap']=(s['path']-t['path'])-(s['trunk']-t['trunk'])
                summaries[bucket].append(row)
                # Omitted mass zero is -inf internally; JSON uses null for log(0).
                def finite(obj):
                    if isinstance(obj,dict): return {k:finite(v) for k,v in obj.items()}
                    return None if isinstance(obj,float) and not math.isfinite(obj) else obj
                stream.write(json.dumps(finite(row),allow_nan=False)+'\n')
    buckets=[]
    for b,rows in enumerate(summaries):
        scale=sum(abs(r['gap']) for r in rows)/max(len(rows),1)
        errors={m:sum(r[m].get('error_upper',0.) for r in rows)/max(len(rows),1) for m in ('student','teacher')}
        status=('not_measured' if args.trunk_only else 'insufficient' if not rows else
                'pass' if all(e<.1*scale for e in errors.values()) else 'fail')
        buckets.append(dict(bucket=b,count=len(rows),mean_abs_gap=scale,error_upper_mean=errors,status=status))
    summary=dict(source=source_identity(),archive_sha256=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
        student=snapshot_identity(args.student),teacher=snapshot_identity(args.teacher),dropped_responses=dropped,buckets=buckets,
        gate='not_measured' if args.trunk_only else 'pass' if all(b['status']=='pass' for b in buckets) else 'not_passed',
        search_backend='full-prefix reference' if args.full_prefix else 'depth-bounded DFS KV stack',
        qualification='Canonical-prefix conditional bounds; no real-model KV/B200 throughput qualification')
    Path(str(target)+'.summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__': main()
