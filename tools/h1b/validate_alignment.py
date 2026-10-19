"""Real-tokenizer T1/T3/T9 gates; requires local tokenizer snapshots and archived text."""
import argparse
import hashlib
import json
from pathlib import Path
from kdflow.token_alignment import get_tokenizer_byte_codec,align_byte_span_batch_1to1
from kdflow.algorithms._mp_opd_port._mp_opd_atoms import SimCTAtomizer
from .common import archive_rows,tokenize_record,row_groups


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('student','teacher','archive','texts','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--assume-shared-prompt',action='store_true')
    args=p.parse_args(argv)
    target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    from transformers import AutoTokenizer
    toks=[AutoTokenizer.from_pretrained(s,local_files_only=True) for s in (args.student,args.teacher)]
    codecs=[get_tokenizer_byte_codec(t) for t in toks]
    texts=Path(args.texts).read_text().splitlines()
    if len(texts)<10000: raise ValueError('T1 requires at least 10,000 texts')
    # User-supplied corpus must include challenging scripts, not just 10k copies
    # of one ASCII fixture. This list is supplemental and reported separately.
    probes=['Tiếng Việt: một phép thử','中文 日本語 한국어','🙂🚀','\\frac{120}{3} = 40','def f(x):\n  return x+1','a\t b\n']
    failures=[]
    for model,tok,codec in zip(('student','teacher'),toks,codecs):
        for i,text in enumerate(texts+probes):
            ids=tok(text,add_special_tokens=False)['input_ids']
            raw=b''.join(codec.piece_to_bytes(p) for p in tok.convert_ids_to_tokens(ids))
            if raw!=text.encode(): failures.append(dict(test='T1',model=model,index=i))
    records=list(archive_rows(args.archive,500))
    if len(records)!=500: raise ValueError('T3/T9 require 500 real responses')
    atomizer=SimCTAtomizer(toks[0],toks[1])
    common=0; dropped=0
    for ri,record in enumerate(records):
        batch=tokenize_record(record,*toks,args.assume_shared_prompt)
        grouped=row_groups(batch,*toks)[0]
        sid=batch['stu_input_ids'].roll(-1,1)[batch['stu_loss_mask'].bool()].tolist()
        tid=batch['tea_input_ids'].roll(-1,1)[batch['tea_loss_mask'].bool()].tolist()
        st,tt=toks[0].convert_ids_to_tokens(sid),toks[1].convert_ids_to_tokens(tid)
        strict=align_byte_span_batch_1to1([tt],[st],toks[1].eos_token,toks[0].eos_token,
            codecs[1].piece_to_bytes,codecs[0].piece_to_bytes)
        actual=([g.t_idx[0] for g in grouped.groups if g.kind=='strict'],[g.s_idx[0] for g in grouped.groups if g.kind=='strict'])
        if actual!=strict:
            failures.append(dict(test='T3',response=ri,reason=grouped.reason or 'strict_set_difference',author_indices=strict,h1b_indices=actual))
        atomized=atomizer.atomize(sid,tid,sample_id=str(ri))
        if not grouped.valid or not atomized.valid: dropped+=1; continue
        common+=1
        # EOS is a Strict-only synchronization event; compare content spans.
        pairs=[(g.s_idx[0],g.s_idx[-1]+1,g.t_idx[0],g.t_idx[-1]+1,g.kind) for g in grouped.groups if not g.eos]
        reference=[(a.student_start,a.student_end,a.teacher_start,a.teacher_end,
                    'strict' if a.student_end-a.student_start==a.teacher_end-a.teacher_start==1 else 'mis') for a in atomized.atoms]
        if pairs!=reference: failures.append(dict(test='T9',response=ri,h1b=pairs,mp_opd=reference))
    out=dict(text_count=len(texts),supplemental_probes=len(probes),responses=500,common_accepted=common,
             excluded_from_common=dropped,disagreements=failures,passed=not failures,
             archive_sha256=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
             qualification='Real-input alignment gate only; no GPU training')
    target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(out,indent=2))
    if failures: raise SystemExit('H1b alignment gate has disagreements; see report')


if __name__=='__main__': main()
