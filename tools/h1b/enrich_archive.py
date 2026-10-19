"""Recover teacher prompts in old rollout archives by exact dataset prompt lookup."""
import argparse
import hashlib
import json
from pathlib import Path
from .common import archive_rows,source_identity,snapshot_identity


def recover(records,dataset_rows,student_prompt,teacher_prompt):
    mapping={}
    for row in dataset_rows:
        s,t=student_prompt(row),teacher_prompt(row)
        if s in mapping and mapping[s]!=t: raise ValueError('Ambiguous teacher prompt for identical student prompt')
        mapping[s]=t
    output=[]
    for record in records:
        if record['prompt'] not in mapping: raise ValueError('Archived student prompt has no exact dataset/template match')
        teacher=mapping[record['prompt']]
        if 'tea_prompt' in record and record['tea_prompt']!=teacher:
            raise ValueError('Existing archived teacher prompt differs from recovered prompt')
        output.append(dict(record,tea_prompt=teacher))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','dataset','student','teacher','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--input-key',default='prompt'); p.add_argument('--teacher-input-key',default='prompt')
    args=p.parse_args(argv); target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from kdflow.datasets.prompts_dataset import PromptDataset
    toks=[AutoTokenizer.from_pretrained(p,local_files_only=True) for p in (args.student,args.teacher)]
    builder=object.__new__(PromptDataset)
    builder.apply_chat_template=True; builder.image_key=None; builder.enable_thinking=False
    builder.input_template=None
    rows=pq.read_table(args.dataset,columns=list(dict.fromkeys((args.input_key,args.teacher_input_key)))).to_pylist()
    records=recover(list(archive_rows(args.archive)),rows,
        lambda r:builder._build_prompt(r,toks[0],args.input_key),
        lambda r:builder._build_prompt(r,toks[1],args.teacher_input_key))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:
        for record in records: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    receipt=dict(source=source_identity(),records=len(records),
        archive_sha256=hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        student=snapshot_identity(args.student),teacher=snapshot_identity(args.teacher),
        output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        contract='Exact student prompt lookup, production PromptDataset template method; enable_thinking=False; no response regeneration or sampled-ID reconstruction')
    Path(str(target)+'.receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
