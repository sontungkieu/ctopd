"""Separate five-step diagnostic (T10); never a reduced-workload reproduction."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import subprocess
import time
from tools.launch_single_b200 import parser,build_plan,validate_files,run_environment,H1B_ARMS


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm',choices=H1B_ARMS,required=True); p.add_argument('--runtime',required=True)
    p.add_argument('--root',default='/workspace/storage-shared/nlp/tungks/ctopd_all')
    p.add_argument('--run-id',required=True); p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if not args.run_id.startswith('canary-h1b-'): p.error('Use a distinct canary-h1b-* run ID')
    plan=build_plan(parser().parse_args(['--algorithm',args.algorithm,'--runtime',args.runtime,'--root',args.root,'--run-id',args.run_id]))
    plan['flags']['max_steps']=5
    plan['qualification']='T10 five-step diagnostic only; full 512/8192 workload per step; not reproduction'
    command=['bash',str(Path(args.runtime)/'python-b200.sh'),'-u','-m','kdflow.cli.train_kd_on_policy']
    for k,v in plan['flags'].items(): command.extend(['--'+k,str(v)])
    plan['command']=command
    if not args.execute:
        print(json.dumps(plan,indent=2)); return
    source=Path(plan['source'])
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Commit source before diagnostic execution')
    validate_files(plan)
    env=run_environment()
    probe="import torch; assert torch.cuda.device_count()==1; assert 'B200' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"
    subprocess.run(['bash',str(Path(args.runtime)/'python-b200.sh'),'-c',probe],env=env,check=True)
    root=Path(plan['output']); root.mkdir(parents=True,exist_ok=False)
    plan['source_commit']=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    (root/'canary-plan.json').write_text(json.dumps(plan,indent=2))
    start=time.monotonic()
    with (root/'canary.log').open('x') as log:
        with subprocess.Popen(command,env=env,cwd=source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            code=process.wait()
    receipt=dict(source_commit=plan['source_commit'],algorithm=args.algorithm,exit_code=code,
                 elapsed_seconds=time.monotonic()-start,passed=False,steps=[],qualification='Company-node canary only; no metric-match claim')
    import pyarrow.parquet as pq
    if code==0:
        values=re.findall(r'train/(?:loss|kd_loss|h1b_span_loss|grad_norm):\s*([^\s,]+)',(root/'canary.log').read_text())
        if not values or not all(math.isfinite(float(v)) for v in values):
            raise ValueError('Missing/nonfinite train loss/gradient metrics')
        for step in range(5):
            path=root/'checkpoints'/'h1b'/f'step_{step}.json'
            summary=json.loads(path.read_text()); metrics=summary['metrics']
            required={'gap_mean','slack_stu','slack_tea','bias_floor','var_path','var_str','var_path_bl','var_str_bl',
                      'corr_path_str','overhang_share_stu','overhang_share_tea','reencode_match','dropped_byte_mismatch'}
            if not required<=metrics.keys() or not all(math.isfinite(float(v)) for v in metrics.values()):
                raise ValueError('Missing/nonfinite canary metrics')
            file=path.parent/f'chunks_step_{step}.parquet'
            rows=pq.read_table(file).num_rows
            receipt['steps'].append(dict(step=step,sampled_chunks=rows,dump_bytes=file.stat().st_size))
        receipt['passed']=True
    (root/'canary-receipt.json').write_text(json.dumps(receipt,indent=2))
    if not receipt['passed']: raise SystemExit('Canary failed; preserve log and receipt')
    print(json.dumps(receipt))


if __name__=='__main__': main()
