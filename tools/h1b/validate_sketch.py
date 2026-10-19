"""Validate saved group 0/1 sketches against memory-mapped exact gradients (T11)."""
import argparse
import json
from pathlib import Path
import math
import torch


def compare(root,manifest,chunk_size=2**20):
    results={}; passed=True
    for key in manifest['records'][0]['files']:
        exact=[torch.load(root/f'group_{i:03d}_{key}_exact.pt',map_location='cpu',weights_only=True,mmap=True) for i in (0,1)]
        sketch=[torch.load(root/manifest['records'][i]['files'][key],map_location='cpu',weights_only=True) for i in (0,1)]
        aa=bb=ab=0.
        if exact[0].shape!=exact[1].shape: raise ValueError('Exact gradient shapes differ')
        for start in range(0,exact[0].numel(),chunk_size):
            a,b=(x[start:start+chunk_size].double() for x in exact)
            aa+=float(a.square().sum()); bb+=float(b.square().sum()); ab+=float((a*b).sum())
        norms=[math.sqrt(aa),math.sqrt(bb)]
        sn=[float(x.double().norm()) for x in sketch]
        errors=[abs(s/n-1) if n else 0. if s==0 else math.inf for s,n in zip(sn,norms)]
        ec=ab/math.sqrt(aa*bb) if aa*bb else None
        sc=float(torch.nn.functional.cosine_similarity(sketch[0].double(),sketch[1].double(),dim=0)) if sn[0]*sn[1] else None
        ce=abs(ec-sc) if ec is not None and sc is not None else None
        ok=all(e<.01 for e in errors) and (ce is None or ce<.01)
        passed=passed and ok
        results[key]=dict(norm_relative_errors=errors,exact_cosine=ec,sketch_cosine=sc,cosine_abs_error=ce,passed=ok)
    return dict(passed=passed,terms=results,zero_vectors='cosine undefined; reported null')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--replay',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(argv); root=Path(args.replay); target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    meta=json.loads((root/'manifest.json').read_text())
    result=compare(root,meta); target.write_text(json.dumps(result,indent=2,allow_nan=False))
    if not result['passed']: raise SystemExit('Sketch accuracy gate failed')


if __name__=='__main__': main()
