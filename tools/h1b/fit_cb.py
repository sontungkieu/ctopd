"""c(B), split-half, SNR and group bootstrap from recorded CountSketch groups."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den else None


def fit_curve(batch_sizes,values):
    valid=[(b,c) for b,c in zip(batch_sizes,values) if c is not None]
    if len(valid)<2: return dict(c_infinity=None,b_noise=None,identified=False)
    x,y=np.array(valid).T
    grid=np.r_[0.,np.geomspace(1e-3,1e9,2048)]
    basis=1/np.sqrt(1+grid[:,None]/x)
    c=np.clip((basis*y).sum(1)/(basis*basis).sum(1),-1,1)
    error=((basis*c[:,None]-y)**2).sum(1)
    k=int(error.argmin())
    return dict(c_infinity=float(c[k]),b_noise=float(grid[k]),identified=k not in (0,len(grid)-1),squared_error=float(error[k]))


def analyze_matrices(strict,span,control0,control1,group_size=64,seed=42,repeats=20,weights=None):
    """Small reference API, also used by tests. Undefined zero-vector cosines are null."""
    rng=np.random.default_rng(seed); n=len(strict)
    weights=np.ones(n) if weights is None else np.asarray(weights,dtype=float)
    if np.any(weights<=0) or len(weights)!=n: raise ValueError('Positive group token weights required')
    def pooled(matrix,ids): return np.average(matrix[ids],axis=0,weights=weights[ids])
    sizes=[b for b in (64,128,256,512,1024,2048,4096) if b>=group_size and b%group_size==0 and b//group_size<=n]
    rows=[]
    for b in sizes:
        m=b//group_size; cs=[]; ctrl=[]; halves=[]; ratios=[]; samples=[]
        for _ in range(repeats):
            ids=rng.choice(n,m,replace=False); s=pooled(strict,ids); v=pooled(span,ids)
            cs.append(cosine(v,s)); ctrl.append(cosine(pooled(control0,ids),pooled(control1,ids)))
            ratios.append(float(np.linalg.norm(v)/np.linalg.norm(s)) if np.linalg.norm(s) else None)
            if m>=2:
                h=m//2; halves.append(cosine(pooled(span,ids[:h]),pooled(span,ids[h:])))
            samples.append(v)
        sample=np.stack(samples[:16]); mu=sample.mean(0); noise=float(np.square(sample-mu).sum(1).mean())
        if m==n: noise=0.  # Resampling the entire fixed pool cannot identify sampling noise.
        def avg(v):
            v=[x for x in v if x is not None]
            return float(np.mean(v)) if v else None
        rows.append(dict(batch=b,cosine=avg(cs),control=avg(ctrl),split_half=avg(halves),norm_ratio=avg(ratios),
                         snr=float(np.dot(mu,mu)/noise) if noise else None,
                         snr_defined=bool(noise),split_half_responses=b//2 if m>=2 else None,
                         finite_population=True))
    mean=np.average(span,axis=0,weights=weights); signal=float(np.dot(mean,mean)); noise=float(np.square(span-mean).sum(1).mean())
    return dict(curve=rows,fit=fit_curve(sizes,[r['cosine'] for r in rows]),
                group_noise_scale=group_size*noise/signal if signal else None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',required=True); p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=200); p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(argv)
    root=Path(args.replay); meta=json.loads((root/'manifest.json').read_text())
    if len(meta['records'])!=meta['groups']: raise ValueError('Incomplete replay manifest')
    target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    def load(key):
        return np.stack([torch.load(root/r['files'][key],map_location='cpu',weights_only=True).numpy() for r in meta['records']])
    # Gram factorization reduces all subsequent resampling to <= 7*N dimensions;
    # raw sketches are loaded one term at a time, not retained for bootstraps.
    keys=list(meta['records'][0]['files'])
    grams={}
    for key in keys:
        a=load(key)
        for other in ('strict','strict_0','strict_1','path','str'):
            b=a if key==other else load(other)
            grams[(key,other)]=a@b.T
        grams[(key,key)]=a@a.T
    def factor(names):
        block=np.block([[grams.get((a,b),grams.get((b,a),None).T if (b,a) in grams else None)
                          for b in names] for a in names])
        val,vec=np.linalg.eigh((block+block.T)/2)
        return np.split(vec*np.sqrt(np.maximum(val,0))[None,:],len(names))
    variants={'path':('path',None),'str':('str',None),'path_bl':('path','base_path'),'str_bl':('str','base_str')}
    for cls in meta.get('bucket_classes',[]):
        variants.update({f'class{cls}_{k}':(f'class{cls}_{v[0]}',f'class{cls}_{v[1]}' if v[1] else None) for k,v in list(variants.items()) if not k.startswith('class')})
    results={}; rng=np.random.default_rng(args.seed)
    # Cross Grams involving baseline buffers are required for subtraction; compute
    # only the pairs needed for each analysis, retaining small matrices.
    for variant,(key,base) in variants.items():
        names=['strict',key,'strict_0','strict_1']+([base] if base else [])
        for a in names:
            for b in names:
                if (a,b) not in grams and (b,a) not in grams: grams[(a,b)]=load(a)@load(b).T
        mat=factor(names); strict,span,c0,c1=mat[:4]
        if base: span=span-mat[4]
        weights=np.array([r['tokens'] for r in meta['records']])
        result=analyze_matrices(strict,span,c0,c1,meta['group_size'],args.seed,weights=weights)
        bootstrap=[]
        for _ in range(args.bootstrap):
            ids=rng.integers(0,len(strict),len(strict))
            bootstrap.append(analyze_matrices(strict[ids],span[ids],c0[ids],c1[ids],meta['group_size'],int(rng.integers(2**31)),weights=weights[ids])['fit'])
        result['bootstrap_intervals']={k:(np.quantile([b[k] for b in bootstrap if b[k] is not None],[.025,.975]).tolist()
                                         if any(b[k] is not None for b in bootstrap) else None) for k in ('c_infinity','b_noise')}
        results[variant]=result
    names=['path','str']; a,b=factor(names)
    output=dict(results=results,cos_mean_path_str=cosine(np.average(a,axis=0,weights=weights),np.average(b,axis=0,weights=weights)),
        qualification='CountSketch estimates; group-bootstrap intervals; finite-pool SNR; undefined diagnostics are null',
        strict_control='disjoint alternating strict positions within each microbatch',
        missing='Adam-preconditioned analysis, fresh on-policy generation, end-task statistics are separate operations')
    target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__': main()
