"""Streaming window moments and bounded, deterministic chunk sampling."""
import hashlib
import json
import math
from pathlib import Path


class Moments:
    def __init__(self): self.n=0; self.s={}; self.ss={}; self.cross=0.; self.floor_n=0; self.floor_sum=0.
    def add(self,row):
        self.n+=1
        for key in ('path','str','path_bl','str_bl','gap','slack_stu','slack_tea','overhang_stu','overhang_tea'):
            v=row[key]; self.s[key]=self.s.get(key,0.)+v; self.ss[key]=self.ss.get(key,0.)+v*v
        self.cross+=row['path']*row['str']
        if abs(row['str'])<.05: self.floor_n+=1; self.floor_sum+=row['gap']
    def summary(self):
        mean={k:v/max(self.n,1) for k,v in self.s.items()}
        var={k:max(0.,self.ss[k]/max(self.n,1)-mean[k]**2) for k in self.s}
        denom=math.sqrt(var.get('path',0.)*var.get('str',0.))
        corr=(self.cross/max(self.n,1)-mean.get('path',0.)*mean.get('str',0.))/denom if denom else 0.
        return dict(count=self.n,mean=mean,variance=var,correlation=corr,
                    correlation_defined=bool(denom),bias_floor=self.floor_sum/max(self.floor_n,1),bias_floor_count=self.floor_n)


class ChunkLogger:
    def __init__(self,save_path=None,sample_rate=.05,dump_steps=(0,20,60,100)):
        self.root=Path(save_path)/'h1b' if save_path else None
        self.sample_rate=sample_rate; self.dump_steps=set(dump_steps); self.step=0
        self.reset()
    def reset(self):
        self.total=Moments(); self.buckets=[Moments() for _ in range(24)]
        self.samples=[]; self.rows=0; self.dropped=0; self.byte_dropped=0
        self.stu=self.tea=self.strict_stu=self.strict_tea=self.mis_stu=0
        self.reencode_n=0; self.reencode_sum=0.
    def add(self,row):
        self.total.add(row); self.buckets[row['bucket']].add(row)
        key=f"{self.step}:{row['response_hash']}:{row['b0']}:{row['b1']}"
        u=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')/2**64
        if self.step in self.dump_steps or u<self.sample_rate:
            # JSON/parquet store absent overhang as null, never -Infinity.
            self.samples.append({k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in row.items()})
    def end_window(self,step):
        summary=self.total.summary()
        buckets=[b.summary() for b in self.buckets]
        mean=summary['mean']; var=summary['variance']
        metrics={'strict_cov_stu':self.strict_stu/max(self.stu,1),'strict_cov_tea':self.strict_tea/max(self.tea,1),
                 'mis_tok_frac':self.mis_stu/max(self.stu,1),'gap_mean':mean.get('gap',0.),
                 'slack_tea':mean.get('slack_tea',0.),'slack_stu':mean.get('slack_stu',0.),
                 'bias_floor':summary['bias_floor'],'bias_floor_count':summary['bias_floor_count'],
                 'corr_path_str':summary['correlation'],'corr_path_str_defined':float(summary['correlation_defined']),
                 'overhang_share_tea':mean.get('overhang_tea',0.),'overhang_share_stu':mean.get('overhang_stu',0.),
                 'reencode_match':self.reencode_sum/max(self.reencode_n,1),'reencode_count':self.reencode_n,
                 'dropped_byte_mismatch':self.byte_dropped,'window_responses':self.rows,'window_mis_groups':self.total.n}
        for k in ('path','str','path_bl','str_bl'): metrics['var_'+k]=var.get(k,0.)
        for k,b in enumerate(buckets):
            metrics[f'gap_mean_b{k}']=b['mean'].get('gap',0.)
            metrics[f'count_b{k}']=b['count']
        if self.root:
            self.root.mkdir(parents=True,exist_ok=True)
            target=self.root/f'step_{step}.json'
            with target.open('x') as f:
                json.dump(dict(step=step,metrics=metrics,total=summary,buckets=buckets),f,allow_nan=False,indent=2)
            import pyarrow as pa
            import pyarrow.parquet as pq
            table=pa.Table.from_pylist(self.samples) if self.samples else pa.table({'response_hash':pa.array([],type=pa.string())})
            pq.write_table(table,self.root/f'chunks_step_{step}.parquet')
        self.reset(); self.step=step+1
        return {'distill/h1b_'+k:v for k,v in metrics.items()}
