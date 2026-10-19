"""Previous completed optimizer-window bucket means; no same-window leakage."""
import torch


class BucketBaseline:
    def __init__(self):
        self.prev=torch.zeros(2,24,dtype=torch.float64)
        self.sums=torch.zeros_like(self.prev)
        self.counts=torch.zeros_like(self.prev)

    def lookup(self,bucket,which):
        return float(self.prev[0 if which=='path' else 1,bucket])

    def accumulate(self,bucket,delta,which):
        i=0 if which=='path' else 1
        value=float(delta)
        if not torch.isfinite(torch.tensor(value)): raise FloatingPointError('Nonfinite bucket delta')
        self.sums[i,bucket]+=value; self.counts[i,bucket]+=1

    def end_window(self):
        # H1b currently rejects multi-rank training; no collective is hidden here.
        self.prev=torch.where(self.counts>0,self.sums/self.counts.clamp_min(1),self.prev)
        self.sums.zero_(); self.counts.zero_()

    def state_dict(self): return {'prev':self.prev.clone()}

    def load_state_dict(self,state):
        value=state['prev'].double().cpu()
        if value.shape!=(2,24) or not bool(torch.isfinite(value).all()): raise ValueError('Bad H1b baseline state')
        self.prev=value.clone(); self.sums.zero_(); self.counts.zero_()
