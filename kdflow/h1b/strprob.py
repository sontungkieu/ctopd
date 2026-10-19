"""Canonical path and one-exit trunk prefix probability (detached diagnostics)."""
from dataclasses import dataclass
import torch


@dataclass
class ChunkLP:
    log_q_path: torch.Tensor
    log_p_trunk: torch.Tensor
    log_overhang: torch.Tensor


def chunk_logprobs(logp_rows, path_ids, chunk_bytes, vi):
    if not chunk_bytes or not len(path_ids) or len(path_ids) != len(logp_rows):
        raise ValueError('A chunk needs nonempty bytes and one prediction row per path token')
    if b''.join(vi.bytes(i) for i in path_ids) != chunk_bytes:
        raise ValueError('Canonical path does not spell chunk bytes')
    terms, over, used = [], [], 0
    acc = logp_rows.new_zeros(())
    for row, token in zip(logp_rows, path_ids):
        rem = chunk_bytes[used:]
        ids = vi.cover_ids(rem, row.device)
        terms.append(acc + torch.logsumexp(row[ids], 0))
        oi = [i for i in vi.cover(rem) if len(vi.bytes(i)) > len(rem)]
        if oi:
            over.append(acc + torch.logsumexp(row[oi], 0))
        acc = acc + row[int(token)]
        used += len(vi.bytes(token))
    trunk = torch.logsumexp(torch.stack(terms), 0)
    if not bool(torch.isfinite(acc)) or not bool(torch.isfinite(trunk)):
        raise FloatingPointError('Nonfinite H1b path/trunk probability')
    if bool(trunk < acc - 2e-5): raise ArithmeticError('H1b trunk lower than canonical path')
    return ChunkLP(acc, trunk, torch.logsumexp(torch.stack(over),0) if over else acc.new_tensor(-torch.inf))


def score_hidden_chunks(head, hidden, labels, chunks, vi, tile_size, *, student):
    """Keep selected path gradients; reduce trunk tiles without retaining logits.

    chunks is a list of (flat indices, bytes). Each checkpoint contains only a
    tile, its selected logps and detached covering-event log-masses.
    """
    from torch.utils.checkpoint import checkpoint
    if tile_size < 1: raise ValueError('Positive head tile size required')
    paths, covers, overs = [], [], []
    label_ids=labels.detach().cpu().tolist()
    flat = []
    for indices,content in chunks:
        used = 0
        ids = [label_ids[i] for i in indices]
        if b''.join(vi.bytes(i) for i in ids) != content:
            raise ValueError('H1b head path/byte mismatch')
        for pos,tid in zip(indices,ids):
            rem=content[used:]
            cov=vi.cover(rem)
            ov=tuple(i for i in cov if len(vi.bytes(i))>len(rem))
            flat.append((pos,tid,cov,ov))
            used += len(vi.bytes(tid))
    for start in range(0,len(flat),tile_size):
        records=flat[start:start+tile_size]
        indices=torch.tensor([r[0] for r in records],device=hidden.device)
        def tile(h, records=records):
            logits=head(h,skip=False) if student else head(h)
            lp=logits.float().log_softmax(-1)
            q=lp[torch.arange(len(records),device=lp.device),[r[1] for r in records]]
            with torch.no_grad():
                c=torch.stack([torch.logsumexp(lp[j,list(r[2])],0) for j,r in enumerate(records)])
                o=torch.stack([torch.logsumexp(lp[j,list(r[3])],0) if r[3] else lp.new_tensor(-torch.inf)
                               for j,r in enumerate(records)])
            return q,c,o
        with torch.set_grad_enabled(student and torch.is_grad_enabled()):
            q,c,o=(checkpoint(tile,hidden[indices],use_reentrant=False)
                   if student and torch.is_grad_enabled() else tile(hidden[indices]))
        paths.extend(q.unbind()); covers.extend(c.unbind()); overs.extend(o.unbind())
    result=[]
    offset=0
    for indices,_ in chunks:
        n=len(indices)
        qp=torch.stack(paths[offset:offset+n])
        prefix=torch.cat((qp.new_zeros(1),qp.detach().cumsum(0)[:-1]))
        trunk=torch.logsumexp(prefix+torch.stack(covers[offset:offset+n]),0)
        over=torch.logsumexp(prefix+torch.stack(overs[offset:offset+n]),0)
        path=qp.sum()
        if not bool(torch.isfinite(trunk)) or bool(trunk < path.detach()-2e-5):
            raise FloatingPointError('Invalid H1b trunk probability')
        result.append(ChunkLP(path,trunk.detach(),over.detach()))
        offset+=n
    return result
