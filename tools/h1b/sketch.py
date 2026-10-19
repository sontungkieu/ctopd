"""Streaming CountSketch keyed by tensor name, independent of chunk boundaries."""
import hashlib
import torch


def count_sketch(named_tensors,dimension=2**22,seed=42,chunk_size=2**20):
    if dimension<1 or chunk_size<1: raise ValueError('Positive sketch dimensions required')
    output=torch.zeros(dimension,dtype=torch.float32)
    norm2=0.
    prime=2147483647
    for name,tensor in named_tensors:
        flat=tensor.detach().reshape(-1)
        digest=hashlib.sha256((str(seed)+':'+name).encode()).digest()
        a,b,c,d=[int.from_bytes(digest[i:i+8],'little')%(prime-1)+1 for i in (0,8,16,24)]
        for start in range(0,flat.numel(),chunk_size):
            values=flat[start:start+chunk_size].float().cpu()
            indices=torch.arange(start,start+len(values),dtype=torch.int64)
            bucket=((a*indices+b)%prime)%dimension
            signs=(((c*indices+d)%prime)%2)*2-1
            output.scatter_add_(0,bucket,values*signs)
            norm2+=float(values.double().square().sum())
    return output,norm2**.5
