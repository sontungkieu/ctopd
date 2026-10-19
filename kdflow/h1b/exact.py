"""Finite exhaustive cover-path enumeration with a certified omitted-mass bound."""
import math


def logadd(a,b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m=max(a,b)
    return m+math.log(math.exp(a-m)+math.exp(b-m))


def exact_chunk_logprob(next_logprobs,chunk,vi,prune=0.,max_nodes=10000):
    """next_logprobs(tuple_of_branch_ids) -> normalized logp vector.

    Empty/special token paths are outside this byte-cover event. Duplicate byte
    token IDs are distinct branches. Returned bounds remain valid on pruning or
    node-budget exhaustion; only zero omitted mass merits the label exact.
    """
    if not chunk or not 0<=prune<1 or max_nodes<1: raise ValueError('Bad exact-search contract')
    frontier=[(0.,0,())]; total=omitted=-math.inf; nodes=0
    while frontier:
        lp,used,path=frontier.pop()
        if nodes>=max_nodes:
            omitted=logadd(omitted,lp)
            continue
        row=next_logprobs(path); nodes+=1
        rem=chunk[used:]
        cover=-math.inf
        for token in vi.cover(rem): cover=logadd(cover,float(row[token]))
        total=logadd(total,lp+cover)
        for k in range(1,len(rem)):
            for token in vi.ids_of(rem[:k]):
                child=lp+float(row[token])
                if prune and child<total+math.log(prune): omitted=logadd(omitted,child)
                else: frontier.append((child,used+k,path+(token,)))
    upper=logadd(total,omitted)
    return dict(log_probability_lower=total,log_probability_upper=upper,
                log_omitted_mass_upper=omitted,nodes=nodes,exact=omitted==-math.inf)
