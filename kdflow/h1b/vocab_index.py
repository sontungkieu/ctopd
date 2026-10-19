"""Duplicate-preserving byte-prefix vocabulary index."""
from bisect import bisect_left
from functools import lru_cache
from kdflow.token_alignment import get_tokenizer_byte_codec, _special_token_ids


def next_prefix(value):
    for i in range(len(value)-1, -1, -1):
        if value[i] < 255:
            return value[:i] + bytes([value[i]+1])
    return None


class VocabIndex:
    def __init__(self, tok, codec=None):
        codec = codec or get_tokenizer_byte_codec(tok)
        self.special = _special_token_ids(tok)
        vocab = tok.get_vocab()
        self._bytes = {i: codec.piece_to_bytes(p) for p,i in vocab.items() if i not in self.special}
        pairs = sorted((b,i) for i,b in self._bytes.items() if b)
        self.keys, self.perm = [b for b,_ in pairs], [i for _,i in pairs]
        self._ids = {}
        for b,i in pairs: self._ids.setdefault(b, []).append(i)
        self.duplicate_byte_strings = sum(len(v)>1 for v in self._ids.values())

    def bytes(self, token_id):
        return self._bytes[int(token_id)]

    def ids_of(self, value):
        return tuple(self._ids.get(value, ()))

    def id_of(self, value):
        ids = self.ids_of(value)
        return ids[0] if ids else None

    @lru_cache(maxsize=16384)
    def cover(self, rem):
        if not rem: raise ValueError('Empty remaining chunk is not a token event')
        high = next_prefix(rem)
        lo, hi = bisect_left(self.keys, rem), (len(self.keys) if high is None else bisect_left(self.keys, high))
        return tuple(self.perm[lo:hi])

    def cover_ids(self, rem, device=None):
        import torch
        return torch.tensor(self.cover(rem), dtype=torch.long, device=device)
