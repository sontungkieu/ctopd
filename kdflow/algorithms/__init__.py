"""Algorithms included in the submission."""

ALGO_DICT = {}


def register_algorithm(name):
    def decorator(cls):
        ALGO_DICT[name] = cls
        return cls
    return decorator


from kdflow.algorithms.simple_ctkd import SimpleCrossTokenizerKD
from kdflow.algorithms.byte_align_ctkd import ByteAlignCrossTokenizerKD
from kdflow.algorithms.topk_byte_align_ctkd import TopKByteAlignCrossTokenizerKD

__all__ = ["ALGO_DICT", "SimpleCrossTokenizerKD", "ByteAlignCrossTokenizerKD", "TopKByteAlignCrossTokenizerKD"]
