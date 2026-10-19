"""Pinned author's SimCT code; never register its legacy name in KDFlow."""


def register_algorithm(_name):
    return lambda cls: cls
