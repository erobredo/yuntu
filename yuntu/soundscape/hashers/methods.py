"""Hasher importer and high level functions."""
from yuntu.soundscape.hashers.generic import StrHasher, IntHasher
from yuntu.soundscape.hashers.crono import CronoHasher

def hasher(hash_type, hash_method=None, unhash_method=None, **kwargs):
    """Return hasher instance by name."""
    if hash_type in ["string", "integer"]:
        if hash_method is None:
            raise ValueError("Hash method can not be undefined for generic"
                             " string and integer hashers.")
    if hash_type == "string":
        return StrHasher(hash_method, unhash_method=unhash_method, **kwargs)
    if hash_type == "integer":
        return IntHasher(hash_method, unhash_method=unhash_method, **kwargs)
    if hash_type == "crono":
        return CronoHasher(**kwargs)

    raise NotImplementedError(f"Hasher type unknown.")
