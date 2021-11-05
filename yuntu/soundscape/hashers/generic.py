"""Hashers that specify an input hash method."""
from yuntu.soundscape.hashers.base import GenericHasher

class StrHasher(GenericHasher):
    name = "string_hasher"
    @property
    def dtype(self):
        return np.dtype('str')


class IntHasher(GenericHasher):
    name = "integer_hasher"
    def __init__(self, *args, precision=32, **kwargs):
        if precision not in [8, 16, 32, 64]:
            raise ValueError("Argument precision must be 8, 16, 32 or 64.")
        super().__init__(*args, **kwargs)
        self.precision = precision

    @property
    def dtype(self):
        strtype = f"int{self.precision}"
        return np.dtype(strtype)
