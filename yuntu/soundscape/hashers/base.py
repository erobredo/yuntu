"""Row hashers for category assingment."""
from abc import ABC
from abc import abstractmethod

class Hasher(ABC):
    name = "hasher"
    columns = "__all__"

    @abstractmethod
    def hash(self, row, out_name="hash"):
        """Return row hash."""

    @abstractmethod
    def unhash(self, hashed):
        """Invert hash."""

    def validate(self, df):
        """Check it self can be applied to dataframe."""
        if self.columns != "__all__":
            df_cols = df.columns
            for col in self.columns:
                if col not in df_cols:
                    return False
        return True

    @property
    @abstractmethod
    def dtype(self):
        """Return row hash."""

    def __call__(self, row, **kwargs):
        """Hash object."""
        return self.hash(row, **kwargs)


class GenericHasher(Hasher, ABC):
    name = "generic_hasher"
    def __init__(self, hash_method, unhash_method=None, columns="__all__"):
        if not hasattr(hash_method, "__call__"):
            raise ValueError("Argument 'hash_method' must"
                             "be a callable object.")
        if unhash_method is not None:
            if not hasattr(hash_method, "__call__"):
                raise ValueError("Argument 'unhash_method' must be a"
                                 "callable object.")
        if columns != "__all__":
            if not isinstance(columns, (tuple, list)):
                raise ValueError("Argument 'columns' must be a list of "
                                 "column names.")
        self.columns = columns
        self._hash_method = hash_method
        self._unhash_method = unhash_method

    def hash(self, row, out_name="hash", **kwargs):
        """Return row hash."""
        new_row = {}
        new_row[out_name] = self._hash_method(new_row, **kwargs)
        return pd.Series(new_row)

    def unhash(self, hashed, **kwargs):
        """Invert hash (if possible)."""
        if self._unhash_method is not None:
            return self._unhash_method(hashed, **kwargs)
        raise NotImplementedError("Hasher has no inverse.")
