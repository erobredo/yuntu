"""Input pipeline nodes."""
from abc import ABC
from abc import abstractmethod
import os
import warnings
import pickle
import dill
import pandas as pd
from yuntu.core.pipeline.base import Node
from yuntu.core.pipeline.base import Pipeline


class Place(Node, ABC):
    """Pipeline place base class."""
    node_type = "place"

    def __init__(self,
                 *args,
                 data=None,
                 parent=None,
                 is_output=False,
                 persist=False,
                 keep=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not self.validate(data):
            message = "Data is invalid for this type of node."
            raise ValueError(message)
        if parent is not None:
            if not isinstance(parent, Node):
                message = "Argument 'parent' must be a transition node."
                raise ValueError(message)
            if parent.node_type != "transition":
                message = "Argument 'parent' must be a transition node."
                raise ValueError(message)
        self._parent = parent
        if self.can_persist:
            self.is_output = is_output
            self.persist = persist

        self.keep = keep
        self._result = data

        if self.pipeline is None:
            if self._parent is not None:
                self.pipeline = self._parent.pipeline

        if self.pipeline is None:
            self.pipeline = Pipeline(name=self.name)

        self.attach()

    @property
    def can_persist(self):
        return True

    def is_persisted(self):
        return os.path.exists(self.get_persist_path())

    def clear(self):
        if self.is_persisted():
            os.remove(self.get_persist_path())
        self._result = None

    @property
    def is_transition(self):
        return False

    @property
    def is_place(self):
        return True

    @property
    def meta(self):
        meta = {"key": self.key,
                "name": self.name,
                "node_type": self.node_type,
                "is_output": self.is_output,
                "persist": self.persist,
                "keep": self.keep}
        return meta

    @property
    def data(self):
        return self._result

    @property
    def parent(self):
        if self.pipeline is None or self.key is None:
            return self._parent
        parent = self.pipeline.get_parent(self.key)
        if parent is None:
            return self._parent
        return parent

    @abstractmethod
    def get_persist_path(self):
        """Path to operation persisted outputs."""

    @abstractmethod
    def read(self, path, **kwargs):
        """Read node from path."""

    @abstractmethod
    def write(self, path, **kwargs):
        """Write node to path."""

    def is_compatible(self, other):
        """Check if nodes is compatible with self for replacement."""
        if not other.is_place:
            return False
        return isinstance(self, other.__class__)

    def set_parent(self, parent):
        """Set parent."""
        if not isinstance(parent, Node):
            message = "Argument 'parent' must be a transition node."
            raise ValueError(message)
        if parent.node_type != "transition":
            message = "Argument 'parent' must be a transition node."
            raise ValueError(message)

        if self._parent != parent:
            self._parent = parent
        if self.parent.pipeline != self.pipeline:
            if self.pipeline is not None and self.key is not None:
                if self._parent.pipeline is not None:
                    if self._parent.pipeline != self.pipeline:
                        self.set_pipeline(self._parent.pipeline)
                        self.attach()

    def future(self,
               feed=None,
               read=None,
               force=False,
               **kwargs):
        """Return node's future."""
        if self.pipeline is None:
            message = "This node does not belong to any pipeline. " + \
                      "Please assign a pipeline using method " + \
                      "'set_pipeline'."
            raise ValueError(message)

        if isinstance(feed, dict):
            if self.key in feed:
                del feed[self.key]

        return self.pipeline.get_node(self.key,
                                      feed=feed,
                                      read=read,
                                      force=force,
                                      compute=False,
                                      **kwargs)

    def compute(self,
                feed=None,
                read=None,
                write=None,
                keep=None,
                force=False,
                **kwargs):
        """Compute self."""
        if self.pipeline is None:
            message = "This node does not belong to any pipeline. " + \
                      "Please assign a pipeline using method " + \
                      "'set_pipeline'."
            raise ValueError(message)

        if isinstance(feed, dict):
            if self.key in feed:
                del feed[self.key]

        result = self.pipeline.compute([self.key],
                                       feed=feed,
                                       read=read,
                                       write=write,
                                       keep=keep,
                                       force=force,
                                       **kwargs)[self.key]
        return result

    def __call__(self,
                 client=None,
                 strict=False,
                 feed=None,
                 read=None,
                 write=None,
                 keep=None,
                 **inputs):
        """Execute node results on local inputs."""
        if feed is not None:
            if not isinstance(feed, dict):
                message = "Argument 'feed' must be a dictionary."
                raise ValueError(message)
        else:
            feed = {}

        if self.parent is not None:
            if len(inputs) > 0:
                ikeys = list(inputs.keys())
                for key in ikeys:
                    if key not in self.pipeline.nodes_up[self.parent.key]:
                        message = (f"Unknown input {key} for parent " +
                                   f"transition {self.key}.")
                        raise ValueError(message)
                    if not self.pipeline.places[key].validate(inputs[key]):
                        data_class = self.pipeline.places[key].data_class
                        message = (f"Wrong type for parameter {key}" +
                                   ". Parent transition expects: " +
                                   f"{data_class}")
                        raise TypeError(message)
            feed.update(inputs)
            return self.compute(feed=feed,
                                read=read,
                                write=write,
                                keep=keep,
                                force=True,
                                client=client)

        if len(inputs) > 0:
            message = "No inputs needed. Ignoring."
            warnings.warn(message)

        return self.compute(feed=feed,
                            read=read,
                            write=write,
                            keep=keep,
                            force=True,
                            client=client)

    def __copy__(self):
        """Copy self."""
        place_class = self.__class__

        return place_class(name=self.name,
                           pipeline=None,
                           parent=None,
                           is_output=self.is_output,
                           persist=self.persist,
                           keep=self.keep,
                           data=self.data)


class DynamicPlace(Place):
    """A place that is never persisted."""

    @property
    def can_persist(self):
        return False

    @property
    def persist(self):
        return False

    @property
    def is_output(self):
        return False

    def is_persisted(self):
        return False

    def write(self, path=None, data=None):
        return None

    def read(self, path=None):
        return None

    def get_persist_path(self):
        return None


class PickleablePlace(Place):
    """Input that can be dumped to a pickle."""

    def validate(self, data):
        if data is None:
            return True
        if isinstance(data, pd.DataFrame):
            return True
        try:
            result = dill.pickles(data)
        except ValueError:
            result = False
        return result

    def write(self, path=None, data=None):
        if path is None:
            if self.pipeline is None:
                raise ValueError("Can not infer output path for node without a"
                                 "pipeline")
            self.pipeline.init_dirs()
            path = self.get_persist_path()
        if data is None:
            data = self.data
        if data is None:
            message = "No data to write, compute or pass data as argument."
            raise ValueError(message)
        if not self.validate(data):
            message = "Data is invalid."
            raise ValueError(message)
        with open(path, 'wb') as file:
            pickle.dump(data, file)

    def read(self, path=None):
        if path is None:
            path = self.get_persist_path()
        if not os.path.exists(path):
            message = "No pickled data at path."
            raise ValueError(message)
        with open(path) as file:
            data = pickle.load(file)
        return data

    def get_persist_path(self):
        if self.key is None:
            base_name = self.name
        else:
            base_name = self.key
        return os.path.join(self.pipeline.persist_dir, base_name+".pickle")


class BoolPlace(PickleablePlace):
    """Boolean place."""
    data_class = bool

    def validate(self, data):
        if data is None:
            return True
        return (super().validate(data)
                and isinstance(data, self.data_class))


class DictPlace(PickleablePlace):
    """Python dictionary place."""
    data_class = dict

    def validate(self, data):
        if data is None:
            return True
        return (super().validate(data)
                and isinstance(data, self.data_class))


class ScalarPlace(PickleablePlace):
    """Scalar places: numbers and strings."""
    data_class = (str, int, float)

    def validate(self, data):
        if data is None:
            return True
        return (super().validate(data)
                and isinstance(data, self.data_class))
