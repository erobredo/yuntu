"""Base classes for audio processing pipelines."""
import os
from abc import ABC
from abc import abstractmethod
import warnings
from copy import copy
from collections import OrderedDict
import uuid
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from dask.threaded import get
from dask.optimization import cull
from dask.optimization import inline as dask_inline
from dask.optimization import inline_functions
from dask.optimization import fuse
from dask.base import compute as group_compute

DASK_CONFIG = {'npartitions': 1}


class Node(ABC):
    """Pipeline node."""
    data_class = None
    node_type = "abstract"
    is_output = False
    is_transition = False
    is_place = False

    def __init__(self, name=None, pipeline=None):
        if name is not None:
            if not isinstance(name, str):
                message = "Node name must be a string."
                raise ValueError(message)
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.pipeline = pipeline
        self._key = None
        self._index = None
        self._result = None

    @property
    def key(self):
        """Get key within pipeline if exists."""
        self.refresh_key()
        return self._key

    @property
    def index(self):
        """Get index within pipeline if exists."""
        self.refresh_index()
        return self._index

    @property
    def meta(self):
        meta = {"key": self.key,
                "name": self.name,
                "node_type": self.node_type}
        return meta

    def refresh_index(self):
        """Retrieve index from pipeline if exists and update."""
        if self.pipeline is None:
            self._index = None
        elif self._index is not None:
            if self._index >= len(self.pipeline):
                self._index = None
            elif self.pipeline[self._index] != self:
                self._index = None
        else:
            keys = list(self.pipeline.nodes.keys())
            for i in range(len(keys)):
                if self.pipeline.nodes[keys[i]] == self:
                    self._index = i

    def refresh_key(self):
        """Retrieve key from pipeline if exists and update."""
        if self.pipeline is None:
            self._key = None
        elif self._key is not None:
            if self._key not in self.pipeline.nodes:
                self._key = None
            elif self.pipeline.nodes[self._key] != self:
                self._key = None
        else:
            for key in self.pipeline.keys():
                if self.pipeline.nodes[key] == self:
                    self._key = key
                    break

    def set_value(self, value):
        """Set result value manually."""
        if not self.validate(value):
            raise ValueError("Value is incompatible with node type " +
                             f"{type(self)}")
        self._result = value

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self._key = None
        self._index = None

    def attach(self):
        """Attach self to pipeline."""
        if self.pipeline is None:
            message = "This node does not belong to any pipeline. Please " + \
                      " assign a pipeline using method 'set_pipeline'."
            raise ValueError(message)
        if self.key not in self.pipeline:
            self.pipeline.add_node(self)
            self.refresh_key()
            self.refresh_index()
        elif self.pipeline[self.key] != self:
            self.pipeline[self.key] = self

    def validate(self, data):
        """Validate data."""
        if data is None:
            return True
        if self.data_class is None:
            return True
        return isinstance(data, self.data_class)

    def plot(self, **kwargs):
        if self.pipeline is None:
            raise ValueError("No pipeline context. Set pipeline first.")
        nodes = [self.key]
        if "nodes" in kwargs:
            if not isinstance(kwargs["nodes"], (tuple, list)):
                raise ValueError("Argument 'nodes' must be a list or a tuple")
            nodes = list(set(nodes + list(kwargs["nodes"])))
        return self.pipeline.plot(nodes=nodes, **kwargs)

    def __rshift__(self, other):
        """Embed self in other's pipeline.

        Merge other's pipeline with self's pipeline and return other.
        """
        if not isinstance(other, Node):
            raise ValueError("Both operands must be nodes.")
        other.pipeline.merge(self.pipeline)
        return other

    def __lshift__(self, other):
        """Embed other in self's pipeline.

        Merge self's pipeline with other's pipeline and return self.
        """
        if not isinstance(other, Node):
            raise ValueError("Both operands must be nodes.")
        self.pipeline.merge(other.pipeline)
        return self

    @abstractmethod
    def is_compatible(self, other):
        """Check if nodes is compatible with self for replacement."""

    @abstractmethod
    def future(self,
               feed=None,
               read=None,
               force=False,
               **kwargs):
        """Return node's future."""

    @abstractmethod
    def compute(self,
                feed=None,
                read=None,
                write=None,
                keep=None,
                force=False,
                **kwargs):
        """Compute self."""

    @abstractmethod
    def __call__(self,
                 client=None,
                 strict=False,
                 feed=None,
                 read=None,
                 write=False,
                 keep=None,
                 **inputs):
        """Execute node on inputs."""

    @abstractmethod
    def __copy__(self):
        """Return a shallow copy of self."""


class MetaPipeline(ABC):
    """Base class for processing pipelines."""

    def __init__(self, name):
        if name is None:
            message = "A pipeline must have a name."
            raise ValueError(message)
        self.name = name
        self.nodes = OrderedDict()
        self.places = OrderedDict()
        self.transitions = OrderedDict()
        self.nodes_up = OrderedDict()
        self.nodes_down = OrderedDict()

    @property
    def struct(self):
        """Full networkx directed acyclic graph."""
        return self.build_struct()

    @property
    def outputs(self):
        """Pipeline outputs."""
        return [key for key in self.nodes
                if len(self.nodes_down[key]) == 0]

    @property
    def inputs(self):
        inputs = [key for key in self.places
                  if len(self.nodes_up[key]) == 0]
        for node in inputs:
            yield node

    @property
    def operations(self):
        operations = [key for key in self.transitions]
        for node in operations:
            yield node

    @property
    def names(self):
        """Return iterator of names."""
        keys = list(self.keys())
        unique_names = list(set([self.nodes[key].name for key in keys]))
        for name in unique_names:
            yield name

    @property
    def identifiers(self):
        """Return an iterator of tuples of the form (index, key, name)."""
        keys = list(self.keys())
        for i in range(len(keys)):
            yield i, keys[i], self.nodes[keys[i]].name

    def add_node(self, node):
        """Add node to pipeline."""
        if node.is_transition:
            self.add_transition(node)
        elif node.is_place:
            self.add_place(node)
        else:
            message = ("Argument 'node' must be an object of class " +
                       "Place or Tansition.")
            raise ValueError(message)

    def add_place(self, place):
        """Add place node to pipeline.

        Add new place to pipeline.

        Parameters
        ----------
        place: Place
            Place to add.

        Raises
        ------
        ValueError
            If place exists or is not of class Place.
        """
        if not isinstance(place, Node):
            raise ValueError("Argument 'place' must be of class Node.")
        if not place.is_place:
            raise ValueError("Argument 'place' must be of class Place.")
        if place in self:
            raise ValueError(f"Node exists.")
        place.set_pipeline(self)
        key = place.name
        if key is not None:
            if key in self.nodes:
                key = f"{key}-{uuid.uuid1()}"
        else:
            key = f"place-{uuid.uuid1()}"

        self.nodes[key] = place
        self.places[key] = place
        self.add_neighbours(place.key)

    def add_transition(self, transition):
        """Add transition node to pipeline.

        Add new transition to pipeline.

        Parameters
        ----------
        place: Place
            Transition to add.

        Raises
        ------
        ValueError
            If transition exists or is not of class Transition.
        """
        if not isinstance(transition, Node):
            raise ValueError("Argument 'transition' must be of class " +
                             "Node.")
        if not transition.is_transition:
            raise ValueError("Argument 'transition' must be of class " +
                             "Transition.")
        if transition in self:
            raise ValueError(f"Node exists.")
        transition.set_pipeline(self)
        key = transition.name
        if key is not None:
            if key in self.nodes:
                key = f"{key}-{uuid.uuid1()}"
        else:
            key = f"transition-{uuid.uuid1()}"

        self.nodes[key] = transition
        self.transitions[key] = transition
        self.add_neighbours(transition.key)

    def add_neighbours(self, key):
        """Add dependencies to pipeline in case they are absent.

        Parameters
        ----------
        key: str
            Node key within pipeline.

        Raises
        ------
        KeyError
            If key does not exist within pipeline.
        """
        if key not in self.nodes:
            message = "Key not found."
            raise KeyError(message)

        nodes_up = []
        nodes_down = []

        if key in self.places:
            parent = self.nodes[key].parent
            if parent is not None:
                if parent.pipeline != self:
                    parent.set_pipeline(self)
                if parent.key is None:
                    parent.attach()
                nodes_up.append(parent.key)
            for tkey in self.transitions:
                if tkey in self.nodes_up:
                    if key in self.nodes_up[tkey]:
                        nodes_down.append(tkey)
        else:
            for dep in self.nodes[key].inputs:
                if dep.key is None:
                    dep.set_pipeline(self)
                    dep.attach()
                else:
                    if dep.key not in self.nodes:
                        dep.set_pipeline(self)
                        dep.attach()
                    elif dep.pipeline != self:
                        dep.set_pipeline(self)
                        dep.attach()

                nodes_up.append(dep.key)
                if key not in self.nodes_down[dep.key]:
                    self.nodes_down[dep.key].append(key)

            for child in self.nodes[key].outputs:
                if child.key is None:
                    child.set_parent(self.nodes[key])
                    child.attach()
                else:
                    if child.key not in self.nodes or child.pipeline != self:
                        child.set_parent(self.nodes[key])
                        child.attach()
                    else:
                        child.set_parent(self.nodes[key])
                        if key not in self.nodes_up[child.key]:
                            self.nodes_up[child.key] = [key]

                nodes_down.append(child.key)

        self.nodes_up[key] = nodes_up
        self.nodes_down[key] = nodes_down

    def get_parent(self, key):
        """Get parent for place with key."""
        if key not in self.nodes_up:
            return None
        if len(self.nodes_up[key]) == 0:
            return None
        return self.transitions[self.nodes_up[key][0]]

    def upper_neighbours(self, key):
        """Return node's upper neighbours.

        Parameters
        ----------
        key: str
            Node key within pipeline.

        Returns
        -------
        upper: list
            Upper neighbours as list of objects of class Node.

        Raises
        ------
        KeyError
            If key does not exist within pipeline's nodes or dependencies.
        """
        if key not in self.nodes:
            message = "Key not found."
            raise KeyError(message)
        if key not in self.nodes_up:
            message = "Upper neighbours have not been assigned."
            raise KeyError(message)

        return [self.nodes[dkey] for dkey in self.nodes_up[key]]

    def lower_neighbours(self, key):
        """Return node's lower neighbours.

        Parameters
        ----------
        key: str
            Node key within pipeline.

        Returns
        -------
        lower: list
            Lower neighbours as list of objects of class Node.

        Raises
        ------
        KeyError
            If key does not exist within pipeline's nodes or dependencies.
        """
        if key not in self.nodes:
            message = "Key not found."
            raise KeyError(message)
        if key not in self.nodes_down:
            message = "Lower neighbours have not been assigned."
            raise KeyError(message)

        return [self.nodes[dkey] for dkey in self.nodes_down[key]]

    def node_key(self, index):
        """Return node key if argument makes sense as index or as key.

        Parameters
        ----------
        index: int, str
            The index or key to search for.

        Returns
        -------
        key: str
            A valid node key within pipeline.

        Raises
        ------
        IndexError
            If pipeline is empty or index is an integer greater than pipeline's
            length.
        KeyError
            If argument 'index' is a string that is not a key within pipeline.
        ValueError
            If index is not an integer nor a string.
        """
        if len(self) == 0:
            raise IndexError("Pipeline is empty.")
        if isinstance(index, str):
            if index not in self.nodes:
                raise KeyError(f"Key {index} does not exist.")
            return index
        if not isinstance(index, int):
            raise ValueError("Index must be an integer or a string.")
        if index >= len(self):
            raise IndexError("Index must be smaller than pipeline length.")
        keys = list(self.nodes.keys())
        return keys[index]

    def node_index(self, key):
        """Return node index if argument makes sense as key or as index.

        Parameters
        ----------
        key: str, int
            The key or index to search for.

        Returns
        -------
        index: int
            A valid index within pipeline.

        Raises
        ------
        IndexError
            If pipeline is empty or key is an integer that is greater than
            pipeline's length.
        ValueError
            If argument is not a string or an integer.
        KeyError
            If key is a string that is not a key within pipeline.
        """
        if len(self) == 0:
            raise IndexError("Pipeline is empty.")
        if isinstance(key, int):
            if key >= len(self):
                raise IndexError("If argument is integer, input value" +
                                 " must be lower than pipeline length.")
            return key
        if not isinstance(key, str):
            raise ValueError("Argument must be a string.")
        for i in range(len(self.nodes.items())):
            if self.nodes.items()[i][0] == key:
                return i
        raise KeyError(f"Key {key} does not exist")

    def node_name(self, key):
        """Return node name.

        Parameters
        ----------
        key: str, int
            A valid node key or index.

        Returns
        -------
        name: str
            Node name.

        Raises
        ------
        KeyError
            If key/index does not exist or pipeline is empty.
        TypeError
            If argument is not an integer or string.
        """
        if len(self) == 0:
            raise KeyError("Pipeline is empty.")
        if isinstance(key, int):
            key = self.node_key(key)
        elif isinstance(key, str):
            if key not in self.nodes:
                raise KeyError("Node not found in this pipeline.")
        else:
            raise TypeError("Argument must be a string or an integer.")
        return self.nodes[key]

    def build_struct(self, nodes=None):
        """Return networkx graph from pipeline computation structure.

        Return computation graph as a networkx DiGraph. If node list is not
        None, a graph including nodes in list is generated with all their
        immeadiate dependencies.

        Parameters
        ----------
        nodes: list
            A list of nodes to include in graph. If None, all nodes are
            included.

        Returns
        -------
        graph: networkx.DiGraph
            The resulting graph.

        Raises
        ------
        ValueError
            If argument 'nodes' is not None and is not a list or tuple.
        KeyError
            If node list is not a subset of pipeline node keys.
        """
        if nodes is not None:
            if not isinstance(nodes, (list, tuple)):
                message = "Argument 'nodes' must be a list."
                raise ValueError(message)

            if not set(nodes).issubset(set(self.keys())):
                message = "All strings in argument 'nodes' must be node keys."
                raise KeyError(message)
        else:
            nodes = self.keys()

        G = nx.DiGraph()

        for key in nodes:
            G.add_node(key, **self.nodes[key].meta)

        for key in nodes:
            for dkey in self.nodes_up[key]:
                if dkey is not None:
                    if dkey not in G:
                        G.add_node(dkey, **self.nodes[dkey].meta)
                    edge = (dkey, key)
                    G.add_edge(*edge)
                    if dkey in self.transitions:
                        for sdkey in self.nodes_up[dkey]:
                            if sdkey not in G:
                                G.add_node(sdkey, **self.nodes[sdkey].meta)
                            if sdkey != key:
                                edge = (sdkey, dkey)
                                G.add_edge(*edge)
                        for ddkey in self.nodes_down[dkey]:
                            if ddkey not in G:
                                G.add_node(ddkey, **self.nodes[ddkey].meta)
                            if ddkey != key:
                                edge = (dkey, ddkey)
                                G.add_edge(*edge)

            for dkey in self.nodes_down[key]:
                if dkey is not None:
                    if dkey not in G:
                        G.add_node(dkey, **self.nodes[dkey].meta)
                    edge = (key, dkey)
                    G.add_edge(*edge)
                    if dkey in self.transitions:
                        for sdkey in self.nodes_up[dkey]:
                            if sdkey not in G:
                                G.add_node(sdkey, **self.nodes[sdkey].meta)
                            if sdkey != key:
                                edge = (sdkey, dkey)
                                G.add_edge(*edge)
                        for ddkey in self.nodes_down[dkey]:
                            if ddkey not in G:
                                G.add_node(ddkey, **self.nodes[ddkey].meta)
                            if ddkey != key:
                                edge = (dkey, ddkey)
                                G.add_edge(*edge)
        return G

    def node_order(self, nodes=None):
        """Return node order within pipeline.

        The order of a node is the length of the longest path from any input.
        """
        if nodes is not None:
            if not isinstance(nodes, (list, tuple)):
                message = "Argument 'nodes' must be a list of node keys."
                raise ValueError(message)
            for key in nodes:
                if key not in self.nodes:
                    raise KeyError(f"Key {key} from arguments 'nodes' not " +
                                   "found within pipeline.")
        else:
            nodes = list(self.keys())

        G = self.struct
        operation_order = OrderedDict()
        all_inputs = self.inputs
        for key in nodes:
            max_path_length = 0
            if key not in all_inputs:
                for source in all_inputs:
                    path = self.shortest_path(source=source,
                                              target=key,
                                              nxgraph=G)
                    if path is not None:
                        path_length = len(path)
                        if path_length > max_path_length:
                            max_path_length = path_length
            operation_order[key] = max_path_length
        return operation_order

    def shortest_path(self, source, target, nxgraph=None):
        """Return shortest path from node 'source' to node 'target'."""
        if source not in self.nodes or target not in self.nodes:
            message = ("Arguments 'source' and 'target' must be valid node " +
                       "keys.")
            raise KeyError(message)

        if nxgraph is None:
            nxgraph = self.struct

        try:
            path = nx.shortest_path(nxgraph, source=source, target=target)
        except nx.NetworkXNoPath:
            path = None

        return path

    def plot(self,
             ax=None,
             nodes=None,
             trans_color="white",
             trans_edge_color="black",
             trans_size=6500,
             trans_shape="s",
             place_size=7500,
             place_color="lightgrey",
             place_edge_color="black",
             place_shape="o",
             input_color="tomato",
             keep_color="gold",
             persist_color="goldenrod",
             keep_persist_color="yellowgreen",
             font_size=14,
             font_color="black",
             font_weight=1.0,
             font_family='sans-serif',
             font_alpha=1.0,
             min_target_margin=60,
             min_source_margin=40,
             head_length=0.6,
             head_width=0.8,
             tail_width=0.4,
             arrow_style='Fancy',
             arrow_width=3.0,
             graph_layout="dot",
             node_labels=None,
             plot_style='fivethirtyeight',
             **kwargs):
        """Plot pipeline's graph."""
        if graph_layout is None:
            graph_layout = "twopi"
        # use_graphviz = True
        # if hasattr(nx.drawing.layout, graph_layout):
        #     use_graphviz = False

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', (20, 20)))

        plt.style.use(plot_style)

        G = self.build_struct(nodes)
        input_places = []
        regular_places = []
        keep_places = []
        persist_places = []
        keep_persist_places = []
        transitions = []
        for key, meta in G.nodes(data=True):
            if meta["node_type"] == "transition":
                transitions.append(key)
            else:
                if key in self.inputs:
                    input_places.append(key)
                elif self.nodes[key].persist:
                    if self.nodes[key].keep:
                        keep_persist_places.append(key)
                    else:
                        persist_places.append(key)
                elif self.nodes[key].keep:
                    keep_places.append(key)
                else:
                    regular_places.append(key)

        pos = graphviz_layout(G, prog=graph_layout, args='')
        # if use_graphviz:
        #     pos = graphviz_layout(G, prog=graph_layout, args='')
        # else:
        #     pos = getattr(nx.drawing.layout, graph_layout)

        if len(regular_places) > 0:
            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   ax=ax,
                                   nodelist=regular_places,
                                   node_size=place_size,
                                   node_color=place_color,
                                   node_shape=place_shape,
                                   edgecolors=place_edge_color)
        if len(input_places) > 0:
            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   ax=ax,
                                   nodelist=input_places,
                                   node_size=place_size,
                                   node_color=input_color,
                                   node_shape=place_shape,
                                   edgecolors=place_edge_color)
        if len(keep_places) > 0:
            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   ax=ax,
                                   nodelist=keep_places,
                                   node_size=place_size,
                                   node_color=keep_color,
                                   node_shape=place_shape,
                                   edgecolors=place_edge_color)
        if len(persist_places) > 0:
            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   ax=ax,
                                   nodelist=persist_places,
                                   node_size=place_size,
                                   node_color=persist_color,
                                   node_shape=place_shape,
                                   edgecolors=place_edge_color)
        if len(keep_persist_places) > 0:
            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   ax=ax,
                                   nodelist=keep_persist_places,
                                   node_size=place_size,
                                   node_color=keep_persist_color,
                                   node_shape=place_shape,
                                   edgecolors=place_edge_color)
        if len(transitions) > 0:
            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   ax=ax,
                                   nodelist=transitions,
                                   node_size=trans_size,
                                   node_color=trans_color,
                                   node_shape=trans_shape,
                                   edgecolors=trans_edge_color)

        astyle = mpl.patches.ArrowStyle(arrow_style,
                                        head_length=head_length,
                                        head_width=head_width,
                                        tail_width=tail_width)

        nx.draw_networkx_edges(G,
                               pos=pos,
                               ax=ax,
                               edge_color="black",
                               width=arrow_width,
                               arrowstyle=astyle,
                               min_target_margin=min_target_margin,
                               min_source_margin=min_source_margin)

        nx.draw_networkx_labels(G,
                                pos=pos,
                                ax=ax,
                                labels=node_labels,
                                font_size=font_size,
                                font_color=font_color,
                                font_weight=font_weight,
                                font_family=font_family,
                                alpha=font_alpha
                                )
        return ax

    def keys(self):
        """Return iterator of keys."""
        return self.nodes.keys()

    @abstractmethod
    def get_node(self, key, feed=None, compute=False, force=False, **kwargs):
        """Get node from pipeline graph."""

    @abstractmethod
    def compute(self,
                nodes=None,
                feed=None,
                read=None,
                write=False,
                keep=None,
                force=False,
                **kwargs):
        """Compute pipeline."""

    @abstractmethod
    def __copy__(self):
        """Copy self."""

    def __call__(self,
                 client=None,
                 strict=False,
                 feed=None,
                 read=None,
                 write=None,
                 keep=None,
                 **inputs):
        """Execute pipeline on inputs."""
        if feed is not None:
            if not isinstance(feed, dict):
                message = "Argument 'feed' must be a dictionary."
                raise ValueError(message)
        else:
            feed = {}

        if len(inputs) > 0:
            ikeys = list(inputs.keys())
            for key in ikeys:
                if key not in self.inputs:
                    if strict:
                        message = (f"Unknown input {key} for pipeline " +
                                   f"{self.name}. Use 'strict=False' to " +
                                   "evaluate this pipeline " +
                                   "ignoring unknown parameters without " +
                                   "raising errors. A warning will still " +
                                   "be raised.")
                        raise ValueError(message)
                    message = (f"Dropping unknown input {key} for pipeline " +
                               f"{self.name}.")
                    warnings.warn(message)
                    del inputs[key]
                else:
                    if not self.places[key].validate(inputs[key]):
                        data_class = self.places[key].data_class
                        message = (f"Wrong type for parameter {key}" +
                                   f". Pipeline expects: {data_class}")
                        raise TypeError(message)

        feed.update(inputs)

        return self.compute(client=client,
                            feed=feed,
                            read=read,
                            write=write,
                            keep=keep,
                            force=True)

    def __len__(self):
        """Return the number of pipeline nodes."""
        return len(self.nodes)

    def __getitem__(self, key):
        """Return node with key."""
        key = self.node_key(key)
        return self.nodes[key]

    def __delitem__(self, key):
        key = self.node_key(key)
        if self.nodes[key].is_place:
            if self.nodes[key].parent is not None:
                parent_key = self.nodes[key].parent.key
                message = (f"This node is an output place for {parent_key}" +
                           f" transition. Delete node {parent_key} fisrt. " +
                           "If you want to set this place use " +
                           "pipeline['node_key'] = new_node")
                raise ValueError(message)
        self.nodes[key].clear()
        del self.nodes_up[key]
        del self.nodes_down[key]
        del self.nodes[key]

    def _replace_neighbour(self, key, before, after, dir="up"):
        if dir == "up":
            for ind, node in enumerate(self.nodes_up[key]):
                if node == before:
                    self.nodes_up[key][ind] = after
        else:
            for ind, node in enumerate(self.nodes_down[key]):
                if node == before:
                    self.nodes_down[key][ind] = after

    def __setitem__(self, key, value):
        """Set node with key to value."""
        if not isinstance(value, Node):
            if key not in self.nodes:
                raise KeyError("Key not found. Setting node result value is" +
                               " only allowded for existing nodes.")
            self.nodes[key].set_value(value)
        else:
            node_exists = False
            if key in self.nodes:
                if not self.nodes[key].is_compatible(value):
                    raise TypeError("Can not replace existing "
                                    "node by value. Nodes are "
                                    "incompatible.")

                node_exists = True
            if value in self:
                if value.key != key:
                    if node_exists:
                        nxgraph = self.struct
                        pathr = self.shortest_path(value.key, key, nxgraph)
                        if pathr is not None:
                            raise ValueError("Cycles are not permitted.")

                    if value.key in self.transitions or not node_exists:
                        if node_exists and value.is_transition:
                            inputs = []
                            outputs = []
                            for ind, node in enumerate(self.nodes_up[key]):
                                anc_key = self.nodes_up[value.key][ind]
                                self[node] = self.nodes[anc_key]
                                inputs.append(self.nodes[node])

                            down_enum = list(enumerate(self.nodes_down[key]))
                            for ind, node in down_enum:
                                dec_key = self.nodes_down[value.key][ind]
                                self._replace_neighbour(node,
                                                        key,
                                                        value.key)
                                self[node] = self.nodes[dec_key]
                                outputs.append(self.nodes[node])
                            value.set_inputs(inputs)
                            value.set_outputs(outputs)

                        for node_key in self.nodes:
                            self._replace_neighbour(node_key,
                                                    value.key,
                                                    key)
                            self._replace_neighbour(node_key,
                                                    value.key,
                                                    key,
                                                    "down")
                    else:
                        for node_key in self.nodes:
                            self._replace_neighbour(node_key,
                                                    value.key,
                                                    key)
                        if value.parent is not None:
                            if self.nodes[key].parent is not None:
                                val_parent = value.parent
                                curr_parent = self.nodes[key].parent
                                if curr_parent != val_parent:
                                    val_parent_key = val_parent.key
                                    val_parent_children = self.nodes_down[val_parent_key]
                                    new_child = copy(value)
                                    new_child.set_parent(val_parent)
                                    new_child.attach()
                                    value.set_parent(curr_parent)
                                    self.nodes_up[value.key] = [curr_parent.key]
                                    for ind, node in enumerate(val_parent_children):
                                        if node == value.key:
                                            val_parent_children[ind] = new_child.key
                                    self.nodes_down[val_parent_key] = val_parent_children
                                    self._replace_neighbour(curr_parent.key,
                                                            value.key,
                                                            new_child.key,
                                                            "down")
                        for node_key in self.nodes:
                            self._replace_neighbour(node_key,
                                                    value.key,
                                                    key,
                                                    'down')

                    prev_key = value.key
                    self.nodes[key] = self.nodes.pop(prev_key)
                    self.nodes_up[key] = self.nodes_up.pop(prev_key)
                    self.nodes_down[key] = self.nodes_down.pop(prev_key)
                    if self.nodes[key].is_transition:
                        self.transitions[key] = self.transitions.pop(prev_key)
                    else:
                        self.places[key] = self.places.pop(prev_key)
                    self.nodes[key].refresh_key()
            else:
                value.set_pipeline(self)
                self.nodes[key] = value
                if self.nodes[key].is_transition:
                    self.transitions[key] = self.nodes[key]
                else:
                    self.places[key] = self.nodes[key]
                self.nodes[key].refresh_key()
                self.add_neighbours(key)

    def __iter__(self):
        """Return node iterator."""
        for key in self.nodes:
            yield key

    def __contains__(self, node):
        """Return true if item in pipeline."""
        if isinstance(node, Node):
            nodes = list(self.nodes.items())
            return node in [item[1] for item in nodes]
        return node in list(self.nodes.keys())


class Pipeline(MetaPipeline):
    """Processing flux that uses dask as a parallel processing manager."""

    def __init__(self,
                 name,
                 work_dir=None,
                 **kwargs):
        super().__init__(name)
        if work_dir is None:
            work_dir = "/tmp"
        if not os.path.isdir(work_dir):
            message = "Argument 'work_dir' must be a valid directory."
            raise ValueError(message)
        self.work_dir = work_dir

    def build(self):
        """Add operations that are specific to each pipeline."""

    def build_graph(self, nodes=None, feed=None, linearize=None):
        """Build a dask computing graph."""
        if len(self.nodes) == 0:
            raise ValueError("Can not buid a graph from an empty pipeline.")
        if feed is not None:
            if not isinstance(feed, dict):
                message = ("Feed argument must be a dictionary with node" +
                           " names as arguments and input data as values.")
                raise ValueError(message)

        if nodes is not None:
            if not isinstance(nodes, (tuple, list)):
                message = "First argument must be a list of node keys."
                raise ValueError(message)
            for key in nodes:
                if key not in self.nodes:
                    raise KeyError(f"Key {key} not found.")
        else:
            nodes = self.outputs

        G = self.struct

        feed_keys = list(feed.keys())
        for key in feed_keys:
            for nkey in nodes:
                if nkey == key:
                    del feed[key]
                if self.shortest_path(nkey, key, nxgraph=G) is not None:
                    del feed[key]

        feed_keys = list(feed.keys())
        for key in feed_keys:
            for nkey in feed_keys:
                if key != nkey:
                    if self.shortest_path(nkey, key, nxgraph=G) is not None:
                        del feed[key]

        graph = {}
        to_include = []
        if feed is not None:
            for key in feed:
                node = self.nodes[key]
                if not node.validate(feed[key]):
                    data_class = node.data_class
                    raise ValueError("Feeding data is invalid for node " +
                                     f"'{key}' expecting {data_class}.")
                graph[key] = feed[key]
                for nkey in nodes:
                    for path in nx.all_simple_paths(G, key, nkey):
                        to_include += path

        to_include = list(set(to_include))
        for nkey in nodes:
            for ikey in self.inputs:
                if ikey not in to_include:
                    if self.shortest_path(ikey, nkey, nxgraph=G) is not None:
                        for path in nx.all_simple_paths(G, ikey, nkey):
                            include = True
                            for key in feed:
                                if key in path:
                                    include = False
                            if include:
                                to_include += path

        to_include = list(set(to_include))
        for key in to_include:
            node = self.nodes[key]
            if key not in self.nodes_down or key not in self.nodes_up:
                message = (f"Neighbours for node {key} have not been" +
                           " assigned yet.")
                raise KeyError(message)
            if key in self.places:
                if key in self.inputs:
                    if key not in graph:
                        graph[key] = node.data
            else:
                if key not in graph:
                    inputs = []
                    for dkey in self.nodes_up[key]:
                        inputs.append(dkey)
                    graph[key] = (node.operation, *inputs)

                down_nodes = self.nodes_down[key]
                n_down = len(down_nodes)
                for i in range(n_down):
                    dkey = down_nodes[i]
                    if dkey not in graph:
                        index_node = f"{dkey}_index"
                        if n_down == 1:
                            graph[index_node] = None
                        else:
                            graph[index_node] = i
                        dargs = (key, index_node)
                        graph[dkey] = (project, *dargs)

        if linearize is not None:
            if not isinstance(linearize, (tuple, list)):
                message = ("Argument 'linearize' must be a tuple or a list " +
                           "of node names.")
                raise ValueError(message)
            for key in linearize:
                if not isinstance(key, str):
                    message = ("Node names within 'linearize' must be " +
                               "strings.")
                    raise KeyError(message)
                if key not in self.nodes:
                    message = (f"No node named {key} within this pipeline")
            graph = linearize_operations(linearize, graph)

        return graph

    @property
    def persist_dir(self):
        """Initialize pipeline."""
        base_dir = os.path.join(self.work_dir, self.name)
        persist_dir = os.path.join(base_dir, 'persist')
        return persist_dir

    def init_dirs(self):
        base_dir = os.path.join(self.work_dir, self.name)
        persist_dir = os.path.join(base_dir, 'persist')
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(persist_dir):
            os.mkdir(persist_dir)

    @property
    def graph(self):
        """Returns full dask computation graph."""
        return self.build_graph()

    def clear(self):
        """Clear all persisted information."""
        for name in self.nodes:
            self.nodes[name].clear()

    def get_nodes(self,
                  nodes,
                  feed=None,
                  read=None,
                  write=None,
                  keep=None,
                  compute=False,
                  force=False,
                  client=None,
                  linearize=None,
                  scheduler="threads"):
        if len(nodes) == 0:
            raise ValueError("At least one node must be specified.")

        for key in nodes:
            if key not in self.nodes:
                raise ValueError(f'No node with key {key}')

        if feed is not None:
            if not isinstance(feed, dict):
                message = ("Argument 'feed' must be a dictionary of node " +
                           "keys")
                raise ValueError(message)
            for key in feed:
                if key not in self.nodes:
                    message = f"Key {key} from feed dict not found in nodes."
                    raise KeyError(message)
        else:
            feed = {}

        if read is not None:
            if not isinstance(read, dict):
                message = ("Argument 'read' must be a dictionary of node " +
                           "keys")
                raise ValueError(message)
            for key in read:
                if key not in self.nodes:
                    message = f"Key {key} from read dict not found in nodes."
                    raise KeyError(message)
                if isinstance(read[key], str):
                    path = read[key]
                    if not os.path.exists(read[key]):
                        message = f"Path {path} does not exist"
                        raise ValueError(message)
        else:
            read = {}

        if keep is not None:
            if not isinstance(keep, dict):
                message = ("Argument 'keep' must be a dictionary of node " +
                           "keys")
                raise ValueError(message)
            for key in keep:
                if key not in self.nodes:
                    message = f"Key {key} from keep dict not found in nodes."
                    raise KeyError(message)
        else:
            keep = {}

        if write is not None:
            if not isinstance(keep, dict):
                message = ("Argument 'write' must be a dictionary of node " +
                           "keys")
                raise ValueError(message)
            for key in write:
                if key not in self.nodes:
                    message = f"Key {key} from write dict not found in nodes."
                    raise KeyError(message)
                if not self.nodes[key].can_persist:
                    message = (f"Node {key} is a dynamic node and can not be" +
                               " persisted automatically. You can retrieve " +
                               " the result or future and save them manually.")
                    raise ValueError(message)

        else:
            write = {}

        nxgraph = self.struct

        feed_keys = list(feed.keys())
        for key in feed_keys:
            for nkey in nodes:
                if nkey != key:
                    if self.shortest_path(nkey, key, nxgraph) is not None:
                        if key in feed:
                            del feed[key]

        for key in self.places:
            if key not in read:
                if self.places[key].is_persisted() and not force:
                    read[key] = True

        read_keys = list(read.keys())
        for key in read_keys:
            for nkey in nodes:
                if self.shortest_path(nkey, key, nxgraph) is not None:
                    if key in read:
                        del read[key]
            for fkey in feed:
                if self.shortest_path(fkey, key, nxgraph) is not None:
                    if key in read:
                        del read[key]
            for rkey in read_keys:
                if self.shortest_path(rkey, key, nxgraph) is not None:
                    if key in read and key != rkey:
                        del read[key]

        for key in nodes:
            if key not in keep:
                keep[key] = self.nodes[key].keep

        write_keys = list(write.keys())
        for key in write_keys:
            if key not in nodes:
                if key in write:
                    del write[key]

        for key in nodes:
            if key in self.places and key not in write:
                if self.nodes[key].can_persist:
                    write[key] = self.nodes[key].persist

        if len(write) > 0:
            self.init_dirs()

        for key in read:
            if isinstance(read[key], bool):
                if read[key]:
                    node = self.nodes[key].read()
                    feed[key] = node
            else:
                node = self.nodes[key].read(path=read[key])
                feed[key] = node

        graph = self.build_graph(nodes=nodes, feed=feed, linearize=linearize)

        results = {}
        if client is not None:
            retrieved = client.get(graph,
                                   nodes,
                                   sync=False)
            retrieved = [x.result() for x in retrieved]
            if compute:
                retrieved = group_compute(*retrieved, scheduler="distributed")
        else:
            retrieved = get(graph, nodes)
            if compute:
                retrieved = group_compute(*retrieved, scheduler=scheduler)

        for ind, xnode in enumerate(retrieved):
            key = nodes[ind]
            if compute and hasattr(xnode, 'compute'):
                node = xnode.compute()
                if not self.nodes[key].can_persist:
                    data = xnode
                else:
                    data = node
                if key in self.places:
                    if key in write:
                        if isinstance(write[key], str):
                            path = write[key]
                            self.nodes[key].write(path=path, data=data)
                        elif write[key]:
                            self.nodes[key].write(data=data)
                if key in keep:
                    if keep[key]:
                        self.nodes[key].set_value(data)
                results[key] = node
            elif key in self.transitions and self.nodes[key].coarity > 0:
                if compute:
                    if self.nodes[key].coarity > 1:
                        computed = []
                        for oind in range(self.nodes[key].coarity):
                            sub_node = xnode[oind]
                            if hasattr(sub_node, "compute"):
                                computed.append(sub_node.compute())
                            else:
                                computed.append(sub_node)
                        results[key] = tuple(computed)
                    else:
                        if hasattr(xnode, "compute"):
                            results[key] = xnode.compute()
                        else:
                            results[key] = xnode
                else:
                    results[key] = xnode
            else:
                results[key] = xnode
                if compute and self.nodes[key].can_persist:
                    if key in self.places:
                        if key in write:
                            if isinstance(write[key], str):
                                path = write[key]
                                self.nodes[key].write(path=path, data=xnode)
                            elif write[key]:
                                self.nodes[key].write(data=xnode)

        return results

    def get_node(self,
                 key,
                 feed=None,
                 read=None,
                 write=None,
                 keep=None,
                 compute=False,
                 force=False,
                 client=None,
                 linearize=None):
        """Get node from pipeline graph."""
        return self.get_nodes(nodes=[key],
                              feed=feed,
                              read=read,
                              write=write,
                              keep=keep,
                              compute=compute,
                              force=force,
                              client=client,
                              linearize=linearize)[key]

    def compute(self,
                nodes=None,
                feed=None,
                read=None,
                write=None,
                keep=None,
                force=False,
                client=None,
                linearize=None):
        """Compute pipeline."""
        if nodes is None:
            nodes = self.outputs
        elif len(nodes) == 0:
            nodes = self.outputs

        if not isinstance(nodes, (tuple, list)):
            message = "Argument 'nodes' must be a tuple or a list."
            raise ValueError(message)

        return self.get_nodes(nodes,
                              feed=feed,
                              read=read,
                              write=write,
                              keep=keep,
                              client=client,
                              compute=True,
                              force=force,
                              linearize=linearize)

    def prune(self):
        """Remove all nodes without any neighbours."""
        G = self.struct
        keys = list(self.keys())
        for key in keys:
            ancestors = [node for node in
                         nx.algorithms.dag.ancestors(G, key)]
            descendants = [node for node in
                           nx.algorithms.dag.descendants(G, key)]
            if len(descendants) == 0 and len(ancestors) == 0:
                del self[key]

    def union(self, other):
        """Disjoint parallel union of self and other."""
        _ = union(self, other, new=False)

    def merge(self,
              other,
              on="key",
              knit_points=None,
              prune=False):
        """Indetify nodes from different pipelines and build a new pipeline.

        Use node keys or names to identify nodes between pipelines directly
        or specify key to key mappings in argument 'knit_points'.
        Ambiguous specifications such as key dupicates for new keys in any
        pipeline as well as incompatibilities between nodes will throw
        exceptions. The resulting pipeline's 'work_dir' will be that of the
        first operand.

        Parameters
        ----------
        other: Pipeline
            Pipeline to knit with.
        on: str
            Specify either 'key' or 'name' to define knitting points in case
            of identification.
        knit_points: dict
            A dictionary that defines key mapping between pipelines and new
            keys and names to be set with attributes of the form:
            <second_pipeline_key>: {
                'map': <first_pipeline_key>,
                'new_key': <key_in_resulting_pipeline>,
                'new_name': <name_in_resulting_pipeline>
            }

        Returns
        -------
        pipeline: Piepeline
            New pipeline with identified nodes.

        Raises
        ------
        ValueError
            If argument 'on' is not 'key' or 'name' or if names are duplicated
            within any of the pipelines.
        KeyError
            If one of the mapping keys does not exist in any of the pipelines.
        """
        if self != other:
            _ = merge(self, other,
                      on=on,
                      knit_points=knit_points,
                      prune=prune,
                      new=False)

    def collapse(self, prune=False):
        """Identify all compatible nodes with the same name."""
        _ = collapse(self, new=False, prune=prune)

    def __copy__(self):
        """Return a shallow copy of self."""
        name = f"copy({self.name})"
        work_dir = self.work_dir
        new_pipeline = Pipeline(name, work_dir=work_dir)

        for key in self.inputs:
            new_pipeline[key] = copy(self.nodes[key])

        for key in self.transitions:
            if key not in new_pipeline:
                node = self.nodes[key]
                new_node = copy(node)
                new_node.set_pipeline(new_pipeline)
                new_inputs = []
                new_outputs = []

                for ind, inode in enumerate(node.inputs):
                    ikey = inode.key
                    if ikey not in new_pipeline:
                        new_in = copy(inode)
                        new_in.set_pipeline(new_pipeline)
                    else:
                        new_in = new_pipeline[ikey]
                    new_inputs.append(new_in)

                for ind, onode in enumerate(node.outputs):
                    okey = onode.key
                    if okey not in new_pipeline:
                        new_out = copy(onode)
                        new_out.set_pipeline(new_pipeline)
                    else:
                        new_out = new_pipeline[okey]
                    new_outputs.append(new_out)

                new_node.set_inputs(new_inputs)
                new_node.set_outputs(new_outputs)

                for ind, inode in enumerate(node.inputs):
                    ikey = inode.key
                    new_pipeline[ikey] = new_node.inputs[ind]

                for ind, onode in enumerate(node.outputs):
                    okey = onode.key
                    new_pipeline[okey] = new_node.outputs[ind]

                new_pipeline[key] = new_node

        return new_pipeline

    def __and__(self, other):
        """Intetrwine operator '&'.

        Returns a new pipeline that is the result of replacing nodes in the
        second pipeline with nodes from the first pipeline by key
        identification. Nodes from the first pipeline keep all their
        dependencies, disconnecting second pipeline's dependencies.
        Orphan nodes are removed at the end of the process. All other nodes are
        preserved. It folows that if 'p' and 'q' are two pipelines that do not
        share any keys then:
            p & q = p | q
        On the other hand, if 'e' is an empty pipeline and 'p' is any pipeline,
        then the former assumption is True. Additionally we have that
            p & e = p
        and
            p | e = p
        but also
            p & p = p
        The last relation does not hold for disjoint parallelism:
            p | p != p
        The resulting pipeline's 'work_dir' will be that of the first operand.
        """
        name = self.name + "&" + other.name
        new_pipeline = merge(self, other)
        new_pipeline.name = name

        return new_pipeline

    def __gt__(self, other):
        """Return pipeline concatenation by input/output keys.

        If pipelines have no inputs/outputs in common, the result is
        equivalent to disjoint paralelism.
        """
        knit_points = {}
        for key in self.outputs:
            if key in other.inputs:
                knit_points[key] = {"map": key,
                                    "new_key": key,
                                    "new_name": self.nodes[key].name}
        if len(knit_points) == 0:
            new_pipeline = union(self, other)
        else:
            new_pipeline = merge(self, other,
                                 knit_points=knit_points,
                                 prune=False)

        return new_pipeline

    def __lt__(self, other):
        """Return pipeline concatenation by input/output keys.

        If pipelines have no inputs/outputs in common, the result is
        equivalent to disjoint paralelism.
        """
        message = "Only forward concatenation is allowded."
        raise NotImplementedError(message)

    def __or__(self, other):
        """Disjoint sum operator '|'.

        Returns a new pipeline with nodes from both pipelines running in
        disjoint paralellism. The resulting pipeline's 'work_dir' will be that
        of the first operand.

        Parameters
        ----------
        other: Pipeline
            Pipeline to add.

        Returns
        -------
        pipeline: Pipeline
            New pipeline with all nodes from both pipelines, possibly renaming
            some of the nodes where duplicate key's are met.
        """
        name = self.name + "|" + other.name

        new_pipeline = union(self, other)
        new_pipeline.name = name

        return new_pipeline

    def __rshift__(self, other):
        """Embed self in other.

        Merge other with self and return other.
        """
        if not isinstance(other, Pipeline):
            raise ValueError("Both operands must be pipelines.")
        other.merge(self)
        return other

    def __lshift__(self, other):
        """Embed other in self.

        Merge self with other and return self.
        """
        if not isinstance(other, Pipeline):
            raise ValueError("Both operands must be pipelines.")
        self.merge(other)
        return self


def linearize_operations(op_names, graph):
    """Linearize dask operations."""
    graph1, deps = cull(graph, op_names)
    graph2 = dask_inline(graph1, dependencies=deps)
    graph3 = inline_functions(graph2,
                              op_names,
                              [len, str.split],
                              dependencies=deps)
    graph4, deps = fuse(graph3)
    return graph4


def project(output, index):
    if index is not None:
        return output[index]
    return output


def union(p1, p2, new=True):
    if not new:
        new_pipeline = p1
    else:
        name = f"union({p1.name},{p2.name})"
        new_pipeline = copy(p1)
        new_pipeline.name = name

    node_map = {}
    for key in p2.inputs:
        node = p2.nodes[key]
        new_node = copy(node)
        new_pipeline.add_place(new_node)
        node_map[key] = new_node

    for key in p2.transitions:
        if key not in node_map:
            node = p2.nodes[key]
            new_node = copy(node)
            new_node.set_pipeline(new_pipeline)
            new_inputs = []
            new_outputs = []

            for ind, inode in enumerate(node.inputs):
                ikey = inode.key
                if ikey not in node_map:
                    new_in = copy(inode)
                    new_in.set_pipeline(new_pipeline)
                    node_map[ikey] = new_in
                else:
                    new_in = node_map[ikey]
                new_inputs.append(new_in)

            for ind, onode in enumerate(node.outputs):
                okey = onode.key
                if okey not in node_map:
                    new_out = copy(onode)
                    new_out.set_pipeline(new_pipeline)
                    node_map[okey] = new_out
                else:
                    new_out = node_map[okey]
                new_outputs.append(new_out)

            new_node.set_inputs(new_inputs)
            new_node.set_outputs(new_outputs)

            new_node.attach()
            node_map[key] = new_node

    return new_pipeline


def merge(p1, p2, on='key', knit_points=None, new=True, prune=True):
    if on is None:
        on = "key"

    if knit_points:
        if not isinstance(knit_points, dict):
            message = "Argument 'knit_points' must be a dictionary."
            raise ValueError(message)

        knit_keys = list(knit_points.keys())
        for key in knit_keys:
            p1_key = knit_points[key]["map"]
            new_key = knit_points[key]["new_key"]
            safe = True

            if new_key in p1.keys():
                if new_key != p1_key:
                    safe = False
            if new_key in p2.keys():
                if new_key != key:
                    safe = False

            if not safe:
                message = ("Attribute 'new_key' of knit points " +
                           "should be a brand new key within both " +
                           "pipelines, otherwise entry key,  " +
                           "'new_key' and 'map' should be equal.")
                raise ValueError(message)

            if key not in p2.keys():
                message = (f"Key {key} not found in second pipeline. " +
                           "All keys in argument " +
                           "'knit_points' must exist within the second " +
                           "pipeline. Removing entry from" +
                           " knitting points.")
                warnings.warn(message)
                del knit_points[key]

            if p1_key not in p1.keys():
                message = (f"Key {p1_key} not found " +
                           "in first pipeline. All map keys in argument " +
                           "'knit_points' must exist within the first " +
                           "pipeline. Removing entry from" +
                           " knitting points.")
                warnings.warn(message)
                del knit_points[key]

            if not p1.nodes[p1_key].is_compatible(p2.nodes[key]):
                message = (f"Node {p1_key} of first pipeline is not " +
                           f"compatible with node {key} of second " +
                           "pipeline. Removing entry from" +
                           " knitting points.")
                warnings.warn(message)
                del knit_points[key]
    else:
        if on not in ["key", "name"]:
            raise ValueError("Argument 'by' should be 'key' or 'name'.")

        knit_points = {}
        if on == "key":
            common_keys = []
            for p1_key in p1.nodes:
                for p2_key in p2.nodes:
                    if (p1_key == p2_key
                       and p1.nodes[p1_key].is_compatible(p2.nodes[p2_key])
                       and p1_key not in common_keys):
                        common_keys.append(p1_key)
            knit_points = {}
            for key in common_keys:
                knit_points[key] = {'map': key,
                                    'new_key': key,
                                    'new_name': p1.nodes[key].name}
        else:
            if (len(p1.keys()) != len(p1.names) or
               len(p2.keys() != len(p2.names))):
                message = ("Names are duplicated within one of the " +
                           "pipelines. Change names, prune pipelines " +
                           "or use on='key'")
                raise ValueError(message)

            common_names = list(set(list(p1.names)) &
                                set(list(p2.names)))
            for name in common_names:
                p2_key = None
                for key in p2.nodes:
                    if p2.nodes[key].name == name:
                        p2_key = key
                        break

                p1_key = None
                for key in p1.nodes:
                    if p1.nodes[key].name == name:
                        p1_key = key
                        break

                if (p1.nodes[p1_key].is_compatible(p2.nodes[p2_key])
                   and p2_key not in knit_points):
                    knit_points[p2_key] = {"map": p1_key,
                                           "new_key": p1_key,
                                           "new_name": name}

    if not new:
        new_pipeline = p1
    else:
        name = f"merge({p1.name},{p2.name})"
        new_pipeline = copy(p1)
        new_pipeline.name = name

    do_not_knit = []
    for key in p2.nodes:
        if key not in knit_points:
            do_not_knit.append(key)
    nxgraph = p2.struct

    knit_keys = list(knit_points.keys())
    for key in knit_keys:
        for not_key in do_not_knit:
            if p2.shortest_path(not_key, key, nxgraph) is not None:
                del knit_points[key]
                do_not_knit.append(key)
                break

    knit_points_inv = {}
    for key in knit_points:
        new_key = knit_points[key]["new_key"]
        new_name = knit_points[key]["new_name"]
        p1_key = knit_points[key]["map"]
        new_pipeline[new_key] = new_pipeline[p1_key]
        new_pipeline[new_key].name = new_name
        knit_points_inv[p1_key] = {"new_key": new_key,
                                   "map": key,
                                   "new_name": new_name}
    node_map = {}
    orders = p2.node_order()
    all_orders = list(set(orders[ordkey]
                          for ordkey in orders))
    all_orders.sort()
    for level in all_orders:
        level_keys = [key for key in p2.nodes
                      if orders[key] == level]
        for key in level_keys:
            if key not in node_map:
                if key in knit_points:
                    new_key = knit_points[key]["new_key"]
                else:
                    new_key = key
                if new_key not in new_pipeline or new_key in do_not_knit:
                    node = p2.nodes[key]
                    if key in p2.inputs:
                        new_node = copy(node)
                        new_node.set_pipeline(new_pipeline)
                        new_node.attach()
                        node_map[key] = new_node
                    elif key in p2.transitions:
                        new_node = copy(node)
                        new_node.set_pipeline(new_pipeline)
                        new_inputs = []
                        new_outputs = []

                        for ind, inode in enumerate(node.inputs):
                            ikey = inode.key
                            if ikey in knit_points:
                                new_ikey = knit_points[ikey]["new_key"]
                            else:
                                new_ikey = ikey
                            if (new_ikey not in new_pipeline or
                               new_ikey in do_not_knit):
                                if ikey not in node_map:
                                    new_in = copy(inode)
                                    new_in.set_pipeline(new_pipeline)
                                    node_map[ikey] = new_in
                                else:
                                    new_in = node_map[ikey]
                            else:
                                new_in = new_pipeline[new_ikey]

                            new_inputs.append(new_in)

                        for ind, onode in enumerate(node.outputs):
                            okey = onode.key
                            if okey in knit_points:
                                new_okey = knit_points[okey]["new_key"]
                            else:
                                new_okey = okey
                            if (new_okey not in new_pipeline or
                               new_okey in do_not_knit):
                                if okey not in node_map:
                                    new_out = copy(onode)
                                    new_out.set_pipeline(new_pipeline)
                                    node_map[okey] = new_out
                                else:
                                    new_out = node_map[okey]
                            else:
                                new_out = new_pipeline[new_okey]
                            new_outputs.append(new_out)

                        new_node.set_inputs(new_inputs)
                        new_node.set_outputs(new_outputs)

                        for ind, inode in enumerate(node.inputs):
                            ikey = inode.key
                            if new_node.inputs[ind] not in new_pipeline:
                                if (ikey not in new_pipeline.keys() and
                                   ikey not in do_not_knit):
                                    new_pipeline[ikey] = new_node.inputs[ind]

                        for ind, onode in enumerate(node.outputs):
                            okey = onode.key
                            if new_node.outputs[ind] not in new_pipeline:
                                if (okey not in new_pipeline.keys() and
                                   ikey not in do_not_knit and
                                   key not in do_not_knit):
                                    new_pipeline[okey] = new_node.outputs[ind]

                        if new_key not in do_not_knit:
                            new_pipeline[new_key] = new_node
                        else:
                            new_node.attach()
                        node_map[key] = new_node
                else:
                    node_map[key] = new_pipeline[new_key]

    if prune:
        new_pipeline.prune()

    return new_pipeline


def collapse(pipeline, new=True, prune=True):
    """Identify all compatible nodes with the same name within pipeline."""
    if new:
        dpipeline = copy(pipeline)
        dpipeline.name = f'collapse({pipeline.name})'
    else:
        dpipeline = pipeline

    nxgraph = dpipeline.struct
    node_cats = {}
    for name in dpipeline.names:
        if name not in node_cats:
            node_cats[name] = {}
        for key in dpipeline.nodes:
            node = dpipeline.nodes[key]
            if name == dpipeline.nodes[key].name:
                found = False
                for rkey in node_cats[name]:
                    if key != rkey:
                        if node.is_compatible(node_cats[name][rkey][0]):
                            spath = dpipeline.shortest_path(rkey,
                                                            node.key,
                                                            nxgraph)
                            if spath is None:
                                node_cats[name][rkey].append(node)
                                found = True
                                break
                if not found:
                    node_cats[name][key] = [node]
    for name in node_cats:
        for rkey in node_cats[name]:
            for node in node_cats[name][rkey]:
                dpipeline[rkey] = node

    if prune:
        dpipeline.prune()
    return dpipeline
