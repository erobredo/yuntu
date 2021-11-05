"""Operation pipeline nodes."""
from copy import copy
from yuntu.core.pipeline.base import Node
from yuntu.core.pipeline.base import Pipeline


class Transition(Node):
    """Pipeline transition base node."""
    signature = None
    operation_class = None
    node_type = "transition"

    def __init__(self,
                 *args,
                 operation=None,
                 inputs=None,
                 outputs=None,
                 is_output=None,
                 persist=None,
                 keep=None,
                 signature=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if signature is None:
            message = "A signature must be provided."
            raise ValueError(message)
        if not isinstance(signature, tuple):
            message = "Signature must be a tuple of length 2."
            raise ValueError(message)
        if len(signature) != 2:
            message = "Signature must be a tuple of length 2."
            raise ValueError(message)
        for class_list in signature:
            if not isinstance(class_list, tuple):
                message = "Signature elements must be tuples as well."
                raise ValueError(message)
        self.signature = signature
        if not hasattr(operation, "__call__"):
            message = "Argument 'operation' must be a callable object."
        if not self.validate_operation(operation):
            raise ValueError(f"Operation type {type(operation)} incorrect " +
                             "for this type of operation node.")

        if not isinstance(inputs, (list, tuple)):
            message = ("Argument 'inputs' must be a list or a " +
                       "tuple of nodes.")
            raise ValueError(message)
        sig_inputs_len = len(self.signature[0])
        if len(inputs) != sig_inputs_len:
            message = ("Wrong number of inputs. " +
                       f"Transition expects {sig_inputs_len}")
            raise ValueError(message)

        if not isinstance(outputs, (list, tuple)):
            message = ("Argument 'outputs' must be a list or a " +
                       "tuple of nodes.")
            raise ValueError(message)
        sig_outputs_len = len(self.signature[1])
        if len(outputs) != sig_outputs_len:
            message = ("Wrong number of outputs. " +
                       f"Transition produces {sig_outputs_len}")
            raise ValueError(message)

        for index, node in enumerate(inputs):
            if not isinstance(node, Node):
                message = ("At least one of the inputs is not a node.")
                raise ValueError(message)
            if not isinstance(node, self.signature[0][index]):
                message = (f"Place at position {index} is not compatible.")
                raise ValueError(message)
        if is_output is not None:
            if not isinstance(is_output, (bool, list, tuple)):
                raise ValueError("Argument 'is_output' must be bool, list or" +
                                 "tuple.")
            if isinstance(is_output, (list, tuple)):
                if len(is_output) != sig_outputs_len:
                    message = ("If argument 'is_output' is a list, then it " +
                               "must have the same length as 'outputs'.")
                    raise ValueError(message)
            else:
                is_output = [is_output for i in range(sig_outputs_len)]
            for val in is_output:
                if not isinstance(val, bool):
                    raise ValueError("All elements of 'is_output' must be" +
                                     " of type bool.")

        if keep is not None:
            if not isinstance(keep, (bool, list, tuple)):
                raise ValueError("Argument 'keep' must be bool, list or" +
                                 "tuple.")
            if isinstance(keep, (list, tuple)):
                if len(keep) != sig_outputs_len:
                    message = ("If argument 'keep' is a list, then it must " +
                               "have the same length as 'outputs'.")
                    raise ValueError(message)
            else:
                keep = [keep for i in range(sig_outputs_len)]
            for val in keep:
                if not isinstance(val, bool):
                    raise ValueError("All elements of 'keep' must be bool.")

        if persist is not None:
            if not isinstance(persist, (bool, list, tuple)):
                raise ValueError("Argument 'persist' must be bool, list or" +
                                 "tuple.")
            if isinstance(persist, (list, tuple)):
                if len(persist) != sig_outputs_len:
                    message = ("If argument 'persist' is a list, then it " +
                               "must have the same length as 'outputs'.")
                    raise ValueError(message)
            else:
                persist = [persist for i in range(sig_outputs_len)]
            for val in persist:
                if not isinstance(val, bool):
                    raise ValueError("All elements of 'persist' must be bool.")

            for index, node in enumerate(outputs):
                if not isinstance(node, Node):
                    message = ("At least one of the outputs is not a node.")
                    raise ValueError(message)
                if not isinstance(node, self.signature[1][index]):
                    message = (f"Output at position {index} is not" +
                               " compatible.")
                    raise ValueError(message)

        for index, node in enumerate(outputs):
            node.set_parent(self)

        if keep is not None:
            for index, node in enumerate(outputs):
                node.keep = keep[index]
        if persist is not None:
            for index, node in enumerate(outputs):
                node.persist = persist[index]
        if is_output is not None:
            for index, node in enumerate(outputs):
                node.is_output = is_output[index]

        self._inputs = inputs
        self._outputs = outputs
        self.operation = operation
        self.keep = False

        if self.pipeline is None:
            self.pipeline = self._places_pipeline()

        if self.pipeline is None:
            self.pipeline = Pipeline(name=self.name)

        self.attach()

    def set_value(self, value):
        """Set result value manually."""
        self._result = value

    def set_pipeline(self, pipeline):
        """Set pipeline for self and dependencies if needed."""
        self.pipeline = pipeline

    def _places_pipeline(self):
        """Try to guess pipeline from initial inputs."""
        pipeline = None
        for node in self._inputs + self._outputs:
            if pipeline is None:
                pipeline = node.pipeline
            else:
                pipeline = pipeline.merge(node.pipeline)
        return pipeline

    @property
    def is_transition(self):
        return True

    @property
    def is_place(self):
        return False

    @property
    def meta(self):
        meta = {"key": self.key,
                "name": self.name,
                "node_type": self.node_type,
                "keep": self.keep}
        return meta

    @property
    def inputs(self):
        if self.pipeline is None or self.key is None:
            return self._inputs
        if self.key not in self.pipeline.nodes_up:
            return self._inputs
        deps = self.pipeline.upper_neighbours(self.key)
        for node in deps:
            if node is None:
                return self._inputs
        return deps

    @property
    def outputs(self):
        if self.pipeline is None or self.key is None:
            return self._outputs
        if self.key not in self.pipeline.nodes_down:
            return self._outputs
        subs = self.pipeline.lower_neighbours(self.key)
        for node in subs:
            if node is None:
                return self._outputs
        return subs

    @property
    def arity(self):
        return len(self.signature[0])

    @property
    def coarity(self):
        return len(self.signature[1])

    def is_compatible(self, other):
        """Check if nodes is compatible with self for replacement."""
        if not other.is_transition:
            return False
        return self.signature == other.signature

    def set_inputs(self, places):
        """Set hard value for inputs (when pipeline is None)"""
        if not isinstance(places, (tuple, list)):
            raise ValueError("Argument must be a list.")
        for node in places:
            if not isinstance(node, Node):
                raise ValueError("All inputs must be nodes.")
            if node.node_type != "place":
                raise ValueError("All inputs must be places.")
        self._inputs = places
        if self.pipeline is not None and self.key is not None:
            for i in range(len(self._inputs)):
                if self._inputs[i].pipeline != self.pipeline:
                    self._inputs[i].set_pipeline(self.pipeline)
                    if self._inputs[i].key not in self.pipeline:
                        self._inputs[i].attach()
                self.pipeline.nodes_up[self.key][i] = self._inputs[i].key

    def set_outputs(self, places):
        """Set hard value for inputs (when pipeline is None)"""
        if not isinstance(places, (tuple, list)):
            raise ValueError("Argument must be a list.")
        for node in places:
            if not isinstance(node, Node):
                raise ValueError("All elements must be nodes.")
            if node.node_type != "place":
                raise ValueError("All inputs must be places.")
        self._outputs = places
        if self.pipeline is not None and self.key is not None:
            for i in range(len(self._outputs)):
                if self._outputs[i].pipeline != self.pipeline:
                    self._outputs[i].set_pipeline(self.pipeline)
                    if self._outputs[i].key not in self.pipeline:
                        self._outputs[i].attach()
                self.pipeline.nodes_down[self.key][i] = self._outputs[i].key

    def validate_operation(self, operation):
        """Validates method to be set according to operation type."""
        if self.operation_class is None:
            return True
        return isinstance(operation, self.operation_class)

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
            message = ("This node does not belong to any pipeline. " +
                       "Please assign a pipeline using method " +
                       "'set_pipeline'.")
            raise ValueError(message)
        if feed is not None:
            if self.key in feed:
                del feed[self.key]

        if self._result is not None and not force:
            return self._result

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
        """Execute node on inputs."""
        if feed is not None:
            if not isinstance(feed, dict):
                message = "Argument 'feed' must be a dictionary."
                raise ValueError(message)
        else:
            feed = {}
        if len(inputs) > 0:
            ikeys = list(inputs.keys())
            for key in ikeys:
                if key not in self.pipeline.nodes_up[self.key]:
                    message = (f"Unknown input {key} for transition " +
                               f"{self.key}.")
                    raise ValueError(message)
                if not self.pipeline.places[key].validate(inputs[key]):
                    data_class = self.pipeline.places[key].data_class
                    message = (f"Wrong type for parameter {key}" +
                               f". Transition expects: {data_class}")
                    raise TypeError(message)

        feed.update(inputs)
        return self.compute(feed=feed,
                            read=read,
                            write=write,
                            keep=keep,
                            force=True,
                            client=client)

    def __copy__(self):
        inputs = [copy(node) for node in self.inputs]
        outputs = [copy(node) for node in self.outputs]
        inp_sig = []
        out_sig = []
        for inp in inputs:
            inp_sig.append(inp.__class__)
        for out in outputs:
            out_sig.append(out.__class__)
        signature = (tuple(inp_sig), tuple(out_sig))
        op_class = self.__class__
        return op_class(name=self.name,
                        pipeline=None,
                        operation=self.operation,
                        inputs=inputs,
                        outputs=outputs,
                        signature=signature)
