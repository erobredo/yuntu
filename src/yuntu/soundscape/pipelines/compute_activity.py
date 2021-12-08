# from yuntu.core.pipeline.base import Pipeline
# from yuntu.core.pipeline.places.extended import place
# from yuntu.soundscape.transitions.basic import get_partitions
# from yuntu.soundscape.transitions.activity import compute_activity
#
# class ActivityFromAnnotations(Pipeline):
#     """Pipeline to apply probe using dask."""
#
#     def __init__(self,
#                  name,
#                  collection_config,
#                  min_t,
#                  max_t,
#                  query=None,
#                  limit=None,
#                  offset=0,
#                  time_unit=60,
#                  time_module=None,
#                  target_labels=None,
#                  exclude=[],
#                  **kwargs):
#
#         super().__init__(name, **kwargs)
#
#         self.time_unit = time_unit
#         self.min_t = min_t
#         self.max_t = max_t
#         self.target_labels = target_labels
#         self.query = query
#         self.limit = limit
#         self.offset = offset
#         self.collection_config = collection_config
#         self.time_module = time_module
#         self.exclude = exclude
#         self.build()
#
#     def build(self):
#         self["col_config"] = place(self.collection_config, 'dict', 'col_config')
#         self["query"] = place(self.query, 'dynamic', 'query')
#         self["npartitions"] = place(1, 'scalar', 'npartitions')
#         self["limit"] = place(self.limit, 'scalar', 'limit')
#         self["offset"] = place(self.offset, 'scalar', 'offset')
#         self["target"] = place("annotations", "dynamic", "target")
#         self["partitions"] = get_partitions(self["col_config"],
#                                             self["query"],
#                                             self["npartitions"],
#                                             self["limit"],
#                                             self["offset"],
#                                             self["target"])
#         config = {
#             'time_unit': self.time_unit,
#             'min_t': self.min_t,
#             'max_t': self.max_t,
#             'time_module': self.time_module,
#             'target_labels': self.target_labels,
#             'exclude': self.exclude
#         }
#
#         self["activity_config"] = place(config, 'dict', 'activity_config')
#         self["activities"] = compute_activity(self["partitions"],
#                                               self["activity_config"],
#                                               self["col_config"])
