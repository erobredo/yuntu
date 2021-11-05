# from yuntu.core.pipeline.base import Pipeline
# from yuntu.core.pipeline.places.extended import place
# from yuntu.soundscape.transitions.activity import get_activities
#
# class ActivityFromAnnotations(Pipeline):
#     """Pipeline to apply probe using dask."""
#
#     def __init__(self,
#                  name,
#                  annotations,
#                  min_t,
#                  max_t,
#                  time_unit=60,
#                  target_labels=None,
#                  **kwargs):
#
#         if not isinstance(probe_config, dict):
#             raise ValueError("Argument 'probe_config' must be a dictionary.")
#
#         super().__init__(name, **kwargs)
#
#         self.annotations = annotations
#         self.time_unit = time_unit
#         self.min_t = min_t
#         self.max_t = max_t
#         self.target_labels = target_labels
#         self.build()
#
#     def build(self):
#         self['annotations'] = place(data=self.annotations,
#                                     name='annotations',
#                                     ptype='pandas_dataframe')
#         self['npartitions'] = place(data=10,
#                                     name='npartitions',
#                                     ptype='scalar')
#         config = {
#             'time_unit': self.time_unit,
#             'min_t': self.min_t,
#             'max_t': self.max_t,
#             'target_labels': self.target_labels
#         }
#
#         self["config"] = place(self.time_unit, 'dict', 'config')
#         self['annotations_bag'] = bag_dataframe(self['annotations'],
#                                                 self['npartitions'])
#         self["activities"] = get_activities(self["annotations_bag"],
#                                             self["config"])
