#
#
# def get_activity(annotations, config):
#     min_t = config["min_t"]
#     max_t = config["max_t"]
#     target_labels = config["target_labels"]
#     time_unit = config["time_unit"]
#
#     if min_t >= max_t:
#         raise ValueError("Wrong time range. Try a more accurate specification.")
#
#     total_time = datetime.timedelta.total_seconds(max_t - min_t)
#     nframes = int(np.round(total_time/time_unit))
#
#     activities = {}
#     if target_labels is None:
#         activity = np.zeros([nframes])
#         for row in annotations:
#             start_time = row["start_time"]
#             end_time = row["end_time"]
#             abs_start_time = row["abs_start_time"]
#             start = int(np.round(float(datetime.timedelta.total_seconds(abs_start_time - min_t))/time_unit))
#             stop = max(int(np.round(float(end_time - start_time)/time_unit)), start)
#             activity[start:stop+1] += 1
#         activities["Any"] = activity
#     else:
#         nlabels = len(target_labels)
#         for n in range(nlabels):
#             label_activity = np.zeros([nframes])
#             for row in in annotations:
#                 start_time = row["start_time"]
#                 end_time = row["end_time"]
#                 abs_start_time = row["abs_start_time"]
#                 for l in labels:
#                     if l["key"] == target_labels[n]["key"] and l["value"] == target_labels[n]["value"]:
#                         start = int(np.round(float(datetime.timedelta.total_seconds(abs_start_time - min_t))/time_unit))
#                         stop = max(int(np.round((end_time - start_time)/time_unit)), start)
#                         label_activity[start:stop+1] += 1
#             activities[target_labels[n]["value"]] = label_activity
#
#     return activities
