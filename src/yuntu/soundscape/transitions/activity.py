# 
#
# @transition(name='compute_activity', outputs=["activity"],
#             keep=True, persist=True, is_output=True,
#             signature=((DaskBagPlace, PickleablePlace, ScalarPlace),
#                        (DaskDataFramePlace, )))
# def compute_activity(annotations, config):
#     meta = [("abs_start_time", np.dtype(float)),
#             ("abs_end_time", np.dtype(float))]
#     if config["target_labels"] is not None:
#         for l in config["target_labels"]:
#             meta.append((l["value"], np.dtype(float)))
#
#     results = annotations.map(lambda x: pd.DataFrame(x).get_activity(**config)).flatten()
#     ann_df = to_dataframe(meta=meta)
#
#     return ann_df.groupby(by=["abs_start_time", "abs_end_time"]).sum().reset_index()
