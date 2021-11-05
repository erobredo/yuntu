"""Audio with a type of local reference."""
from yuntu.core.audio.audio import Audio


class AnnotatedAudio(Audio):
    """Annotation extension of Audio class.

    This extension exposes methods that manage the presentation
    of data in fragments induced by time/frequency annotations.
    These methods include the production of binary masks representing
    the occurrence of a sound event in different ways that are useful
    for supervised learning.
    """


class GriddedAudio(Audio):
    """Audio class to produce sound grids.

    Add methods that produce regular local reference systems in time and
    frequency.
    """

    def __getitem__(self, key):
        pass

    def __iter__(self):
        pass

#
#     twin = 0.01
#     fwin = 50
#     thop = 0.005
#     fhop = 25
#     metadata.vegetacion = {"dadsf"} -> 'alias'
#     meta..
#     ------------
#     ------------
#     ------------
#     metadatos' = metadatos + ref_temporal y de frec
#     (i,j) -> spec X metadatos'
#
#
# Annotation(DB) <--> Annotation(Pony) <--> Annotation(Yuntu)
#
#
#
# AnnotatedObject:
#     - Annotation set(Yuntu)
#
#
# Annotation
#     - geometry
#
# AnnotatedAudio = Audio + AnnotatedObject
#
# AudioFeature
#     - Spectrogram
#     - ZeroCrossingRate
#     - MFCC
#     - Indices
#
#
# class Bla:
#     @property
#     def blu(self):
#         ...
#         return algo
#
# bla.blu()
# collection.media(...)
# col.recordings(...)
# col.annotations(..)
# col.apply(...)
#
# audio.features.spectrogram.apply()
# audio.features.spectrogram(hop, win, ... )
#
# AnnotatedSpectrogram = Spectrogram + AnnotatedObject
#
# Audio:AnnotatedAudio.get_spectrogram() -> spec:AnnotatedSpectrogram
#
# spec.grid_cut() -> SpectrogramFragment + AnnotatedObject
#
#
# AudioDataFrame
#     - Lista de Audios
#     - DataFrame de metadatos
#
#     .filter
#     .apply
#
#
#
# col = merge_collections(mcol1, mcol2)
#
#
#
# AudioCollection = AudioDataFrame + DBManager
#
# Collection <-> DBManager <-> Db
# AudioDataFrame <-> DBManager <-> Collection
#
# AudioDataFrame
#     managers
# col.pull.. .
# df = DataFrame(col, query, fields)
# df = DataFrame()
# df.insert()
# daskBag -> result (meta, datatype) -> dataframe
# dataframe -> result (meta, datatype) -> dataframe
# [dict, ..] -> (daskBag) split(Nodos)                            --> gather --> DaskDataFrame (futures)
#                                                                      PandasDatFrame
#                 || au = Audio(dict["media_info"])
#                    spec = au.features.spectrogam(...)
#                    ...
#                    output_nodox
#
# DataFrame(col, query) <-> AudioDataFrame
# Collection(db) <--> DB
# df (media_info.saplerate,..) -> Audio
#
# grid(Audio, twin, thop) -> output.shape == (N, M) -> Audio.mask ~ (i, j)
#
# matches = AnnAudio.get_spec_pieces(query, fields={"metadata.vegetacion":"vegetacion",...}, twin...)
#
# spectrogram = audio.get_spectrogram()
# grid = spectrogram.grid_cut()
# for piece in grid:
#     ...
#
# grid.plot()
#
# cell = grid[i, j]
#
#     results.append(obj = {"piece": piece, "spec": spec_piece, "metadata": metadata})
#
#
# Audio + AnnotatedObject
#     - Spectrogram
#
#
#                  .---- AudioFragment (Audio + Fragment)
#                 /
# -- Fragment --<
#                 \
#                  .---- SpectrogramFragment (Spectrogram + Fragment)
#
#
#              .---- TimeGrid
#             /
# -- Grid --<
#             \
#              .---- TimeFrequencyGrid
#
# -- FragmentGrid = Audio/Spectrogram/... Fragments + Grid
#
#
#     grid = TimeFrequencyGrid(time_width=1, freq_height=10000, time_hop=10, freq_hop=1000)
#     spec_grid = grid.cut(spectrogram, lazy=True)
#     audio_grid = grid.cut(audio, lazy=True)
#
#     spec = audio.features.spectrogram(n_fft=1024, lazy=True)
#     grid = spec.grid_cut(lazy=True, grid=grid)
#
#     grid[i, j]
#
#     spec_grid = SpecGrid(Audio, twin, fwin... )
#
#                  .---- Spectrogram --> self.cut() -> [Spectrograms, ..]
#                 /
# -- Features --<
#                 \
#                  .---- ...
#
# -- Annotation --
#
#
#
#     ------ Audio / Features / Annotations
#   /
# --|
#   \
#    ------- AudioDataFrame (has DBManager) / Soundscapes / ...
#
# -- DBManager
#
# -- Collection (has DBManager)
