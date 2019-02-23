from yuntu.core.audio.base import Audio

def audioIterator(dataArr):
    for row in dataArr:
        yield Audio(row["media_info"],from_config=True)

def audioArray(dataArr):
    return [Audio(row["media_info"],from_config=True) for row in dataArr]