import os
from yuntu.core.audio.base import Audio

def audioIterator(dataArr):
    for row in dataArr:
        yield Audio(row["media_info"],from_config=True)

def audioArray(dataArr):
    return [Audio(row["media_info"],from_config=True) for row in dataArr]

def cleanDirectory(folder):
    if os.path.exists(folder):
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
    return True

