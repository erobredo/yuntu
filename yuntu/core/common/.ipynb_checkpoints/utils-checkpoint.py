import os
import json
import hashlib
import shutil
import importlib.util
from importlib import import_module

def loadMethod(methodName):
    mNameArr = methodName.split(".")
    if len(mNameArr) > 1:
        meth = import_module(mNameArr[0])
        for i in range(1,len(mNameArr)):
            last = getattr(meth,mNameArr[i])
            meth = last

        return meth

    else:
        raise ValueError("Only module specified.")

def loadMethodFromFile(path,methodName):
    spec = importlib.util.spec_from_file_location(methodName,path)
    modl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modl)

    mNameArr = methodName.split(".")
    
    if len(mNameArr) > 1:
        meth = getattr(modl,mNameArr[0])
        for i in range(1,len(mNameArr)):
            last = getattr(meth,mNameArr[i])
            meth = last
            
        return meth

    else:
        return getattr(modl,methodName)


def loadJsonFile(path):
    with open(path) as f:
        jdict = json.load(f)
    return jdict

def dumpJsonFile(path,data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
            
    return True

def binaryMD5(path):
    if path is not None:
        if os.path.isfile(path):
            BLOCKSIZE = 65536
            hasher = hashlib.md5()
            with open(path,"rb") as media:
                buf = media.read(BLOCKSIZE)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = media.read(BLOCKSIZE)
            return hasher.hexdigest()
        else:
            return None
    else:
        return None

def cleanDirectory(folder,dirs=True):
    if os.path.exists(folder):
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path) and dirs:
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
    return True