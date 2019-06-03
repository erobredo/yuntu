import hashlib
import json

def hashDict(dDict):
    #Not reliable for non str keys (tuples, etc)
    m = hashlib.md5(json.dumps(dDict, sort_keys=True, ensure_ascii=True).encode('utf-8'))

    return m.hexdigest()