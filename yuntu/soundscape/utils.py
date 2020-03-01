from yuntu.core.common.utils import loadMethod,loadMethodFromFile
import itertools

def loadTransform(transformDict):
    if transformDict is not None:
        if "path" in transformDict:
            return loadMethodFromFile(transformDict["path"],transformDict["method"])
        else:
            return loadMethod(transformDict["method"])
    else:
        return None

def getCombinations(listArr):
    return list(itertools.product(*listArr))

def filterExpr(group):
    expr = ""
    cols = list(group.keys())
    val = group[cols[0]]
    if isinstance(val,str):
        val = "'"+val+"'"
    expr = cols[0]+"=="+val
    for i in range(1,len(cols)):
        val = group[cols[i]]
        if isinstance(val,str):
            val = "'"+val+"'"
        expr += " & "+cols[i]+"=="+str(val)

    return ex

    
