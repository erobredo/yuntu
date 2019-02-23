import os
import shutil
import sqlite3
import json
import yuntu.core.db.utils as dbUtils


def lDbParseQuery(query):
    fullStatement = None

    if query is not None:
        fullStatement = ""
        qkeys = list(query.keys())
        qlen = len(qkeys)
        for k in range(qlen):
            key = qkeys[k]
            subStatement = ""

            if key in ["$or","$and"]:
                operator = "OR"
                if key == "$and":
                    operator = "AND"
                if qlen > 1:
                    subStatement += "("

                agglen = len(query[key])
                for i in range(agglen):
                    subQuery = query[key][i]
                    if i > 0:
                        subStatement += " "+operator+" "

                    subStatement += lDbParseQuery(subQuery)

                if qlen > 1:
                    subStatement += ")"
            elif key == "$not":
                subQuery = query[key]
                subStatement += "NOT ("
                subStatement += lDbParseQuery(subQuery)
                subStatement += ")"
            else:
                condition = query[key]
                isDict = False
                if isinstance(condition,dict):
                    isDict = True
                print(key)
                if "metadata" in key:
                    subStatement += "json_extract(metadata,'$."+key.replace("metadata.","")+"')"
                elif "media_info" in key:
                    subStatement += "json_extract(media_info,'$."+key.replace("media_info.","")+"')"
                elif "parse_seq" in key:
                    print(key)
                    if isDict:
                        if "$size" in condition:
                            subStatement += "json_array_length(parse_seq)"
                        else:
                            raise ValueError("Not implemented")
                    else:
                        subStatement += "json_extract(parse_seq,'$"+key.replace("parse_seq","")+"')" 
                else:
                    subStatement += key

                if isDict:
                    if "$ne" in condition:
                        if isinstance(condition,str):
                            subStatement += " <> "+"'"+condition["$ne"]+"'"
                        else:
                            subStatement += " <> "+str(condition["$ne"])
                    elif "$gt" in condition:
                        subStatement += " > "+str(condition["$gt"])
                    elif "$lt" in condition:
                        subStatement += " < "+str(condition["$lt"])
                    elif "$gte" in condition:
                        subStatement += " >= "+str(condition["$gte"])
                    elif "$lte" in condition:
                        subStatement += " <= "+str(condition["$lte"])
                    elif "$in" in condition:
                        if isinstance(condition["$in"],list):
                            condTuple = tuple(condition["$in"])
                            subStatement += " IN "+str(condTuple)
                        else:
                            raise ValueError("'$in' must be used with a list of values to compare")
                    elif "$size" in condition:
                        subStatement += " = "+str(condition["$size"])
                    else:
                        raise ValueError("Not implemented")
                else:
                    if isinstance(condition,str):
                        subStatement += " = "+"'"+condition+"'"
                    else:
                        subStatement += " = "+str(condition)

            if k > 0:
                fullStatement += " AND "
            fullStatement += subStatement


    return fullStatement



def lDbConnPath(db):
    name = db.name
    dirPath = db.dirPath
    return os.path.join(dirPath,name+".sqlite")

def lDbDrop(db):
    name = db.name
    dirPath = db.dirPath
    connPath = os.path.join(dirPath,name+".sqlite")
    os.remove(connPath)
    return True

def lDbExists(db):
    name = db.name
    dirPath = db.dirPath
    connPath = os.path.join(dirPath,name+".sqlite")
    if os.path.isfile(connPath):
        return True
    else:
        return False

def lDbRowFactory(cursor,row):
    d = {}
    for idx, col in enumerate(cursor.description):
        if col[0] in ["metadata","media_info","parse_seq","source"]:
            d[col[0]] = json.loads(row[idx])
        else:
            d[col[0]] = row[idx]
    return d

def lDbConnect(db):
    connPath = db.connPath
    cnn = sqlite3.connect(connPath)
    cnn.row_factory = lDbRowFactory
    db.connection = cnn

    return True

def lDbClose(db):
    if db.connection is not None:
        db.connection.close()
        db.connection = None

    return True

def lDbCreateStructure(db):
    connPath = db.connPath
    cnn = sqlite3.connect(connPath)
    cursor = cnn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS original (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source JSON NOT NULL,
            metadata JSON NOT NULL
        )
        """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parsed (
            orid INTEGER PRIMARY KEY,
            md5 TEXT NOT NULL,
            path TEXT NOT NULL,
            original_path TEXT NOT NULL,
            parse_seq JSON NOT NULL,
            media_info JSON NOT NULL,
            metadata JSON NOT NULL,
            FOREIGN KEY(orid) REFERENCES original(id)    
        )
        """)
    cnn.commit()
    cnn.row_factory = lDbRowFactory

    db.connection = cnn

    return True

def lDbParseSeqConcat(row,parseSeq):
    orid = row["orid"]
    metadata = row["metadata"]

    for parserDict in parseSeq:
        parserFunction = dbUtils.loadParser(parserDict)
        metadata = parserFunction(metadata)
    
    parse_seq = row["parse_seq"] + parseSeq

    return orid,metadata,parse_seq

def lDbParseSeqOverwrite(row,cursor,parseSeq):
    orid = row["orid"]
    oriItem = cursor.execute('SELECT * FROM {tn} WHERE id = {orid}'.format(tn="original",orid=orid)).fetchone()
    metadata = oriItem["metadata"]

    for parserDict in parseSeq:
        parserFunction = dbUtils.loadParser(parserDict)
        metadata = parserFunction(metadata)
    
    parse_seq = parseSeq

    return orid,metadata,parse_seq

def lDbUpdateParseSeq(db,parseSeq,orid=None,whereSt=None,operation="append"):
    cnn = db.connection
    cursor = cnn.cursor()

    if whereSt is not None:
        matches = cursor.execute('SELECT * FROM {tn} WHERE {whereSt}'.format(tn="parsed",whereSt=whereSt)).fetchall()
    elif orid is not None:
        matches = cursor.execute('SELECT * FROM {tn} WHERE orid = {orid}'.format(tn="parsed",orid=orid)).fetchone()
    else:
        matches = cursor.execute('SELECT * FROM {tn}'.format(tn="parsed",whereSt=whereSt)).fetchall()

    if operation == "append":
        for row in matches:
            orid,metadata,parse_seq = lDbParseSeqConcat(row,parseSeq)
            cursor.execute('UPDATE {tn} SET parse_seq = {parse_seq}, metadata = {metadata} WHERE orid = {orid}'.format(tn="parsed",parse_seq=json.dumps(parse_seq),metadata=json.dumps(metadata),orid=orid))
    
    elif operation == "overwrite":
        for row in matches:
            orid,metadata,parse_seq = lDbParseSeqOverwrite(row,cursor,parseSeq)
            cursor.execute('UPDATE {tn} SET parse_seq = {parse_seq}, metadata = {metadata} WHERE orid = {orid}'.format(tn="parsed",parse_seq=json.dumps(parse_seq),metadata=json.dumps(metadata),orid=orid))
    else:
        raise ValueError("Operation "+str(operation)+" not found.")

    cnn.commit()

    return True


def lDbInsert(db,dataArray,parseSeq=[]):
    cnn = db.connection
    cursor = cnn.cursor()

    for dataObj in dataArray:
        source = dataObj["source"]
        rawMetadata = dataObj["metadata"]
        orid = cursor.execute("""
            INSERT INTO original (source,metadata)
                VALUES (?, ?)
            """, (json.dumps(source),json.dumps(rawMetadata)))


        metadata = row["metadata"]

        for parserDict in parseSeq:
            parserFunction = dbUtils.loadParser(parserDict)
            metadata = parserFunction(metadata)

        path = metadata["path"]
        timeexp = metadata["timeexp"]

        md5 = None
        if "md5" in metadata:
            md5 = metadata["md5"]

        media_info,md5 = dbUtils.describeAudio(path,timeexp,md5)

        cursor.execute("""
            INSERT INTO parsed (orid, md5,path,original_path,parse_seq, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (orid,md5,path,path,json.dumps(parseSeq),json.dumps(media_info),json.dumps(metadata)))

    cnn.commit()

    return True

def lDbFind(db,orid=None,query=None):
    wStatement = methods.lDbParseQuery(query)

    return lDbSelect(db,orid,wStatement)

def lDbSelect(db,orid=None,whereSt=None):
    cnn = db.connection
    cursor = cnn.cursor()

    if whereSt is not None:
        matches = cursor.execute('SELECT * FROM {tn} WHERE {whereSt}'.format(tn="parsed",whereSt=whereSt)).fetchall()
    elif orid is not None:
        matches = cursor.execute('SELECT * FROM {tn} WHERE orid = {orid}'.format(tn="parsed",orid=orid)).fetchone()
    else:
        matches = cursor.execute('SELECT * FROM {tn}'.format(tn="parsed",whereSt=whereSt)).fetchall()

    return matches

def lDbRemove(db,orid=None,where=None,query=None):

    if where is not None:
        matches = lDbSelect(db,orid,where)
    else:
        matches = lDbFind(db,orid,query)

    cnn = db.connection
    cursor = cnn.cursor()
    for row in matches:
        cursor.execute('DELETE FROM {tn} WHERE orid = {orid}'.format(tn="parsed",orid=row["orid"]))
        cursor.execute('DELETE FROM {tn} WHERE orid = {orid}'.format(tn="original",orid=row["orid"]))

    cnn.commit()
    return matches

def lDbUpdateField(db,field,value,orid=None,query=None):
    wStatement = methods.lDbParseQuery(query)
    cnn = db.connection
    cursor = cnn.cursor()

    if whereSt is not None:
        cursor.execute('UPDATE {tn} SET {field} = {value} WHERE {whereSt}'.format(tn="parsed",field=field,value=value,whereSt=wStatement))
    elif orid is not None:
        cursor.execute('UPDATE {tn} SET {field} = {value} WHERE orid = {orid}'.format(tn="parsed",field=field,value=value,orid=orid))
    else:
        cursor.execute('UPDATE {tn} SET {field} = {value}'.format(tn="parsed",field=field,value=value))

    cnn.commit()

    return True

def lDbMerge(db1,db2,mergePath,conf1={'parseSeq':[],'operation':'concat'},conf2={'parseSeq':[],'operation':'concat'}):
    pass

def lDbDump(db,dumpPath,overwrite):
    connPath = db.connPath
    if dumpPath != connPath:
        shutil.copyfile(connPath,dumpPath)
    elif overwrite:
        shutil.copyfile(connPath,dumpPath)
    else:
        raise ValueError("Cannot dump db. File exists in directory but overwrite is False")

    return dumpPath




