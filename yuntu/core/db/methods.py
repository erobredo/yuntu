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
                if "groups" in key:
                    subStatement += "json_extract(groups,'$."+key.replace("groups.","")+"')"
                elif "label" in key:
                    subStatement += "json_extract(label,'$."+key.replace("label.","")+"')"
                elif "metadata" in key:
                    subStatement += "json_extract(metadata,'$."+key.replace("metadata.","")+"')"
                elif "media_info" in key:
                    subStatement += "json_extract(media_info,'$."+key.replace("media_info.","")+"')"
                elif "parse_seq" in key:
                    if isDict:
                        if "$size" in condition:
                            subStatement += "json_array_length(parse_seq)"
                        else:
                            raise ValueError("Not implemented")
                    else:
                        subStatement += "json_extract(parse_seq,'$"+key.replace("parse_seq","")+"')"
                elif "verts" in key:
                    if isDict:
                        if "$size" in condition:
                            subStatement += "json_array_length(verts)"
                        else:
                            raise ValueError("Not implemented")
                    else:
                        subStatement += "json_extract(verts,'$"+key.replace("verts","")+"')"
                else:
                    subStatement += key

                if isDict:
                    if "$ne" in condition:
                        if isinstance(condition["$ne"],str):
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
                            if len(condition["$in"]) == 1:
                                if isinstance(condition["$in"][0],str):
                                    subStatement += " = '"+condition["$in"][0]+"'"
                                else:
                                    subStatement += " = "+str(condition["$in"][0])
                            else:
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

def lDbNoRowFactory(cursor,row):
    return row

def lDbRowFactory(cursor,row):
    d = {}
    for idx, col in enumerate(cursor.description):
        if col[0] in ["metadata","media_info","parse_seq","source","conf","groups","label","verts"]:
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
        CREATE TABLE IF NOT EXISTS datastores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT NOT NULL,
            type TEXT NOT NULL,
            conf JSON NOT NULL,
            metadata JSON NOT NULL,
            UNIQUE(hash)
        )
        """)
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
            UNIQUE(md5),
            FOREIGN KEY(orid) REFERENCES original(id)
        )
        """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notetype TEXT NOT NULL,
            orid INTEGER,
            file_start DOUBLE NOT NULL,
            file_end DOUBLE NOT NULL,
            start_time DOUBLE NOT NULL,
            end_time DOUBLE NOT NULL,
            duration DOUBLE NOT NULL,
            max_freq DOUBLE,
            min_freq DOUBLE,
            verts JSON,
            wkt TEXT,
            label JSON NOT NULL,
            groups JSON NOT NULL,
            metadata JSON NOT NULL,
            FOREIGN KEY(orid) REFERENCES original(id),
            CHECK (notetype in ('weak','interval','bbox','linestring',
            'multilinestring','polygon','multipolygon'))
        )
        """)
    cnn.commit()
    cnn.row_factory = lDbRowFactory

    db.connection = cnn

    return True

def lDbCreateFromDump(db,dump):
    connPath = db.connPath
    cnn = sqlite3.connect(connPath)
    cursor = cnn.cursor()

    statements = []
    if not isinstance(dump,list):
        sql = open(dump,'r')
        strMem = ""
        for line in sql:
            strMem = strMem+line
            if strMem[-2:-1] == ";":
                statements.append(strMem)
                strMem = ""

        sql.close()
    else:
        statements = dump

    for st in statements:
        cursor.execute(st)

    cnn.commit()
    cnn.row_factory = lDbRowFactory
    db.connection = cnn

    return True



def lDbParseSeqConcat(row,parseSeq,parsers=None,parsersDir=None):
    orid = row["orid"]
    metadata = row["metadata"]
    metadata = dbUtils.sequentialTransform(metadata,parseSeq,parsers,parsersDir)
    parse_seq = row["parse_seq"] + parseSeq

    return orid,metadata,parse_seq

def lDbParseSeqOverwrite(row,cursor,parseSeq,parsers=None,parsersDir=None):
    orid = row["orid"]
    oriItem = cursor.execute('SELECT * FROM {tn} WHERE id = {orid}'.format(tn="original",orid=orid)).fetchone()
    metadata = oriItem["metadata"]
    metadata = dbUtils.sequentialTransform(metadata,parseSeq,parsers,parsersDir)
    parse_seq = parseSeq

    return orid,metadata,parse_seq

def lDbUpdateParseSeq(db,parseSeq,orid=None,whereSt=None,operation="append"):
    parsers = db.parsers
    parsersDir = db.parsersDir
    cnn = db.connection
    cursor = cnn.cursor()

    if whereSt is not None:
        matches = cursor.execute('SELECT * FROM {tn} WHERE {whereSt}'.format(tn="parsed",whereSt=whereSt)).fetchall()
    elif orid is not None:
        matches = cursor.execute('SELECT * FROM {tn} WHERE orid = {orid}'.format(tn="parsed",orid=orid)).fetchall()
    else:
        matches = cursor.execute('SELECT * FROM {tn}'.format(tn="parsed",whereSt=whereSt)).fetchall()

    if operation == "append":
        for row in matches:
            orid,metadata,parse_seq = lDbParseSeqConcat(row,parseSeq,parsers=parsers,parsersDir=parsersDir)
            cursor.execute('UPDATE {tn} SET parse_seq = ?, metadata = ? WHERE orid = {orid}'.format(tn="parsed",orid=orid),(json.dumps(parse_seq),json.dumps(metadata)))

    elif operation == "overwrite":
        for row in matches:
            orid,metadata,parse_seq = lDbParseSeqOverwrite(row,cursor,parseSeq,parsers=parsers,parsersDir=parsersDir)
            cursor.execute('UPDATE {tn} SET parse_seq = ?, metadata = ? WHERE orid = {orid}'.format(tn="parsed",orid=orid),(json.dumps(parse_seq),json.dumps(metadata)))
    else:
        raise ValueError("Operation "+str(operation)+" not found.")

    cnn.commit()

    return True

def lDbTimedInsert(db,dataArray,parseSeq,timeField,tzField,format='%d-%m-%Y %H:%M:%S'):
    return db.insert(dataArray,parseSeq,timeConf={"timeField":timeField,"tzField":tzField,"format":format})


def lDbAnnotate(db,dataArray):
    cnn = db.connection
    cursor = cnn.cursor()

    for dataObj in dataArray:
        try:
            if dataObj["notetype"] in ['bbox', 'linestring', 'multilinestring',
                                       'polygon', 'multipolygon']:
                if dataObj["max_freq"] is None or dataObj["max_freq"] is None \
                  or dataObj["wkt"] is None:
                    raise Exception("'max_freq','min_freq' and 'wkt' shold be \
                                       defined for this type of annotation")
            cursor.execute("""
                INSERT INTO annotations (notetype,orid,file_start,file_end,
                start_time,end_time,duration,max_freq,min_freq,verts,wkt,label,groups,metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (dataObj["notetype"],
                      dataObj["orid"],
                      dataObj["file_start"],
                      dataObj["file_end"],
                      dataObj["start_time"],
                      dataObj["end_time"],
                      dataObj["duration"],
                      dataObj["max_freq"],
                      dataObj["min_freq"],
                      json.dumps(dataObj["verts"]),
                      dataObj["wkt"],
                      json.dumps(dataObj["label"]),
                      json.dumps(dataObj["groups"]),
                      json.dumps(dataObj["metadata"])))
        except:
           print("Error inserting annotation :",dataObj)


    cnn.commit()

    return True

def lDbInsert(db,dataArray,parseSeq=[],timeConf=None):


    parsers = db.parsers
    parsersDir = db.parsersDir
    cnn = db.connection
    cursor = cnn.cursor()

    for dataObj in dataArray:
        try:
            dsconf = dataObj["datastore"]
            source = dataObj["source"]
            rawMetadata = dataObj["metadata"]

            dsmatches = cursor.execute('SELECT * FROM {tn} WHERE hash = {hash}'.format(tn="datastores",hash="'"+dsconf["hash"]+"'")).fetchall()

            if len(dsmatches) > 0:
                source["source_id"] = dsmatches[0]["id"]
            else:
                cursor.execute("""
                    INSERT INTO datastores (hash,type,conf,metadata)
                        VALUES (?,?,?,?)
                    """, (dsconf["hash"],dsconf["type"],json.dumps(dsconf["conf"]),json.dumps(dsconf["metadata"])))

                source["source_id"] = cursor.lastrowid


            cursor.execute("""
                INSERT INTO original (source,metadata)
                    VALUES (?,?)
                """, (json.dumps(source),json.dumps(rawMetadata)))

            orid = cursor.lastrowid
            metadata = dataObj["metadata"]
            metadata = dbUtils.sequentialTransform(metadata,parseSeq,parsers,parsersDir)




            path = metadata["path"]
            timeexp = metadata["timeexp"]

            md5 = None
            if "md5" in metadata:
                md5 = metadata["md5"]

            media_info,md5 = dbUtils.describeAudio(path,timeexp,md5)
            media_info["md5"] = md5

            if timeConf is not None:
                timeField = timeConf["timeField"]
                tzField = timeConf["tzField"]
                format = timeConf["format"]
                metadata = dbUtils.getTimeFields(metadata,media_info,timeField,tzField,format)

            cursor.execute("""
                INSERT OR IGNORE INTO parsed (orid,md5,path,original_path,parse_seq,media_info,metadata)
                    VALUES (?,?,?,?,?,?,?)
                """, (orid,md5,path,path,json.dumps(parseSeq),json.dumps(media_info),json.dumps(metadata)))
        except:
           print("Error inserting metadata :",dataObj)



    cnn.commit()

    return True

def lDbFind(db,orid=None,query=None, table="parsed"):
    wStatement = None
    if query is not None:
        wStatement = lDbParseQuery(query)

    return lDbSelect(db,orid,wStatement,table=table)

def lDbSelect(db,orid=None,whereSt=None,freeSt=None,table="parsed"):
    cnn = db.connection
    cursor = cnn.cursor()

    if freeSt is not None:
        matches = cursor.execute('SELECT {freeSt}'.format(freeSt=freeSt)).fetchall()
    else:
        if whereSt is not None:
            matches = cursor.execute('SELECT * FROM {tn} WHERE {whereSt}'.format(tn=table,whereSt=whereSt)).fetchall()
        elif orid is not None:
            matches = cursor.execute('SELECT * FROM {tn} WHERE orid = {orid}'.format(tn=table,orid=orid)).fetchall()
        else:
            matches = cursor.execute('SELECT * FROM {tn}'.format(tn=table)).fetchall()

    return matches

def lDbCount(db,where=None,query=None,groupby=None,table="parsed"):

    whereSt = None
    if where is not None:
        whereSt = where
    elif query is not None:
        whereSt = lDbParseQuery(query)

    cnn = db.connection
    cursor = cnn.cursor()

    if whereSt is not None:
        dcount = cursor.execute('SELECT count(*) as count FROM {tn} WHERE {whereSt}'.format(tn=table,whereSt=whereSt)).fetchone()
    else:
        dcount = cursor.execute('SELECT count(*) as count FROM {tn}'.format(tn=table,whereSt=whereSt)).fetchone()

    return dcount

def lDbRemove(db,orid=None,where=None,query=None,table="parsed"):

    if where is not None:
        matches = lDbSelect(db,orid,where,table=table)
    else:
        matches = lDbFind(db,orid,query,table=table)

    cnn = db.connection
    cursor = cnn.cursor()
    for row in matches:
        cursor.execute('DELETE FROM {tn} WHERE orid = {orid}'.format(tn="parsed",orid=row["orid"]))
        cursor.execute('DELETE FROM {tn} WHERE orid = {orid}'.format(tn="original",orid=row["orid"]))

    cnn.commit()
    return matches

def lDbUpdateField(db,field,value,orid=None,query=None,table="parsed"):
    whereSt = lDbParseQuery(query)
    cnn = db.connection
    cursor = cnn.cursor()

    if whereSt is not None:
        cursor.execute('UPDATE {tn} SET {field} = ? WHERE {whereSt}'.format(tn=table,field=field,whereSt=whereSt),(value,))
    elif orid is not None:
        cursor.execute('UPDATE {tn} SET {field} = ? WHERE orid = {orid}'.format(tn=table,field=field,orid=orid),(value,))
    else:
        cursor.execute('UPDATE {tn} SET {field} = ?'.format(tn=table,field=field),(value,))

    cnn.commit()

    return True

def lDbMerge(db1,db2,mergePath,conf1={'parseSeq':[],'operation':'concat'},conf2={'parseSeq':[],'operation':'concat'}):
    pass

def lDbAsStatements(db):
    cnn = db.connection
    cnn.row_factory = lDbNoRowFactory
    statements = [line for line in cnn.iterdump()]
    cnn.row_factory = lDbRowFactory

    return statements

def lDbDump(db,dumpPath,overwrite):
    if os.path.isfile(dumpPath) and not overwrite:
        raise ValueError("File exists but 'overwrite' is False")
    full_dump = os.linesep.join(db.asStatements())
    f = open(dumpPath, 'w')
    f.writelines(full_dump)
    f.close()

    return dumpPath

def lDbTransfer(db,newDirPath,newName=None,overwrite=False):
    if newName is None:
        newName = db.name
    newPath = os.path.join(newDirPath,newName+".sqlite")

    if os.path.isfile(newPath) and not overwrite:
        raise ValueError("File exists but 'overwrite' is False")

    if db.getType() != "RAMDb":
        connPath = db.connPath
        shutil.copyfile(connPath,newPath)
        return newPath
    else:
        print("In memory database, transfering to disk...")
        db.toDisk(newDirPath,newName).close()
        return newPath
