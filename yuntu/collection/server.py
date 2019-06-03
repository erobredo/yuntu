import os
import json
from yuntu.collection.utils import metadataArray,signalArray,specArray
from yuntu.core.db.base import RAMDb,embeddedDb
from flask import Flask, Response, request

class colAction(object):
    def __init__(self,methodName,db,mediaDir):
        self.action = None
        self.db = db
        self.methodName = methodName
        self.mediaDir = mediaDir
        if self.methodName == "getMetadata":
            self.action = metadataArray
        elif self.methodName == "getSignals":
            self.action = signalArray
        elif self.methodName == "getSpecs":
            self.action = specArray

        self.response = Response(status=200, headers={})

    def __call__(self, *args):
        orid = request.args.get("orid")
        query = request.args.get("query")

        if orid is not None:
            orid = int(orid)
        if query is not None:
            query = json.loads(query)


        self.db.connect()

        matches = self.db.find(orid,query)
        kwargs = {}

        if self.methodName in ["getSpecs","getSignals"]:
            value = request.args.get("readSr")
            if value is not None:
                value = int(value)
                kwargs["readSr"] = value
            else:
                kwargs["readSr"] = None

        if self.methodName == "getSpecs":
            n_fft = request.args.get("n_fft")
            hop_length = request.args.get("hop_length")

            if n_fft is None or hop_length is None:
                n_fft = 1024
                hop_length = 512

            kwargs["n_fft"] = n_fft
            kwargs["hop_length"] = hop_length

        kwargs["dataArr"] = matches
        kwargs["mediaDir"] = self.mediaDir

        try:
            result = self.action(**kwargs)

            if self.methodName == "getSignals":
                result = [{"id":r["id"],"md5":r["md5"],"signal":r["signal"].tolist()} for r in result]
            elif self.methodName == "getSpecs":
                result = [{"id":r["id"],"md5":r["md5"],"freqs":r["freqs"].tolist(),"spec":r["spec"].tolist()} for r in result]

            self.response = Response(response=json.dumps(result),status=200,headers={})
            return self.response
        except:
            self.response = Response(status=500, headers={})
        self.db.close()
        
        return self.response

class yuntuServer(object):
    def __init__(self,collection,port=9797):
        self.colInfo = collection.info
        if collection.db.connPath == ":memory:":
            dump = collection.db.asStatements()
            self.db = RAMDb(collection.db.name,dump)
        else:
            self.db = embeddedDb(collection.db.name,collection.db.dirPath,overwrite=False)
        self.db.close()
        self.mediaDir = os.path.join(collection.colPath,"media")
        self.port = port
        self.app = Flask(collection.name)

        if self.addCollectionEndpoints():
            print("All set to serve!")
        else:
            raise ValueError("Something went wrong while initializing flask server...")

    def addEndPoint(self,endpoint,endpointName,methodName):
        self.app.add_url_rule(endpoint,endpointName,colAction(methodName,self.db,self.mediaDir))

    def addCollectionEndpoints(self):
        self.addEndPoint("/getMetadata","getMetadata","getMetadata")
        self.addEndPoint("/getSpecs","getSpecs","getSpecs")
        self.addEndPoint("/getSignals","getSignals","getSignals")

        return True

    def run(self,debug=False):
        self.app.run(host='0.0.0.0',port=self.port,debug=debug)

