'''Base class for datastores that pull data form mongodb'''
from pymongo import MongoClient
from yuntu.datastore.base import DataBaseDatastore

class MongoDbDatastore(DataBaseDatastore):
  _client = None

  @property
  def client(self):
      if self._client is None:
          self._client = MongoClient(self.db_config["host"])
      return self._client

  @property
  def size(self):
      if self._size is None:
          db = self.client[self.db_config["database"]]
          collection = db[self.db_config["collection"]]
          self._size = collection.count_documents(self.query)
      return self._size

  def iter(self):
      size = self.size

      db = self.client[self.db_config["database"]]
      collection = db[self.db_config["collection"]]

      if self.tqdm is not None:
          with self.tqdm(total=size) as pbar:
              for document in collection.find(self.query, self.db_config["fields"]):
                  pbar.update(1)
                  yield document
      else:
          for document in collection.find(self.query, self.db_config["fields"]):
              yield document

  def get_metadata(self):
      meta = super().get_metadata()
      meta["type"] = "MongoDbDatastore"
      return meta
