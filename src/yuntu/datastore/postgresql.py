
import os
import numpy as np
from psycopg2.extras import RealDictCursor
import psycopg2

from abc import ABC
from yuntu.datastore.base import DataBaseDatastore

class PostgresqlDatastore(DataBaseDatastore, ABC):

    def get_cursor(self, connection):
        return connection.cursor(cursor_factory=RealDictCursor)

    def connect_to_db(self):
        return psycopg2.connect(**self.db_config)

    def iter(self):
        size = self.size

        connection = self.connect_to_db()
        cursor = self.get_cursor(connection)
        query = self.query
        cursor.execute(self.query)

        if self.tqdm is not None:
            with self.tqdm(total=size) as pbar:
                for document in cursor:
                    pbar.update(1)
                    yield document
        else:
            for document in cursor:
                yield document

        cursor.close()
        connection.close()

    def get_metadata(self):
        meta = super().get_metadata()
        meta["type"] = "PostgresqlDatastore"
        return meta

    @staticmethod
    def insert_into_dict(d, keys, value):
        current_dict = d
        for key in keys[:-1]:
            if not key in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[keys[-1]] = value

    @property
    def size(self):
        if self._size is None:
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor()
            query = self.query
            count_query = f'SELECT count(*) FROM ({query}) as q;'
            cursor.execute(count_query)
            size = cursor.fetchone()[0]
            cursor.close()
            connection.close()

            self._size = size

        return self._size

    def get_partitions(self, npartitions):
        chunk_size = int(np.ceil(self.size / npartitions))
        return [{
            'limit': chunk_size,
            'offset': n * chunk_size,
        } for n in range(npartitions)]

    def partitionate(self, npartitions):
        chunk_size = int(np.ceil(self.size / npartitions))

        if chunk_size == 0:
            raise ValueError("Too many partitions!")

        partitions = []
        for n in range(npartitions):
            meta = self.get_metadata()
            del meta["type"]

            limit = chunk_size
            offset = n * chunk_size
            query = f'SELECT * FROM ({self.query}) AS foo ORDER BY id ASC LIMIT {limit} OFFSET {offset}'
            meta["query"] = query

            partitions.append(meta)

        return partitions
