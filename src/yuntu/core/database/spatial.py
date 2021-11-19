'''Geo-spatial database manager.'''
from pony.orm import db_session
from pony.orm.dbapiprovider import ProgrammingError
from shapely.geometry import Point
from yuntu.core.database.recordings import build_spatial_recording_model
from yuntu.core.database.base import DatabaseManager


SPATIAL_CAPABLE_PROVIDERS = ["sqlite", "postgres"]

@db_session
def create_sqlite_spatial_structure(db):
    con = db.get_connection()
    con.enable_load_extension(True)
    db.execute('''SELECT load_extension('mod_spatialite.so')''')
    db.execute('''SELECT InitSpatialMetaData()''')
    db.execute('''SELECT AddGeometryColumn('recording','geom' , 4326, 'POINT', 2)''')
    db.execute('''SELECT CreateSpatialIndex('recording', 'geom');''')
    db.commit()

@db_session
def create_postgres_spatial_structure(db):
    try:
        db.execute('''SELECT AddGeometryColumn('public','recording','geom' , 4326, 'POINT', 2);''')
        db.execute('''CREATE INDEX recording_geom_idx ON recording USING GIST(geom);''')
        db.commit()
    except ProgrammingError as e:
        print(e)
        raise ValueError("Postgis extension should be installed as superuser independently in order to create a spatial structure in postgres.")

@db_session
def parse_postgres_geometry(db, entities):
    ids = tuple([ent.id for ent in entities])
    if len(ids) > 1:
        sql = F'''UPDATE recording SET geom=st_setsrid(st_makepoint(longitude, latitude), 4326) WHERE id IN {ids}'''
    else:
        id = ids[0]
        sql = F'''UPDATE recording SET geom=st_setsrid(st_makepoint(longitude, latitude), 4326) WHERE id = {id}'''
    db.execute(sql)
    db.commit()
    return entities

@db_session
def parse_sqlite_geometry(db, entities):
    ids = tuple([ent.id for ent in entities])
    if len(ids) > 1:
        sql = F'''UPDATE recording SET geom=MakePoint(longitude, latitude, 4326) WHERE id IN {ids}'''
    else:
        id = ids[0]
        sql = F'''UPDATE recording SET geom=MakePoint(longitude, latitude, 4326) WHERE id = {id}'''
    db.execute(sql)
    db.commit()
    return entities

def build_query_postgres_with_geom(wkt, method="intersects"):
    if method == "intersects":
        query = F'''lambda recording: raw_sql("st_intersects(geom, st_geomfromtext('{wkt}', 4326))")'''
        return query
    elif method == "within":
        query = F'''lambda recording: raw_sql("st_within(geom, st_geomfromtext('{wkt}', 4326))")'''
        return query
    elif method == "touches":
        query = F'''lambda recording: raw_sql("st_touches(geom, st_geomfromtext('{wkt}', 4326))")'''
        return query
    raise NotImplementedError(F"Method {method} not implemented. Use 'raw_sql'.")

def build_query_sqlite_with_geom(wkt, method="intersects"):
    if method == "intersects":
        query = F'''lambda recording: raw_sql("st_intersects(geom, GeomFromText('{wkt}', 4326))")'''
        return query
    elif method == "within":
        query = F'''lambda recording: raw_sql("st_within(geom, GeomFromText('{wkt}', 4326))")'''
        return query
    elif method == "touches":
        query = F'''lambda recording: raw_sql("st_touches(geom, GeomFromText('{wkt}', 4326))")'''
        return query
    raise NotImplementedError(F"Method {method} not implemented. Use 'raw_sql'.")

def create_spatial_structure(db, provider):
    if provider == "sqlite":
        create_sqlite_spatial_structure(db)
    elif provider == "postgres":
        create_postgres_spatial_structure(db)
    else:
        raise NotImplementedError("Only sqlite and postgres databases support spatial indexing for now")

def parse_geometry(db, entities, provider):
    if provider == "sqlite":
        return parse_sqlite_geometry(db, entities)
    elif provider == "postgres":
        return parse_postgres_geometry(db, entities)
    else:
        raise NotImplementedError("Only sqlite and postgres databases support spatial indexing for now")

def build_query_with_geom(provider, wkt, method="intersects"):
    if provider == "sqlite":
        return build_query_sqlite_with_geom(wkt, method)
    elif provider == "postgres":
        return build_query_postgres_with_geom(wkt, method)
    else:
        raise NotImplementedError("Only sqlite and postgres databases support spatial query for now")



class SpatialDatabaseManager(DatabaseManager):
    def init_db(self):
        """Initialize database.

        Will bind with database and generate all tables.
        """
        if self.provider not in SPATIAL_CAPABLE_PROVIDERS:
            prov = self.provider
            raise NotImplementedError(f"Spatial indexing with provider {prov} not implemented")

        self.db.bind(self.provider, **self.config)
        self.db.generate_mapping(create_tables=True)
        self.create_spatial_structure()

    @db_session
    def insert(self, meta_arr, model="recording"):
        """Directly insert new media entries without a datastore."""
        if model == "recording":
            for n, meta in enumerate(meta_arr):
                meta_arr[n]["geometry"] = Point(meta["longitude"], meta["latitude"]).wkt
            entities = super().insert(meta_arr, model)
            self.db.commit()

            return parse_geometry(self.db, entities, provider=self.provider)
        return super().insert(meta_arr, model)

    def create_spatial_structure(self):
        create_spatial_structure(self.db, self.provider)

    def build_spatialized_recording_model(self, recording):
        return build_spatial_recording_model(recording)

    def build_recording_model(self):
        recording = super().build_recording_model()
        return self.build_spatialized_recording_model(recording)
