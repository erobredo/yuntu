'''Geo-spatial related methods.'''
from pony.orm import db_session

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

def build_query_sqlite_with_geom(wkt, method="intersects"):
    if method == "intersects":
        query = F'''lambda recording: raw_sql("st_intersects(geom, GeomFromText('{wkt}', 4326))")'''
        return query
    elif method == "within":
        query = F'''lambda recording: raw_sql("st_within(geom, GeomFromText('{wkt}', 4326))")'''
        return query
    elif method == "touches":
        query = F'''lambda recording: raw_sql("st_within(geom, GeomFromText('{wkt}', 4326))")'''
        return query
    raise NotImplementedError(F"Method {method} not implemented. Use 'raw_sql'.")

def create_spatial_structure(db, provider):
    if provider == "sqlite":
        create_sqlite_spatial_structure(db)
    else:
        raise NotImplementedError("Only sqlite databases support spatial indexing for now")

def parse_geometry(db, entities, provider):
    if provider == "sqlite":
        return parse_sqlite_geometry(db, entities)
    else:
        raise NotImplementedError("Only sqlite databases support spatial indexing for now")

def build_query_with_geom(provider, wkt, method="intersects"):
    if provider == "sqlite":
        return build_query_sqlite_with_geom(wkt, method)
    else:
        raise NotImplementedError("Only sqlite databases support spatial query for now")
