"""Mixin utilities."""
from psycopg2 import connect, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, AsIs
from psycopg2.errors import DuplicateDatabase, DuplicateObject, InvalidCatalogName

def pg_drop_db(config, admin_user="postgres", admin_password="postgres",
                     admin_db="postgres"):
    database = config["database"]
    username = config["user"]
    password = config["password"]
    host = config["host"]
    port = config["port"]

    conn = connect(dbname=admin_db, user=admin_user, password=admin_password,
                   host=host, port=port)
    try:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(sql.SQL(
         "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname={}"
        ).format(sql.Literal(database)))
        cursor.execute(sql.SQL(
         "DROP DATABASE {}"
        ).format(sql.Identifier(database)))
        print(f"Database '{database}' droped!")
    except InvalidCatalogName:
        print(f"Database '{database}' not found. Nothing droped.")

    conn.close()


def pg_create_db(config, admin_user="postgres", admin_password="postgres",
                 admin_db="postgres"):
    database = config["database"]
    username = config["user"]
    password = config["password"]
    host = config["host"]
    port = config["port"]

    conn = connect(dbname=admin_db, user=admin_user, password=admin_password,
                   host=host, port=port)
    try:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(sql.SQL(
         "CREATE DATABASE {}"
        ).format(sql.Identifier(database)))
        print(f"Database '{database}' created!")
    except DuplicateDatabase:
        print(f"Database '{database}' exists, continue...")

    try:
        cursor.execute("CREATE USER %s WITH PASSWORD %s", (AsIs(username),
                                                           password,))
        print(f"User '{username}' created!")
    except DuplicateObject:
        print(f"User '{username}' exists, continue...")

    cursor.execute(sql.SQL(
        "GRANT ALL PRIVILEGES ON DATABASE {} TO %s"
    ).format(sql.Identifier(database)), [AsIs(username)])

    conn.close()
