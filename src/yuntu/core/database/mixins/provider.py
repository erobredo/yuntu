"""Mixins for distinct database providers.

This is the place to include methods for specific providers.
"""
from yuntu.core.database.mixins.utils import pg_create_db


class SqliteMixin:
    """Bind SQLite database provider."""

    provider = 'sqlite'


class PostgresqlMixin:
    """Bind PostgreSQL database provider."""

    provider = 'postgres'

    def prepare_db(self, admin_user, admin_password, admin_db):
        pg_create_db(self.config, admin_user, admin_password, admin_db)


class OracleMixin:
    """Bind oracle database provider."""

    provider = 'oracle'


class MysqlMixin:
    """Bind MySQL database provider."""

    provider = 'mysql'
