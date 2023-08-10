from sqlalchemy import Table, Column
from sqlalchemy.sql.sqltypes import Integer, String
from config.db import meta, engine

currencies = Table("currencies", meta,
                   Column("id", Integer, primary_key=True),
                   Column("name", String(25), nullable=False))

meta.create_all(bind=engine, tables=[currencies])