from sqlalchemy import Table, Column, ForeignKey
from sqlalchemy.sql.sqltypes import Integer, String, Float, Date
from config.db import meta, engine

detections = Table("detections", meta,
    Column("id", Integer, primary_key=True),
    Column('user_id', Integer, nullable=False),
    Column('detection_date', Date, nullable=False),
    Column('currency_id', Integer, ForeignKey("currencies.id"), nullable=False),
    Column('classification', String(30), nullable=False),
    Column('percentage', Float, nullable=False),
    Column('image_id', Integer, nullable=False),
)

meta.create_all(bind=engine, tables=[detections])