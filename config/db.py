from sqlalchemy import create_engine, MetaData

engine = create_engine("mysql+pymysql://george:admin@localhost:3306/bills_db")

meta = MetaData()

conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")