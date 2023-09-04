from sqlalchemy import create_engine, MetaData

engine = create_engine("mysql+pymysql://uppnsvbhgvadzw9j:waRA1eijCZD4L17TUuMY@blxtzqumgleiuuggukrz-mysql.services.clever-cloud.com:3306/blxtzqumgleiuuggukrz")

# mysql+pymysql://uppnsvbhgvadzw9j:waRA1eijCZD4L17TUuMY@blxtzqumgleiuuggukrz-mysql.services.clever-cloud.com:3306/blxtzqumgleiuuggukrz
# mysql+pymysql://george:admin@localhost:3306/bills_db

meta = MetaData()

conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")