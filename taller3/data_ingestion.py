import os
import pandas as pd
from sqlalchemy import create_engine, text

# Database connection URL
DB_URI = os.getenv("DATA_DB_URI", "mysql+pymysql://mlflow_user:mlflow_pass@localhost:3306/penguins_db")

# Table and schema names
RAW_TABLE = "penguins_raw"

# Connect to the database
engine = create_engine(DB_URI)

# 1. Drop existing table if it exists (to start fresh)
with engine.begin() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {RAW_TABLE}"))
    # (If using schemas, ensure schema exists, but we'll use default database)

# 2. Load the penguins dataset
from palmerpenguins import load_penguins
df = load_penguins()  # load as pandas DataFrame
print(f"Loaded penguins dataset with {len(df)} rows.")

# 3. Save the raw data to MySQL
df.to_sql(RAW_TABLE, engine, if_exists="replace", index=False)
print(f"Inserted raw data into table '{RAW_TABLE}'.")

