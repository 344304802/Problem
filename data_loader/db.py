import pandas as pd
import numpy as np


def to_mysql(df, table, cfg) :
    try :
        import mysql.connector
        conn = mysql.connector.connect(
            host=cfg["host"], port=cfg["port"],
            user=cfg["user"], password=cfg["password"],
            database=cfg["database"],
        )
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cols = ", ".join([f"`{c}` FLOAT" if c != "timestamp" else "`timestamp` DATETIME" for c in df.columns])
        cur.execute(f"CREATE TABLE {table} ({cols})")
        placeholders = ", ".join(["%s"] * len(df.columns))
        colnames = ", ".join([f"`{c}`" for c in df.columns])
        data = [tuple(row) for row in df.itertuples(index = False, name = None)]
        cur.executemany(f"INSERT INTO {table} ({colnames}) VALUES ({placeholders})", data)
        conn.commit()
        cur.close()
        conn.close()
        print(f"[INFO] 写入MySQL表 {table} : {len(df)} 行")
    except Exception as e :
        print(f"[WARN] MySQL不可达({e}), 降级parquet缓存")
        df.to_parquet(f"{table}.parquet", index = False)


def from_mysql(table, cfg) :
    try :
        import mysql.connector
        conn = mysql.connector.connect(
            host=cfg["host"], port=cfg["port"],
            user=cfg["user"], password=cfg["password"],
            database=cfg["database"],
        )
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception as e :
        print(f"[WARN] MySQL不可达({e}), 尝试parquet缓存")
        return pd.read_parquet(f"{table}.parquet")