"""Build features table (one row per credit application).

Each file in src/credit_risk_scoring/sql/ produces aggregates per client (SK_ID_CURR).
They are joined with the application; a client without history in a table keeps missing values,
and their counters are set to 0.

Usage:
    uv run python -m credit_risk_scoring.features
"""
from pathlib import Path

import duckdb
import pandas as pd

from credit_risk_scoring.config import DATA_PROCESSED, DUCKDB_PATH, ID

SQL_DIR = Path(__file__).parent / "sql"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"

APPLICATION_SQL = "01_application.sql"
HISTORY_SQL = [
    "02_bureau.sql",
    "03_previous_application.sql",
    "04_installments.sql",
    "05_credit_card.sql",
    "06_pos_cash.sql",
]
# Counter prefixes: with no history, client has 0 loans (not an unknown value)
COUNT_PREFIXES = ("BUREAU_N_", "PREV_N_", "INST_N_", "CC_N_", "POS_N_")


def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def build_query(con: duckdb.DuckDBPyConnection) -> str:
    """Creates a view per SQL file and returns the query that joins them."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW f_application AS {read_sql(APPLICATION_SQL)}")
    joins, count_cols = [], []
    for name in HISTORY_SQL:
        view = "f_" + name.removesuffix(".sql").split("_", 1)[1]
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS {read_sql(name)}")
        joins.append(f"LEFT JOIN {view} USING ({ID})")
        cols = [r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()]
        count_cols += [c for c in cols if c.startswith(COUNT_PREFIXES)]

    replace = ", ".join(f"COALESCE({c}, 0) AS {c}" for c in count_cols)
    return f"SELECT * REPLACE ({replace}) FROM f_application\n" + "\n".join(joins)


def build_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(build_query(con)).df()


def main() -> None:
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = build_features(con)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"{df.shape[0]:,} applications x {df.shape[1]} columns: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
