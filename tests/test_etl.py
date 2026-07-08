"""Testes do ETL (Etapa 2). Usam SQLite temporário via DATABASE_URL."""
import pandas as pd
from sqlalchemy import create_engine

from src import config
from src.data import etl


def test_transform_removes_invalid_rows():
    df = pd.DataFrame({
        "customer_id": ["A1", "A2", "A2", "A3"],  # A2 duplicado
        "age": [30, 40, 40, 200],                  # 200 é inválido
        "income": [50000, 60000, 60000, 70000],
        "home_ownership": ["RENT", "OWN", "OWN", "RENT"],
        "loan_intent": ["PERSONAL", "MEDICAL", "MEDICAL", "PERSONAL"],
        "loan_amount": [10000, 5000, 5000, 8000],
        "interest_rate": [10.0, 9.0, 9.0, 11.0],
        "loan_percent_income": [0.2, 0.08, 0.08, 0.1],
        "debt_to_income": [0.3, 0.2, 0.2, 0.25],
        "default": [0, 1, 1, 0],
    })
    out = etl.transform(df)
    # sobra A1 e A2 (dedup); A3 cai pela idade inválida
    assert set(out["customer_id"]) == {"A1", "A2"}
    assert out["default"].dtype.kind == "i"


def test_load_writes_table(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setattr(config, "PROCESSED_PARQUET", tmp_path / "credit.parquet")

    df = etl.transform(etl.extract())
    etl.load(df)

    engine = create_engine(f"sqlite:///{db}")
    back = pd.read_sql_table(config.TABLE_NAME, engine)
    assert len(back) == len(df)
    assert (tmp_path / "credit.parquet").exists()
