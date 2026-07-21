"""Testes do ETL (Etapa 2).

O foco é a limpeza dos problemas que o dataset real de fato tem — códigos fora
da documentação, duplicatas e faixas implausíveis.
"""
import pandas as pd
from sqlalchemy import create_engine

from src import config
from src.data import etl


def test_recodifica_education_fora_da_documentacao(amostra):
    """A base real traz 0, 5 e 6 em `education`; a doc só define 1..4."""
    df = amostra.copy()
    df.loc[df.index[:3], "education"] = [0, 5, 6]

    out = etl.transform(df)

    assert set(out["education"].unique()) <= {1, 2, 3, 4}
    assert (out["education"] == etl.EDUCATION_OUTROS).sum() >= 3


def test_recodifica_marriage_fora_da_documentacao(amostra):
    """A base real traz o código 0 em `marriage`; a doc define 1..3."""
    df = amostra.copy()
    df.loc[df.index[:2], "marriage"] = 0

    out = etl.transform(df)

    assert set(out["marriage"].unique()) <= {1, 2, 3}
    assert (out["marriage"] == etl.MARRIAGE_OUTROS).sum() >= 2


def test_remove_duplicatas_exatas(amostra):
    df = pd.concat([amostra, amostra.head(10)], ignore_index=True)
    out = etl.transform(df)
    assert len(out) == len(amostra.drop_duplicates())


def test_descarta_idade_implausivel(amostra):
    df = amostra.copy()
    df.loc[df.index[0], "age"] = 200
    out = etl.transform(df)
    assert out["age"].max() <= 100


def test_preserva_os_codigos_de_pay(amostra):
    """`pay_*` tem -2 e 0 fora da doc, mas eles carregam sinal: não são tocados."""
    df = amostra.copy()
    df.loc[df.index[:5], "pay_1"] = -2
    df.loc[df.index[5:10], "pay_1"] = 0

    out = etl.transform(df)

    assert -2 in set(out["pay_1"].unique())
    assert 0 in set(out["pay_1"].unique())


def test_preserva_fatura_negativa(amostra):
    """Fatura negativa é saldo a favor do cliente, não erro de digitação."""
    df = amostra.copy()
    df.loc[df.index[0], "bill_amt1"] = -1500.0

    out = etl.transform(df)

    assert (out["bill_amt1"] < 0).sum() == 1


def test_alvo_vira_inteiro(amostra):
    out = etl.transform(amostra)
    assert out[config.TARGET].dtype.kind == "i"


def test_load_grava_tabela_e_parquet(tmp_path, monkeypatch, amostra):
    db = tmp_path / "t.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setattr(config, "PROCESSED_PARQUET", tmp_path / "credit.parquet")

    df = etl.transform(amostra)
    etl.load(df)

    engine = create_engine(f"sqlite:///{db}")
    back = pd.read_sql_table(config.TABLE_NAME, engine)
    assert len(back) == len(df)
    assert (tmp_path / "credit.parquet").exists()
