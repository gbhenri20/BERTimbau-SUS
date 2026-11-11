"""Gera dados/base_saude.xlsx a partir do MedPT.

Uso:
    MEDPT_PARQUET=/caminho/train.parquet python scripts/criar_base_excel.py

Sem a variável, baixa AKCIT/MedPT pela Hugging Face.
O arquivo gerado é o que src/treinar.py e app/demo.py leem.
"""

import os
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "dados" / "base_saude.xlsx"


def limpar_texto(texto):
    texto = str(texto).replace("\xa0", " ")
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", texto)
    texto = re.sub(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def carregar_bruto():
    parquet = os.environ.get("MEDPT_PARQUET", "")
    if parquet and Path(parquet).exists():
        return pd.read_parquet(parquet, columns=["id", "question", "question_type"])
    from datasets import load_dataset

    conjunto = load_dataset("AKCIT/MedPT", split="train")
    return conjunto.select_columns(["id", "question", "question_type"]).to_pandas()


def montar(bruto):
    df = bruto.copy()
    df["pergunta"] = df["question"].map(limpar_texto)
    df["intencao"] = df["question_type"].fillna("").astype(str).str.strip()
    df["chave"] = df["pergunta"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    df = df[(df["chave"].str.len() >= 15) & (df["intencao"] != "")].copy()
    varios = df.groupby("chave")["intencao"].nunique()
    conflito = set(varios[varios > 1].index)
    df = df[~df["chave"].isin(conflito)].drop_duplicates("chave")
    return df[["id", "pergunta", "intencao"]].reset_index(drop=True)


def main():
    df = montar(carregar_bruto())
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    origem = pd.DataFrame(
        [
            ["nome", "Base de perguntas de saúde (extrato MedPT)"],
            ["arquivo", "dados/base_saude.xlsx"],
            ["aba_consumida", "perguntas"],
            ["fonte", "MedPT (AKCIT), Hugging Face"],
            ["url", "https://huggingface.co/datasets/AKCIT/MedPT"],
            ["artigo", "Farber et al., LREC 2026, DOI 10.63317/5mrjjtbsf6rv"],
            ["registros_na_fonte", "384095"],
            ["registros_nesta_planilha", str(len(df))],
            ["criterio", "Perguntas únicas do MedPT. Nenhuma linha foi inventada."],
            ["colunas", "id; pergunta; intencao"],
        ],
        columns=["campo", "valor"],
    )
    categorias = df["intencao"].value_counts().rename_axis("intencao").reset_index(name="quantidade")
    with pd.ExcelWriter(DESTINO, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="perguntas", index=False)
        origem.to_excel(writer, sheet_name="origem", index=False)
        categorias.to_excel(writer, sheet_name="categorias", index=False)
    print(f"{DESTINO} com {len(df)} linhas")


if __name__ == "__main__":
    main()
