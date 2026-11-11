"""Converte src/treinar.py em um notebook de células numeradas."""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEM = RAIZ / "src" / "treinar.py"
DESTINO = RAIZ / "notebooks" / "classificacao_saude_bertimbau.ipynb"


def markdown_de_comentarios(bloco):
    linhas = []
    for linha in bloco.splitlines():
        if linha.startswith("# "):
            linhas.append(linha[2:])
        elif linha.strip() == "#":
            linhas.append("")
        else:
            linhas.append(linha)
    return "\n".join(linhas).strip() + "\n"


def celulas(texto):
    atual = None
    buffer = []
    saida = []

    def guardar():
        nonlocal buffer
        if atual is None:
            buffer = []
            return
        bloco = "\n".join(buffer).strip("\n")
        if bloco.strip():
            if atual == "markdown":
                saida.append(("markdown", markdown_de_comentarios(bloco)))
            else:
                saida.append(("code", bloco + "\n"))
        buffer = []

    for linha in texto.splitlines():
        if linha.startswith("# %%"):
            guardar()
            atual = "markdown" if linha.strip() == "# %% [markdown]" else "code"
        elif atual is not None:
            buffer.append(linha)
    guardar()
    return saida


def notebook(pares):
    celulas_nb = []
    for tipo, fonte in pares:
        celulas_nb.append(
            {
                "cell_type": tipo,
                "metadata": {},
                "source": fonte.splitlines(keepends=True),
                **(
                    {"outputs": [], "execution_count": None}
                    if tipo == "code"
                    else {}
                ),
            }
        )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": []},
        },
        "cells": celulas_nb,
    }


def main():
    pares = celulas(ORIGEM.read_text(encoding="utf-8"))
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(notebook(pares), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{DESTINO} com {len(pares)} células")


if __name__ == "__main__":
    main()
