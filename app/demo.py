"""Demonstração local do classificador. O treino continua no notebook do Colab."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

RAIZ = Path(__file__).resolve().parents[1]
PASTA_MODELO = RAIZ / "modelos" / "bertimbau-saude-intencao"
ARQUIVO_EXCEL = RAIZ / "dados" / "base_saude.xlsx"

EXEMPLOS = [
    "Estou precisando agendar uma consulta na unidade básica de saúde.",
    "Estou com febre, dor no corpo e tosse há dois dias. O que pode ser isso?",
    "Posso tomar ibuprofeno junto com o remédio da pressão?",
    "Meu exame de sangue deu glicose alta. Preciso repetir o exame?",
    "Caminhar trinta minutos por dia ajuda a controlar a pressão arterial?",
    "Dor no joelho há uma semana. Devo procurar o clínico geral ou o ortopedista?",
    "Para que serve o hormônio da tireoide no organismo?",
]


@st.cache_data
def ler_planilha():
    if not ARQUIVO_EXCEL.exists():
        return None
    return pd.read_excel(ARQUIVO_EXCEL, sheet_name="perguntas")


def nome_classe(config, indice):
    mapa = config.id2label
    if indice in mapa:
        return mapa[indice]
    return mapa[str(indice)]


@st.cache_resource
def carregar_modelo():
    tokenizador = BertTokenizer.from_pretrained(PASTA_MODELO)
    modelo = BertForSequenceClassification.from_pretrained(PASTA_MODELO)
    modelo.eval()
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)
    with (PASTA_MODELO / "rotulos.json").open(encoding="utf-8") as arquivo:
        registro = json.load(arquivo)
    return tokenizador, modelo, dispositivo, registro


def classificar(texto, tokenizador, modelo, dispositivo, max_length):
    codigos = tokenizador(
        texto.strip(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    codigos = {chave: valor.to(dispositivo) for chave, valor in codigos.items()}
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(**codigos).logits, dim=-1).squeeze(0).cpu()
    indice = int(probabilidades.argmax())
    nomes = [nome_classe(modelo.config, i) for i in range(len(probabilidades))]
    distribuicao = {nome: float(probabilidades[i]) for i, nome in enumerate(nomes)}
    return nomes[indice], float(probabilidades[indice]), distribuicao


def main():
    st.set_page_config(
        page_title="Intenção em textos de saúde",
        page_icon="🩺",
        layout="wide",
    )
    st.title("Organização de mensagens da atenção primária")
    st.caption(
        "Fine-tuning do BERTimbau · Aprendizado de Máquina · DSM · "
        "Gabriel Henrique Santana da Silva e Luiz Felipe dos Santos"
    )
    st.warning(
        "Esta demonstração não faz diagnóstico, não indica medicamento e não "
        "substitui a escuta de um profissional de saúde. Ela só estima a "
        "intenção do texto entre as categorias da planilha de treino."
    )

    planilha = ler_planilha()
    if planilha is None:
        st.error("Não encontrei dados/base_saude.xlsx. O algoritmo de treino lê essa planilha.")
    else:
        st.subheader("Base lida da planilha")
        st.caption(f"{ARQUIVO_EXCEL.name} · aba perguntas · {len(planilha):,} linhas".replace(",", "."))
        contagem = planilha["intencao"].value_counts()
        st.dataframe(
            contagem.rename_axis("intencao").reset_index(name="quantidade"),
            width="stretch",
            hide_index=True,
        )
        st.dataframe(planilha.head(8), width="stretch", hide_index=True)

    pesos = PASTA_MODELO / "model.safetensors"
    if not pesos.exists():
        st.error(
            "Os pesos treinados ainda não estão em modelos/bertimbau-saude-intencao/model.safetensors. "
            "Execute o notebook no Google Colab ou `python src/treinar.py` e abra esta página de novo."
        )
        st.stop()

    tokenizador, modelo, dispositivo, registro = carregar_modelo()
    max_length = int(registro.get("max_length", 96))

    if "historico" not in st.session_state:
        st.session_state.historico = []

    with st.sidebar:
        st.header("Modelo")
        st.write("Base pré-treinada")
        st.code(registro.get("modelo_base", "neuralmind/bert-base-portuguese-cased"), language=None)
        st.write("Categorias")
        for nome in registro.get("classes", []):
            st.write(f"- {nome}")
        st.caption(f"Dispositivo desta sessão: {dispositivo}")

    coluna_texto, coluna_resultado = st.columns([1.2, 1])
    with coluna_texto:
        escolha = st.selectbox(
            "Comece por um exemplo ou apague o texto e escreva outro.",
            ["Escrever um texto novo", *EXEMPLOS],
        )
        if escolha != st.session_state.get("escolha_anterior"):
            st.session_state.escolha_anterior = escolha
            st.session_state.rascunho = "" if escolha == "Escrever um texto novo" else escolha
        texto = st.text_area("Texto recebido", key="rascunho", height=180)
        classificar_clique = st.button("Classificar", type="primary", width="stretch")

    with coluna_resultado:
        st.subheader("Resultado")
        if classificar_clique:
            if len(texto.strip()) < 15:
                st.error("Escreva uma frase com pelo menos 15 caracteres.")
            else:
                categoria, confianca, distribuicao = classificar(
                    texto, tokenizador, modelo, dispositivo, max_length
                )
                st.session_state.historico.insert(
                    0,
                    {
                        "Texto": texto.strip(),
                        "Categoria": categoria,
                        "Confiança": f"{confianca:.1%}",
                    },
                )
                st.session_state.ultimo = (categoria, confianca, distribuicao, texto.strip())

        ultimo = st.session_state.get("ultimo")
        if ultimo:
            categoria, confianca, distribuicao, texto_usado = ultimo
            st.markdown(f"**Texto:** {texto_usado}")
            metrica_categoria, metrica_confianca = st.columns(2)
            metrica_categoria.metric("Categoria prevista", categoria)
            metrica_confianca.metric("Confiança", f"{confianca:.1%}")
            grafico = pd.DataFrame(
                {"Probabilidade": list(distribuicao.values())},
                index=list(distribuicao.keys()),
            )
            st.bar_chart(grafico)
        else:
            st.info("O resultado aparece aqui depois da classificação.")

    st.subheader("Histórico desta sessão")
    if st.session_state.historico:
        st.dataframe(pd.DataFrame(st.session_state.historico), width="stretch", hide_index=True)
    else:
        st.caption("Nenhuma classificação ainda. O histórico vale só enquanto esta página estiver aberta.")


if __name__ == "__main__":
    main()
