# Fonte do notebook notebooks/classificacao_saude_bertimbau.ipynb.
# As marcas "# %%" separam células. O arquivo também pode ser executado
# diretamente: python src/treinar.py

# %% [markdown]
# # Transformação digital no SUS: classificação de textos de saúde com BERTimbau
#
# **Disciplina:** Aprendizado de Máquina
# **Curso:** Desenvolvimento de Software Multiplataforma (DSM), 5º semestre
# **Integrantes:** Gabriel Henrique Santana da Silva e Luiz Felipe dos Santos
# **Professor:** Bruno (sobrenome a confirmar com a turma)
# **Apresentação:** 11/11/2026
#
# ## Objetivo
#
# Classificar, por fine-tuning, a intenção de textos de saúde escritos em português brasileiro. O modelo de partida é o BERTimbau (`neuralmind/bert-base-portuguese-cased`). O treinamento atualiza os pesos com backpropagation e o otimizador AdamW.
#
# O texto de referência é o TCC de Gabriel A. Coelho, *Transformação digital no SUS: explorando o uso de IA generativa no cuidado clínico primário* (Mackenzie, 2025): <https://dspace.mackenzie.br/handle/10899/41755>. O resumo público descreve um chatbot com RAG e LLMs para a atenção primária. Não descreve o BERTimbau. Este notebook usa o artigo como fundamentação do problema e implementa a técnica pedida pela disciplina. A leitura detalhada está em `docs/analise_artigo.md`.
#
# ## Aviso
#
# O programa organiza textos. Não faz diagnóstico, não sugere dose de medicamento e não substitui a equipe da unidade de saúde.

# %% [markdown]
# ## 1. Instalação e ambiente
#
# No Google Colab, use um ambiente de execução com GPU: *Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU*. Sem GPU, o mesmo código roda na CPU e fica mais lento. A célula instala apenas o que ainda não estiver importável.

# %%
import importlib
import subprocess
import sys

pacotes = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "openpyxl": "openpyxl",
}
faltando = []
for modulo, pacote in pacotes.items():
    try:
        importlib.import_module(modulo)
    except ImportError:
        faltando.append(pacote)

if faltando:
    print("Instalando:", ", ".join(faltando))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltando])
else:
    print("Bibliotecas já disponíveis neste ambiente.")

# %% [markdown]
# ## 2. Importação, semente e hiperparâmetros
#
# A semente fixa deixa a divisão e a amostragem repetíveis. Os valores abaixo cabem em uma GPU T4 do Colab e também permitem uma execução de referência em CPU, com amostra estratificada. Para um experimento maior no Colab, aumente `AMOSTRA_POR_CLASSE` e `NUM_EPOCAS`. `LIMITE_LOTES = None` usa todos os lotes. Só altere esse valor para um teste rápido de instalação.

# %%
import gc
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

SEMENTE = 42
random.seed(SEMENTE)
np.random.seed(SEMENTE)
torch.manual_seed(SEMENTE)

MODELO_NOME = "neuralmind/bert-base-portuguese-cased"
AMOSTRA_POR_CLASSE = int(os.environ.get("AMOSTRA_POR_CLASSE", "400"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
TAMANHO_LOTE = int(os.environ.get("TAMANHO_LOTE", "8"))
NUM_EPOCAS = int(os.environ.get("NUM_EPOCAS", "3"))
TAXA_APRENDIZADO = float(os.environ.get("TAXA_APRENDIZADO", "2e-5"))
DECAIMENTO_PESOS = 0.01
PACIENCIA = 1
LIMITE_LOTES = int(os.environ["LIMITE_LOTES"]) if os.environ.get("LIMITE_LOTES") else None
MINIMO_POR_CLASSE = 20

RAIZ = Path.cwd()
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_MODELO = RAIZ / "modelos" / "bertimbau-saude-intencao"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
PASTA_MODELO.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(max(1, os.cpu_count() or 1))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMENTE)
    DISPOSITIVO = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DISPOSITIVO = torch.device("cpu")
    print("GPU não encontrada. O fine-tuning segue na CPU.")
print("Dispositivo:", DISPOSITIVO)
print("PyTorch:", torch.__version__)
print(
    f"Amostra por classe={AMOSTRA_POR_CLASSE}, max_length={MAX_LENGTH}, "
    f"lote={TAMANHO_LOTE}, épocas={NUM_EPOCAS}, lr={TAXA_APRENDIZADO}"
)

# %% [markdown]
# ## 3. Base de dados
#
# O algoritmo lê a planilha `dados/base_saude.xlsx`, aba `perguntas`. As colunas são `id`, `pergunta` e `intencao`. A aba `origem` registra de onde as linhas vieram. A aba `categorias` resume a contagem.
#
# As linhas não foram inventadas. São as perguntas únicas do **MedPT** (AKCIT), corpus público na Hugging Face: <https://huggingface.co/datasets/AKCIT/MedPT>. O card consultado em 11/11/2026 informa 384.095 pares na fonte. A planilha guarda uma linha por pergunta, depois de tirar duplicata. O artigo da base é Farber et al., LREC 2026, DOI <https://doi.org/10.63317/5mrjjtbsf6rv>. O rótulo `intencao` é o `question_type` do MedPT, que o card atribui ao modelo `gpt-oss-120b`. Não é anotação médica humana. O card não declara licença SPDX.
#
# As categorias sugeridas no enunciado (agendamento, sintomas, medicamentos, exames, orientações, encaminhamentos) não são os rótulos desta planilha. O código usa os rótulos que estão na coluna `intencao`.
#
# No Colab, envie o arquivo para `dados/base_saude.xlsx` ou deixe-o na pasta do projeto. Se a planilha não estiver no diretório, a célula usa o Parquet oficial como reserva.

# %%
ARQUIVO_EXCEL = RAIZ / "dados" / "base_saude.xlsx"
PARQUET_LOCAL = os.environ.get("MEDPT_PARQUET", "")

if ARQUIVO_EXCEL.exists():
    print("Lendo a base Excel:", ARQUIVO_EXCEL.resolve())
    print("Tamanho em bytes:", ARQUIVO_EXCEL.stat().st_size)
    df_bruto = pd.read_excel(ARQUIVO_EXCEL, sheet_name="perguntas")
    df_bruto = df_bruto.rename(columns={"pergunta": "question", "intencao": "question_type"})
elif PARQUET_LOCAL and Path(PARQUET_LOCAL).exists():
    from datasets import load_dataset

    print("Planilha não encontrada. Lendo cópia local do Parquet oficial:", PARQUET_LOCAL)
    conjunto = load_dataset("parquet", data_files=PARQUET_LOCAL, split="train")
    df_bruto = conjunto.select_columns(["id", "question", "question_type"]).to_pandas()
    del conjunto
else:
    from datasets import load_dataset

    print("Planilha não encontrada. Baixando AKCIT/MedPT pela Hugging Face.")
    conjunto = load_dataset("AKCIT/MedPT", split="train")
    df_bruto = conjunto.select_columns(["id", "question", "question_type"]).to_pandas()
    del conjunto

print("Registros carregados:", len(df_bruto))
print("Colunas:", list(df_bruto.columns))
print("\nPrimeiras linhas:")
print(df_bruto.head(5).to_string())
print("\nTipos:")
print(df_bruto.dtypes.to_string())
print("\nValores ausentes:")
print(df_bruto.isna().sum().to_string())
print("\nPerguntas duplicadas (texto exato):", int(df_bruto["question"].duplicated().sum()))
print("\nDistribuição bruta de question_type:")
print(df_bruto["question_type"].value_counts(dropna=False).to_string())

# %% [markdown]
# ## 4. Pré-processamento
#
# A limpeza é propositalmente leve. O BERTimbau foi pré-treinado com texto natural em português, então o código não remove acentos, pontuação nem palavras frequentes. Saem registros sem texto, rótulos vazios, perguntas curtas demais para carregar uma intenção, duplicatas da mesma pergunta e perguntas cuja mesma grafia recebeu intenções diferentes. E-mails, URLs e telefones são apagados para reduzir dado pessoal que tenha escapado à curadoria, sem alterar o conteúdo clínico da frase.

# %%
def limpar_texto(texto):
    texto = str(texto).replace("\xa0", " ")
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", texto)
    texto = re.sub(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def chave_duplicata(texto):
    return re.sub(r"\s+", " ", texto.strip().lower())


df = df_bruto.copy()
df["question"] = df["question"].map(limpar_texto)
df["question_type"] = df["question_type"].fillna("").astype(str).str.strip()
df["chave"] = df["question"].map(chave_duplicata)

antes = len(df)
df = df[(df["chave"].str.len() >= 15) & (df["question_type"] != "")].copy()
removidos_invalidos = antes - len(df)

contagem_rotulos = df.groupby("chave")["question_type"].nunique()
chaves_conflito = set(contagem_rotulos[contagem_rotulos > 1].index)
n_conflitos = int(df["chave"].isin(chaves_conflito).sum())
df = df[~df["chave"].isin(chaves_conflito)].copy()
antes_dup = len(df)
df = df.drop_duplicates(subset="chave", keep="first").copy()
removidos_dup = antes_dup - len(df)

frequencia = df["question_type"].value_counts()
classes_pequenas = sorted(frequencia[frequencia < MINIMO_POR_CLASSE].index)
if classes_pequenas:
    print("Classes removidas por terem menos de", MINIMO_POR_CLASSE, "exemplos:", classes_pequenas)
    df = df[~df["question_type"].isin(classes_pequenas)].copy()

print("Inválidos removidos:", removidos_invalidos)
print("Linhas em perguntas com rótulos conflitantes:", n_conflitos)
print("Duplicatas removidas:", removidos_dup)
print("Registros após a limpeza:", len(df))
print("\nCategorias usadas:")
print(df["question_type"].value_counts().to_string())

distribuicao_completa = df["question_type"].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=distribuicao_completa.values, y=distribuicao_completa.index, ax=ax, color="#0f6e6e")
ax.set_title("MedPT após a limpeza: perguntas por intenção")
ax.set_xlabel("Quantidade de perguntas")
ax.set_ylabel("Intenção (question_type)")
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "distribuicao_categorias.png", dpi=140)
plt.close(fig)
print("Gráfico salvo em resultados/distribuicao_categorias.png")

# %% [markdown]
# ## 5. Amostra estratificada
#
# Treinar com as centenas de milhares de linhas é possível no Colab, mas demora. A execução de referência usa até `AMOSTRA_POR_CLASSE` perguntas de cada intenção, sorteadas com a semente fixa. A divisão seguinte continua estratificada. A amostra não é um conjunto sintético: são linhas do MedPT.

# %%
partes = []
for categoria, grupo in df.groupby("question_type"):
    n = min(AMOSTRA_POR_CLASSE, len(grupo))
    partes.append(grupo.sample(n=n, random_state=SEMENTE))
amostra = pd.concat(partes, ignore_index=True)
amostra = amostra.sample(frac=1, random_state=SEMENTE).reset_index(drop=True)

print("Tamanho da amostra:", len(amostra))
print(amostra["question_type"].value_counts().sort_index().to_string())

distribuicao_amostra = amostra["question_type"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=distribuicao_amostra.index, y=distribuicao_amostra.values, ax=ax, color="#b45309")
ax.set_title("Amostra usada no fine-tuning")
ax.set_xlabel("Intenção")
ax.set_ylabel("Perguntas")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "distribuicao_amostra.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## 6. Codificação dos rótulos e divisão 70% / 15% / 15%
#
# O codificador é ajustado na amostra inteira apenas para conhecer os nomes das classes. Isso não mostra ao modelo o texto do conjunto de teste. O primeiro corte separa 70% para treino e 30% para um bloco temporário. O bloco temporário é dividido ao meio, o que produz 15% de validação e 15% de teste. A validação decide o early stopping. O teste fica intocado até a avaliação final.

# %%
rotulador = LabelEncoder()
rotulador.fit(amostra["question_type"])
classes = list(rotulador.classes_)
id2label = {i: nome for i, nome in enumerate(classes)}
label2id = {nome: i for i, nome in id2label.items()}
y = rotulador.transform(amostra["question_type"])

textos = amostra["question"].tolist()
indices = np.arange(len(amostra))

idx_treino, idx_temp, y_treino, y_temp = train_test_split(
    indices, y, test_size=0.30, random_state=SEMENTE, stratify=y
)
idx_val, idx_teste, y_val, y_teste = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=SEMENTE, stratify=y_temp
)

print("Classes, em ordem de índice:", classes)
print("Treino:", len(idx_treino), "| Validação:", len(idx_val), "| Teste:", len(idx_teste))
for nome, fatia in ("treino", y_treino), ("validação", y_val), ("teste", y_teste):
    valores, quantidades = np.unique(fatia, return_counts=True)
    print(nome, {classes[int(i)]: int(q) for i, q in zip(valores, quantidades)})

conjunto_de = {
    "treino": set(idx_treino.tolist()),
    "validacao": set(idx_val.tolist()),
    "teste": set(idx_teste.tolist()),
}
assert conjunto_de["treino"].isdisjoint(conjunto_de["validacao"])
assert conjunto_de["treino"].isdisjoint(conjunto_de["teste"])
assert conjunto_de["validacao"].isdisjoint(conjunto_de["teste"])

# %% [markdown]
# ## 7. Tokenização com o tokenizer oficial do BERTimbau
#
# `input_ids` são os índices das subpalavras. `attention_mask` marca com 1 os tokens reais e com 0 o preenchimento. O corte em `MAX_LENGTH` existe porque o Transformer processa sequências de tamanho fixo neste treinamento. A célula mede, em até mil textos da amostra, quantos passam desse limite.

# %%
tokenizador = BertTokenizer.from_pretrained(MODELO_NOME)
print("Vocabulário:", tokenizador.vocab_size)
print("do_lower_case:", tokenizador.do_lower_case)

amostra_tamanho = amostra["question"].head(1000).tolist()
tamanhos_tokens = [len(tokenizador.encode(texto, add_special_tokens=True)) for texto in amostra_tamanho]
percentis = np.percentile(tamanhos_tokens, [50, 75, 90, 95, 99])
acima = sum(tamanho > MAX_LENGTH for tamanho in tamanhos_tokens)
print("Percentis do número de tokens (p50, p75, p90, p95, p99):", [round(float(p), 1) for p in percentis])
print(f"Textos medidos que excedem {MAX_LENGTH} tokens: {acima} de {len(tamanhos_tokens)}")


def tokenizar(lista_de_textos):
    return tokenizador(
        lista_de_textos,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


codigos_treino = tokenizar([textos[i] for i in idx_treino])
codigos_val = tokenizar([textos[i] for i in idx_val])
codigos_teste = tokenizar([textos[i] for i in idx_teste])

exemplo_tokens = tokenizador.convert_ids_to_tokens(codigos_treino["input_ids"][0][:24].tolist())
print("Exemplo de input_ids:", codigos_treino["input_ids"][0][:24].tolist())
print("Exemplo de attention_mask:", codigos_treino["attention_mask"][0][:24].tolist())
print("Tokens correspondentes:", exemplo_tokens)

# %% [markdown]
# ## 8. Conjuntos que o PyTorch consegue ler em lotes

# %%
class ConjuntoTextos(Dataset):
    def __init__(self, codigos, rotulos):
        self.input_ids = codigos["input_ids"]
        self.attention_mask = codigos["attention_mask"]
        self.rotulos = torch.tensor(rotulos, dtype=torch.long)

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, indice):
        return {
            "input_ids": self.input_ids[indice],
            "attention_mask": self.attention_mask[indice],
            "labels": self.rotulos[indice],
        }


gerador_lotes = torch.Generator()
gerador_lotes.manual_seed(SEMENTE)

carregador_treino = DataLoader(
    ConjuntoTextos(codigos_treino, y_treino),
    batch_size=TAMANHO_LOTE,
    shuffle=True,
    generator=gerador_lotes,
)
carregador_val = DataLoader(
    ConjuntoTextos(codigos_val, y_val),
    batch_size=TAMANHO_LOTE,
    shuffle=False,
)
carregador_teste = DataLoader(
    ConjuntoTextos(codigos_teste, y_teste),
    batch_size=TAMANHO_LOTE,
    shuffle=False,
)
print("Lotes de treino:", len(carregador_treino))

# %% [markdown]
# ## 9. BERTimbau e a camada nova de classificação
#
# `BertForSequenceClassification` carrega os pesos pré-treinados do BERTimbau e acrescenta uma camada linear com uma saída por categoria. Essa camada nasce aleatória. O aviso da biblioteca sobre pesos novos é esperado. Todos os parâmetros ficam abertos para o fine-tuning: o laço seguinte chama `loss.backward()`, que é o backpropagation, e o AdamW atualiza os pesos.

# %%
modelo = BertForSequenceClassification.from_pretrained(
    MODELO_NOME,
    num_labels=len(classes),
    id2label=id2label,
    label2id=label2id,
)
modelo.to(DISPOSITIVO)

total_parametros = sum(p.numel() for p in modelo.parameters())
treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"Parâmetros: {total_parametros:,}".replace(",", "."))
print(f"Parâmetros treináveis: {treinaveis:,}".replace(",", "."))

# A perda de BertForSequenceClassification, com num_labels > 1 e rótulos
# inteiros, é a entropia cruzada categórica.
otimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=TAXA_APRENDIZADO,
    weight_decay=DECAIMENTO_PESOS,
)
passos_totais = max(1, len(carregador_treino) * NUM_EPOCAS)
passos_aquecimento = max(1, int(0.1 * passos_totais))
agendador = get_linear_schedule_with_warmup(
    otimizador,
    num_warmup_steps=passos_aquecimento,
    num_training_steps=passos_totais,
)
print("Passos previstos:", passos_totais, "| aquecimento:", passos_aquecimento)

# %% [markdown]
# ## 10. Fine-tuning, backpropagation e early stopping
#
# Em cada lote o modelo calcula a entropia cruzada, `backward()` propaga o gradiente da saída até os pesos, o gradiente é cortado em norma 1 para evitar explosão e o AdamW dá o passo. A validação roda ao fim de cada época. Se a perda de validação não melhorar, a paciência acaba e os pesos voltam para a melhor época. O conjunto de teste não entra nesta decisão.

# %%
def perda_media(carregador):
    modelo.eval()
    soma = 0.0
    quantidade = 0
    with torch.no_grad():
        for lote in carregador:
            lote = {chave: valor.to(DISPOSITIVO) for chave, valor in lote.items()}
            saida = modelo(**lote)
            tamanho = lote["labels"].size(0)
            soma += saida.loss.item() * tamanho
            quantidade += tamanho
    return soma / max(quantidade, 1)


historico = {"epoca": [], "loss_treino": [], "loss_validacao": []}
melhor_val = float("inf")
epocas_sem_melhora = 0
epoca_selecionada = None
caminho_melhor = PASTA_MODELO / "checkpoint_melhor.pt"

for epoca in range(1, NUM_EPOCAS + 1):
    modelo.train()
    soma = 0.0
    quantidade = 0
    for numero_lote, lote in enumerate(carregador_treino, start=1):
        if LIMITE_LOTES is not None and numero_lote > LIMITE_LOTES:
            break
        lote = {chave: valor.to(DISPOSITIVO) for chave, valor in lote.items()}
        otimizador.zero_grad(set_to_none=True)
        saida = modelo(**lote)
        perda = saida.loss
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()
        agendador.step()
        tamanho = lote["labels"].size(0)
        soma += perda.item() * tamanho
        quantidade += tamanho
        if numero_lote == 1 or numero_lote % 20 == 0:
            print(
                f"Época {epoca} | lote {numero_lote}/{len(carregador_treino)} | "
                f"perda do lote {perda.item():.4f}"
            )

    loss_treino = soma / max(quantidade, 1)
    loss_val = perda_media(carregador_val)
    historico["epoca"].append(epoca)
    historico["loss_treino"].append(loss_treino)
    historico["loss_validacao"].append(loss_val)
    print(f"Época {epoca} concluída | loss treino {loss_treino:.4f} | loss validação {loss_val:.4f}")

    if loss_val < melhor_val:
        melhor_val = loss_val
        epocas_sem_melhora = 0
        epoca_selecionada = epoca
        torch.save(modelo.state_dict(), caminho_melhor)
        print("Validação melhorou. Pesos guardados.")
    else:
        epocas_sem_melhora += 1
        print(f"Validação não melhorou. Paciência {epocas_sem_melhora}/{PACIENCIA}.")
        if epocas_sem_melhora >= PACIENCIA:
            print("Early stopping.")
            break

if epoca_selecionada is None:
    raise RuntimeError("O treinamento não gravou uma época válida.")

modelo.load_state_dict(torch.load(caminho_melhor, map_location=DISPOSITIVO, weights_only=True))
print("Pesos restaurados da época", epoca_selecionada)

# %% [markdown]
# ## 11. Avaliação no conjunto de teste
#
# Accuracy, precisão, revocação e F1 saem somente das perguntas de teste. Há média macro, que trata cada categoria com o mesmo peso, e média ponderada, que considera o tamanho de cada categoria. A matriz de confusão cruza o rótulo da base com a classe prevista.

# %%
modelo.eval()
predicoes = []
verdadeiros = []
perdas_teste = 0.0
quantidade_teste = 0
with torch.no_grad():
    for lote in carregador_teste:
        lote = {chave: valor.to(DISPOSITIVO) for chave, valor in lote.items()}
        saida = modelo(**lote)
        tamanho = lote["labels"].size(0)
        perdas_teste += saida.loss.item() * tamanho
        quantidade_teste += tamanho
        predicoes.extend(saida.logits.argmax(dim=-1).cpu().tolist())
        verdadeiros.extend(lote["labels"].cpu().tolist())

loss_teste = perdas_teste / max(quantidade_teste, 1)
acuracia = accuracy_score(verdadeiros, predicoes)
precisao_macro, revocacao_macro, f1_macro, _ = precision_recall_fscore_support(
    verdadeiros, predicoes, average="macro", zero_division=0
)
precisao_pond, revocacao_pond, f1_pond, _ = precision_recall_fscore_support(
    verdadeiros, predicoes, average="weighted", zero_division=0
)
prec_c, rev_c, f1_c, suporte = precision_recall_fscore_support(
    verdadeiros, predicoes, labels=list(range(len(classes))), average=None, zero_division=0
)
matriz = confusion_matrix(verdadeiros, predicoes, labels=list(range(len(classes))))
relatorio = classification_report(
    verdadeiros,
    predicoes,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
    digits=4,
)

print("Loss de teste (não usada para escolher o modelo):", round(loss_teste, 4))
print("Acurácia:", round(acuracia, 4))
print("Precisão macro / ponderada:", round(precisao_macro, 4), round(precisao_pond, 4))
print("Revocação macro / ponderada:", round(revocacao_macro, 4), round(revocacao_pond, 4))
print("F1 macro / ponderada:", round(f1_macro, 4), round(f1_pond, 4))
print("\nRelatório por categoria:\n")
print(relatorio)

indice_escolhido = historico["epoca"].index(epoca_selecionada)
loss_treino_escolhida = historico["loss_treino"][indice_escolhido]
loss_val_escolhida = historico["loss_validacao"][indice_escolhido]
print("Loss de treino da época escolhida:", round(loss_treino_escolhida, 4))
print("Loss de validação da época escolhida:", round(loss_val_escolhida, 4))

# %% [markdown]
# ## 12. Gráficos

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(historico["epoca"], historico["loss_treino"], marker="o", label="Treino", color="#0f6e6e")
ax.plot(historico["epoca"], historico["loss_validacao"], marker="o", label="Validação", color="#b45309")
ax.set_xlabel("Época")
ax.set_ylabel("Perda (entropia cruzada)")
ax.set_title("Perda durante o fine-tuning")
ax.set_xticks(historico["epoca"])
ax.legend()
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "curva_perda.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=classes,
    yticklabels=classes,
    ax=ax,
)
ax.set_xlabel("Classe prevista")
ax.set_ylabel("Classe da base")
ax.set_title("Matriz de confusão no conjunto de teste")
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=classes, y=f1_c, ax=ax, color="#1d4e89")
ax.set_ylim(0, 1)
ax.set_ylabel("F1")
ax.set_xlabel("Categoria")
ax.set_title("F1 por categoria no conjunto de teste")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "f1_por_categoria.png", dpi=140)
plt.close(fig)
print("Gráficos salvos em resultados/.")

# %% [markdown]
# ## 13. Testes práticos
#
# Os textos abaixo foram escritos para a demonstração. Não fazem parte do conjunto de teste e não entram no cálculo das métricas. A confiança é a maior probabilidade depois da função softmax.

# %%
def classificar_texto(texto, historico_local):
    texto_limpo = limpar_texto(texto)
    modelo.eval()
    codigos = tokenizador(
        texto_limpo,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    codigos = {chave: valor.to(DISPOSITIVO) for chave, valor in codigos.items()}
    with torch.no_grad():
        logits = modelo(**codigos).logits
        probabilidades = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    indice = int(probabilidades.argmax())
    mapa = modelo.config.id2label
    categoria = mapa.get(indice, mapa.get(str(indice)))
    nomes = [mapa.get(i, mapa.get(str(i))) for i in range(len(probabilidades))]
    distribuicao = {nome: float(probabilidades[i]) for i, nome in enumerate(nomes)}
    resultado = {
        "texto": texto_limpo,
        "categoria": categoria,
        "confianca": float(probabilidades[indice]),
        "probabilidades": distribuicao,
    }
    historico_local.append(resultado)
    print("Texto:", resultado["texto"])
    print("Categoria prevista:", resultado["categoria"])
    print("Confiança:", f"{resultado['confianca']:.2%}")
    for nome, probabilidade in sorted(distribuicao.items(), key=lambda item: item[1], reverse=True):
        print(f"  {nome}: {probabilidade:.2%}")
    print("-" * 60)
    return resultado


historico_classificacoes = []
exemplos_demonstracao = [
    "Estou precisando agendar uma consulta na unidade básica de saúde.",
    "Estou com febre, dor no corpo e tosse há dois dias. O que pode ser isso?",
    "Posso tomar ibuprofeno junto com o remédio da pressão?",
    "Meu exame de sangue deu glicose alta. Preciso repetir o exame?",
    "Caminhar trinta minutos por dia ajuda a controlar a pressão arterial?",
    "Dor no joelho há uma semana. Devo procurar o clínico geral ou o ortopedista?",
    "Para que serve o hormônio da tireoide no organismo?",
]
for exemplo in exemplos_demonstracao:
    classificar_texto(exemplo, historico_classificacoes)

fig, ax = plt.subplots(figsize=(11, 5.5))
tabela = pd.DataFrame([item["probabilidades"] for item in historico_classificacoes])
tabela.index = [f"Exemplo {i}" for i in range(1, len(tabela) + 1)]
tabela.plot(kind="bar", ax=ax, colormap="tab10")
ax.set_ylim(0, 1)
ax.set_ylabel("Probabilidade")
ax.set_title("Probabilidades nos exemplos de demonstração")
ax.legend(title="Categoria", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "exemplos_probabilidades.png", dpi=140)
plt.close(fig)

with (PASTA_RESULTADOS / "exemplos_classificacao.json").open("w", encoding="utf-8") as arquivo:
    json.dump(historico_classificacoes, arquivo, ensure_ascii=False, indent=2)

print("Histórico de demonstração:", len(historico_classificacoes), "textos")

# %% [markdown]
# ## 14. Salvamento do modelo
#
# A pasta `modelos/bertimbau-saude-intencao` recebe os pesos, o tokenizer e o mapa de rótulos. No Colab, mude `SALVAR_NO_GOOGLE_DRIVE` para `True` antes de rodar esta célula se quiser copiar o modelo para o Drive e não treinar de novo na próxima abertura.

# %%
modelo.save_pretrained(PASTA_MODELO)
tokenizador.save_pretrained(PASTA_MODELO)
caminho_melhor.unlink(missing_ok=True)

registro_rotulos = {
    "modelo_base": MODELO_NOME,
    "classes": classes,
    "id2label": {str(i): nome for i, nome in id2label.items()},
    "max_length": MAX_LENGTH,
}
with (PASTA_MODELO / "rotulos.json").open("w", encoding="utf-8") as arquivo:
    json.dump(registro_rotulos, arquivo, ensure_ascii=False, indent=2)
with (PASTA_RESULTADOS / "rotulos.json").open("w", encoding="utf-8") as arquivo:
    json.dump(registro_rotulos, arquivo, ensure_ascii=False, indent=2)

por_categoria = []
for i, nome in enumerate(classes):
    por_categoria.append(
        {
            "categoria": nome,
            "precision": float(prec_c[i]),
            "recall": float(rev_c[i]),
            "f1": float(f1_c[i]),
            "suporte": int(suporte[i]),
        }
    )

metricas = {
    "execucao": {
        "data_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dispositivo": str(DISPOSITIVO),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch": torch.__version__,
        "modelo": MODELO_NOME,
        "semente": SEMENTE,
        "max_length": MAX_LENGTH,
        "tamanho_lote": TAMANHO_LOTE,
        "epocas_configuradas": NUM_EPOCAS,
        "epocas_executadas": historico["epoca"],
        "epoca_selecionada": epoca_selecionada,
        "learning_rate": TAXA_APRENDIZADO,
        "weight_decay": DECAIMENTO_PESOS,
        "paciencia_early_stopping": PACIENCIA,
        "amostra_por_classe_limite": AMOSTRA_POR_CLASSE,
        "limite_lotes": LIMITE_LOTES,
        "n_treino": int(len(idx_treino)),
        "n_validacao": int(len(idx_val)),
        "n_teste": int(len(idx_teste)),
    },
    "base": {
        "nome": "MedPT",
        "huggingface": "AKCIT/MedPT",
        "url": "https://huggingface.co/datasets/AKCIT/MedPT",
        "n_bruto": int(len(df_bruto)),
        "n_apos_limpeza": int(len(df)),
        "n_amostra": int(len(amostra)),
        "invalidos_removidos": int(removidos_invalidos),
        "linhas_com_rotulo_conflitante": int(n_conflitos),
        "duplicatas_removidas": int(removidos_dup),
        "distribuicao_bruta": {
            str(k): int(v) for k, v in df_bruto["question_type"].value_counts(dropna=False).items()
        },
        "distribuicao_limpa": {str(k): int(v) for k, v in distribuicao_completa.items()},
        "distribuicao_amostra": {str(k): int(v) for k, v in amostra["question_type"].value_counts().items()},
        "observacao_rotulos": (
            "question_type foi gerado com gpt-oss-120b, segundo o card do MedPT. "
            "As perguntas são do corpus publicado; a intenção não é anotação médica humana."
        ),
    },
    "historico": historico,
    "teste": {
        "loss": float(loss_teste),
        "accuracy": float(acuracia),
        "precision_macro": float(precisao_macro),
        "recall_macro": float(revocacao_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precisao_pond),
        "recall_weighted": float(revocacao_pond),
        "f1_weighted": float(f1_pond),
        "loss_treino_epoca_selecionada": float(loss_treino_escolhida),
        "loss_validacao_epoca_selecionada": float(loss_val_escolhida),
        "por_categoria": por_categoria,
        "matriz_confusao": matriz.tolist(),
        "classes_na_ordem_da_matriz": classes,
        "relatorio": relatorio,
    },
}
with (PASTA_RESULTADOS / "metricas.json").open("w", encoding="utf-8") as arquivo:
    json.dump(metricas, arquivo, ensure_ascii=False, indent=2)
with (PASTA_RESULTADOS / "relatorio_classificacao.txt").open("w", encoding="utf-8") as arquivo:
    arquivo.write(relatorio)

amostra_exportada = amostra.loc[:, ["id", "question", "question_type"]].copy()
divisao = np.full(len(amostra), "", dtype=object)
divisao[idx_treino] = "treino"
divisao[idx_val] = "validacao"
divisao[idx_teste] = "teste"
amostra_exportada["divisao"] = divisao
amostra_exportada.to_csv(PASTA_RESULTADOS / "amostra_utilizada.csv", index=False)

print("Modelo salvo em", PASTA_MODELO)
print("Métricas salvas em", PASTA_RESULTADOS / "metricas.json")

SALVAR_NO_GOOGLE_DRIVE = False
PASTA_DRIVE = "/content/drive/MyDrive/dsm-aprendizado-maquina-bertimbau"
if SALVAR_NO_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        import shutil

        drive.mount("/content/drive")
        destino = Path(PASTA_DRIVE)
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PASTA_MODELO, destino / PASTA_MODELO.name, dirs_exist_ok=True)
        shutil.copytree(PASTA_RESULTADOS, destino / PASTA_RESULTADOS.name, dirs_exist_ok=True)
        print("Cópia enviada para", destino)
    except ImportError:
        print("google.colab não está disponível fora do Colab. A cópia para o Drive foi ignorada.")

# %% [markdown]
# ## 15. Como carregar o modelo sem treinar de novo
#
# Na próxima abertura do Colab, rode as células de instalação e importação e depois esta célula, apontando `ORIGEM_MODELO` para a pasta salva no Drive. Não é preciso repetir o fine-tuning para classificar um texto.

# %%
ORIGEM_MODELO = PASTA_MODELO
# Exemplo no Colab, depois de montar o Drive:
# ORIGEM_MODELO = Path("/content/drive/MyDrive/dsm-aprendizado-maquina-bertimbau/bertimbau-saude-intencao")

del modelo
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tokenizador_carregado = BertTokenizer.from_pretrained(ORIGEM_MODELO)
modelo_carregado = BertForSequenceClassification.from_pretrained(ORIGEM_MODELO)
modelo_carregado.to(DISPOSITIVO)
modelo_carregado.eval()
print("Modelo recarregado de", ORIGEM_MODELO)
print("Rótulos:", modelo_carregado.config.id2label)

# %% [markdown]
# ## 16. Leitura dos resultados para a apresentação
#
# Os números desta execução estão em `resultados/metricas.json` e nos gráficos da mesma pasta. Eles valem para a amostra, a semente e o dispositivo impressos acima. Uma nova execução no Colab, com mais dados ou com GPU, produz outros valores. O modelo não deve ser apresentado como classificador clínico validado: os rótulos de intenção do MedPT foram gerados por outro modelo de linguagem, e os textos são perguntas públicas de um serviço online, não prontuários do SUS.
#
# ## Execução de referência já realizada
#
# Em 11/11/2026, às 16:13 (horário de Brasília), este mesmo arquivo rodou em CPU, sem GPU, em 17 minutos. Amostra de 2.800 perguntas (400 por classe), divisão 1.960 / 420 / 420, `max_length` 128, lote 8, três épocas. A validação melhorou nas três, e a época 3 foi mantida.
#
# No teste de 420 textos: acurácia 0,7381, precisão macro 0,7420, revocação macro 0,7381, F1 macro 0,7372. Perda de treino da época escolhida: 0,4502. Perda de validação: 0,8310. Perda de teste: 0,8222.
#
# O F1 mais alto foi escolha de profissionais de saúde (0,9580). O mais baixo foi Outros (0,5370). A perda de treino caiu bem mais que a de validação, sinal de overfitting incipiente. Esses valores estão gravados em `resultados/metricas.json` e não devem ser substituídos por arredondamento de memória.
