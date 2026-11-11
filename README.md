# Transformação digital no SUS: classificação de textos de saúde com BERTimbau

Projeto da disciplina de Aprendizado de Máquina. O programa classifica a intenção de perguntas de saúde em português brasileiro. O modelo de partida é o BERTimbau e o ajuste é feito por fine-tuning em PyTorch.

O classificador organiza texto. Não faz diagnóstico, não indica medicamento e não substitui a equipe de uma unidade de saúde.

## Identificação

| Campo | Registro |
| --- | --- |
| Disciplina | Aprendizado de Máquina |
| Curso | Desenvolvimento de Software Multiplataforma (DSM) |
| Semestre | 5º semestre |
| Integrantes | Gabriel Henrique Santana da Silva e Luiz Felipe dos Santos |
| Professor | Bruno Zolotareff |
| Data da apresentação | 11/11/2026 |

## Descrição do projeto

**Título.** Transformação digital no SUS: explorando o uso da IA conversacional na unidade clínica primária.

**Resumo.** Mensagens que chegam a uma unidade de atenção primária misturam queixa, dúvida sobre exame, pergunta de medicamento e pedido de orientação. Este trabalho treina um classificador para separar a intenção do texto antes de qualquer resposta. A referência de problema é o TCC de Gabriel A. Coelho (Mackenzie, 2025), que descreve um chatbot com RAG e modelos de linguagem para o SUS. A técnica implementada aqui é outra, porque a disciplina exige o fine-tuning do BERTimbau. A análise do que o resumo público do TCC realmente informa está em [`docs/analise_artigo.md`](docs/analise_artigo.md).

**Problema.** Há muito texto de saúde e pouco tempo para triar o assunto de cada mensagem. Um modelo pode sugerir a categoria. A decisão continua com a equipe.

**Objetivo geral.** Ajustar o BERTimbau, a partir dos pesos pré-treinados em português brasileiro, para classificar a intenção de perguntas de saúde e medir o resultado em um conjunto de teste que não participou do treinamento.

**Objetivos específicos.**

- Usar uma base pública, em português, com texto e rótulo verificáveis.
- Tokenizar com o tokenizer oficial do BERTimbau.
- Treinar com entropia cruzada, backpropagation e AdamW.
- Separar treino, validação e teste sem reutilizar a mesma pergunta nos três conjuntos.
- Publicar acurácia, precisão, revocação, F1, matriz de confusão e as perdas de treino e validação.
- Permitir que um texto digitado receba categoria, confiança e probabilidade das demais classes.

**Como a aplicação funciona.** O texto é limpo de forma leve, vira uma sequência de subpalavras com máscara de atenção e entra no BERTimbau já ajustado. A última camada produz um valor por categoria. A função softmax transforma esses valores em probabilidades. A categoria exibida é a de maior probabilidade, e a confiança é essa probabilidade.

## Tecnologias

- Python
- Google Colab, com o mesmo código executável em CPU
- BERTimbau (`neuralmind/bert-base-portuguese-cased`)
- Fine-tuning
- PyTorch
- Hugging Face Transformers e Datasets
- scikit-learn, pandas, NumPy, Matplotlib e Seaborn
- Streamlit, apenas para a demonstração local

## Modelo

**BERTimbau.** É um modelo BERT treinado para o português brasileiro por Fábio Souza, Rodrigo Nogueira e Roberto Lotufo, publicado no BRACIS 2020. A versão usada aqui é a base, com 12 camadas, identificador `neuralmind/bert-base-portuguese-cased`, pré-treinada no corpus BrWaC. O vocabulário distingue maiúsculas de minúsculas. Na execução, a arquitetura de classificação carregada informa a quantidade de parâmetros treináveis no log do notebook.

**Fine-tuning.** Os pesos do pré-treinamento continuam no modelo. Uma camada linear nova, com uma saída para cada intenção, começa aleatória. O treinamento muda os pesos antigos e os novos para a tarefa de classificação. Não é um uso congelado do modelo só para extrair embeddings.

**Treinamento.** Cada lote produz a entropia cruzada entre a classe correta e a distribuição prevista. `loss.backward()` calcula o gradiente da perda em relação aos pesos. Esse é o backpropagation. O AdamW usa o gradiente, com taxa de aprendizado inicial `2e-5` e decaimento de pesos `0.01`, para atualizar os parâmetros. Um agendador linear esquenta a taxa no começo e a reduz até o fim. O gradiente é cortado na norma 1. Ao fim de cada época, a perda de validação é medida. Se ela não melhorar, o early stopping interrompe e devolve os pesos da melhor época de validação.

## Base de dados

O algoritmo não baixa a base na hora de classificar. O treino e a demonstração leem o arquivo **`dados/base_saude.xlsx`**, aba `perguntas`, colunas `id`, `pergunta` e `intencao`. A aba `origem` guarda a procedência. A aba `categorias` resume as contagens. Dá para abrir a planilha no Excel e filtrar a coluna `intencao`.

| Campo | Registro |
| --- | --- |
| Arquivo usado pelo algoritmo | `dados/base_saude.xlsx` |
| Nome da fonte | MedPT |
| Página | https://huggingface.co/datasets/AKCIT/MedPT |
| Arquivo | https://huggingface.co/datasets/AKCIT/MedPT/resolve/main/data/train-00000-of-00001.parquet |
| Fonte | Farber, Brito, Dollis, Ribeiro, Sousa e Filho. *MedPT: A Massive Medical Question Answering Dataset for Brazilian-Portuguese Speakers*. LREC 2026, p. 2419–2429. DOI: https://doi.org/10.63317/5mrjjtbsf6rv |
| Origem descrita pelos autores | Perguntas e respostas de pacientes e profissionais no Doctoralia, em português brasileiro, com perguntas anônimas |
| Registros na fonte MedPT | 384.095 |
| Registros na planilha | 181.760 perguntas únicas |
| Colunas lidas pelo algoritmo | `pergunta` (texto) e `intencao` (rótulo). `id` só identifica a linha |
| Licença | O card consultado em 11/11/2026 não declara uma licença SPDX. O conjunto estava público e sem pedido de acesso. O uso neste trabalho é acadêmico e a citação do artigo é obrigatória |

**Categorias encontradas em `question_type`, antes da limpeza.**

| Categoria | Registros |
| --- | --- |
| Tratamento | 142.143 |
| Diagnóstico | 130.206 |
| Epidemiologia | 45.511 |
| Estilo de vida saudável | 21.746 |
| Outros | 19.897 |
| Escolha de profissionais de saúde | 19.167 |
| Anatomia e fisiologia | 5.425 |

Na fonte não havia valores ausentes nessas colunas. Havia 202.323 repetições exatas do texto da pergunta, porque a mesma pergunta pode aparecer com mais de uma resposta. A planilha do projeto fica com as 181.760 perguntas únicas. As contagens por categoria e o gráfico estão em [`resultados/distribuicao_categorias.png`](resultados/distribuicao_categorias.png) e em `resultados/metricas.json`.

O card do MedPT informa que `question_type` foi produzido pelo modelo `gpt-oss-120b` a partir do texto da pergunta. As perguntas vêm do corpus publicado. A intenção não é uma anotação feita por profissionais de saúde. As sugestões do enunciado (agendamento, sintomas, medicamentos, exames, orientações e encaminhamentos) não são os rótulos desta base e não foram forçadas.

**Por que esta base.** É pública, está em português do Brasil, tem texto livre de saúde e um rótulo de intenção com sete classes. Esse rótulo serve para ensaiar a organização de mensagens na atenção primária. O domínio não é o prontuário do SUS: são perguntas de um serviço online.

## Resultados

Os números abaixo são os da execução registrada em `resultados/metricas.json`. Eles não devem ser trocados por uma estimativa. Se o notebook for treinado de novo, com outra amostra ou em GPU, o arquivo muda e esta seção precisa ser atualizada com o que o programa imprimir.

Execução de referência concluída em 11/11/2026, às 16:13 (horário de Brasília). Durou 17 minutos. Rodou em CPU, com PyTorch 2.15.0, sem GPU. Semente 42. Amostra de 400 perguntas por classe, 2.800 no total, divididas em 1.960 de treino, 420 de validação e 420 de teste, 60 de cada classe no teste. Sequências de até 128 tokens. Em mil textos medidos da amostra, oito passavam desse limite. Lote de 8. Taxa de aprendizado `2e-5`. Três épocas, todas com melhora na validação, então o early stopping não interrompeu e a época 3 foi a escolhida. O modelo de classificação tem 108.928.519 parâmetros treináveis.

| Métrica no teste | Valor |
| --- | --- |
| Acurácia | 0,7381 |
| Precisão macro | 0,7420 |
| Revocação macro | 0,7381 |
| F1 macro | 0,7372 |
| Precisão ponderada | 0,7420 |
| Revocação ponderada | 0,7381 |
| F1 ponderada | 0,7372 |
| Perda de teste | 0,8222 |

As médias macro e ponderada coincidem até a quarta casa porque o teste tem o mesmo número de exemplos em cada classe.

| Categoria | Precisão | Revocação | F1 | Suporte |
| --- | --- | --- | --- | --- |
| Anatomia e fisiologia | 0,7121 | 0,7833 | 0,7460 | 60 |
| Diagnóstico | 0,8302 | 0,7333 | 0,7788 | 60 |
| Epidemiologia | 0,7368 | 0,7000 | 0,7179 | 60 |
| Escolha de profissionais de saúde | 0,9661 | 0,9500 | 0,9580 | 60 |
| Estilo de vida saudável | 0,7581 | 0,7833 | 0,7705 | 60 |
| Outros | 0,6042 | 0,4833 | 0,5370 | 60 |
| Tratamento | 0,5867 | 0,7333 | 0,6519 | 60 |

| Época | Perda de treino | Perda de validação |
| --- | --- | --- |
| 1 | 1,4812 | 1,0389 |
| 2 | 0,7396 | 0,8502 |
| 3 | 0,4502 | 0,8310 |

A perda de treino cai mais do que a de validação. Na época 3 a validação ainda melhorou, por pouco, de 0,8502 para 0,8310, enquanto o treino foi a 0,4502. Isso é um começo de overfitting: o modelo acerta cada vez mais o treino e ganha pouco fora dele. Mais épocas, nesta amostra, tenderiam a decorar pergunta.

Na matriz, a diagonal concentra os acertos. A classe mais estável é a escolha de profissionais de saúde, com 57 dos 60 textos de teste na célula certa. A classe Outros é a mais frágil: só 29 dos 60 ficam nela, e o restante se espalha, sobretudo para anatomia e fisiologia (10) e epidemiologia (8). Tratamento acerta 44 e troca principalmente com Outros (7) e estilo de vida (5). Epidemiologia manda 9 textos para tratamento. Essas trocas são esperadas quando a pergunta mistura assuntos e quando Outros funciona como resto, não como um tema único.

Sete frases escritas para a demonstração, fora do teste, foram classificadas pelo modelo salvo. Agendar consulta na unidade foi para escolha de profissionais, com confiança 58%, e Outros ficou em segundo, com 23%. Febre e tosse foram para diagnóstico (87%). Ibuprofeno com remédio de pressão foi para tratamento (96%). Glicose alta no exame foi para diagnóstico (87%). Caminhada e pressão foram para estilo de vida saudável (94%). Dor no joelho e a dúvida entre clínico e ortopedista foram para escolha de profissionais (96%). A função da tireoide foi para anatomia e fisiologia (80%). O pedido de agendamento não tem uma classe própria na base, então a confiança fica mais baixa. Isso deve ser dito na apresentação.

Gráficos gerados pela execução:

- [`resultados/distribuicao_categorias.png`](resultados/distribuicao_categorias.png)
- [`resultados/distribuicao_amostra.png`](resultados/distribuicao_amostra.png)
- [`resultados/curva_perda.png`](resultados/curva_perda.png)
- [`resultados/matriz_confusao.png`](resultados/matriz_confusao.png)
- [`resultados/f1_por_categoria.png`](resultados/f1_por_categoria.png)
- [`resultados/exemplos_probabilidades.png`](resultados/exemplos_probabilidades.png)

**Desempenho.** A leitura correta depende das duas curvas e da matriz, não de um único percentual. A curva de treino mostra se a perda cai. A de validação mostra se o modelo continua útil fora do treino. A matriz mostra quais intenções se confundem. O conjunto de teste é o único usado nas métricas de acerto. A perda de teste é informada à parte e não escolhe o checkpoint.

**Limitações.** A amostra estratificada não usa as 181.760 perguntas. Textos longos são cortados em 128 tokens. Os rótulos de intenção foram gerados por outro modelo. O texto não é de uma unidade do SUS. Não houve revisão por equipe de saúde. Por isso o resultado não autoriza dizer que o sistema está pronto para uso clínico, mesmo que o F1 fique alto.

## Como executar

### Google Colab

1. Publique este projeto no GitHub.
2. Abra o Colab em *Arquivo → Abrir notebook → GitHub* e escolha `notebooks/classificacao_saude_bertimbau.ipynb`. O atalho, depois que o repositório existir, tem a forma `https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/main/notebooks/classificacao_saude_bertimbau.ipynb`.
3. Em *Ambiente de execução → Alterar tipo de ambiente de execução*, selecione GPU, de preferência T4.
4. Execute as células na ordem, de cima para baixo. A primeira instala o que faltar. A base é baixada da Hugging Face.
5. Na célula de hiperparâmetros, a execução de referência usa 400 textos por classe, sequências de 128 tokens, lotes de 8 e no máximo 3 épocas, com early stopping. Em uma GPU folgada, `AMOSTRA_POR_CLASSE` pode aumentar. `LIMITE_LOTES` deve continuar `None`.
6. A avaliação imprime as métricas do teste. A célula seguinte classifica sete frases de demonstração, que não entram nessa conta.
7. Para não treinar de novo, mude `SALVAR_NO_GOOGLE_DRIVE` para `True` na célula de salvamento. Na abertura seguinte, monte o Drive e aponte `ORIGEM_MODELO` para `Meu Drive/dsm-aprendizado-maquina-bertimbau/bertimbau-saude-intencao`.

### Máquina local

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/treinar.py
```

Sem GPU, o mesmo arquivo usa a CPU. Os pesos ficam em `modelos/bertimbau-saude-intencao/`. Eles passam de cem megabytes e não entram no Git. O notebook e o script recriam essa pasta.

### Classificar pela interface

Com o modelo já treinado na pasta acima:

```bash
streamlit run app/demo.py --server.port 4728 --server.address 0.0.0.0
```

Digite um texto ou escolha um exemplo, clique em **Classificar** e leia a categoria, a confiança e o histórico da sessão. O aviso no topo da página permanece visível de propósito.

## Organização do repositório

```
dados/base_saude.xlsx               planilha que o algoritmo lê
app/demo.py                         interface Streamlit, com classificação ao vivo
site/index.html                     página com as métricas e os gráficos desta execução
docs/analise_artigo.md              leitura do TCC de referência
docs/roteiro_apresentacao.md        falas sugeridas para a banca
notebooks/classificacao_saude_bertimbau.ipynb
resultados/                         métricas, gráficos e amostra usada
src/treinar.py                      mesma lógica do notebook, executável
requirements.txt
LICENSE                             MIT para o código deste repositório
```

A página `site/index.html` repete os números da execução de referência para consulta rápida. A classificação de um texto novo continua no notebook e, quando os pesos estão na pasta do modelo, no Streamlit.

O notebook é gerado a partir de `src/treinar.py` por `python scripts/py_para_notebook.py`. As células seguem a ordem: apresentação, instalação, configuração, base, análise, pré-processamento, amostra, divisão, tokenização, lotes, carregamento do BERTimbau, fine-tuning, avaliação, gráficos, testes, salvamento, recarga e fechamento.

## Créditos e licenças

- Código deste repositório: Gabriel Henrique Santana da Silva e Luiz Felipe dos Santos, licença MIT, arquivo [`LICENSE`](LICENSE).
- BERTimbau: Souza, Fábio; Nogueira, Rodrigo; Lotufo, Roberto. *BERTimbau: Pretrained BERT Models for Brazilian Portuguese*. In: Cerri, Ricardo; Prati, Ronaldo C. (ed.). *Intelligent Systems*. BRACIS 2020. Lecture Notes in Computer Science, vol. 12319. Springer, Cham, 2020, p. 403–417. DOI: https://doi.org/10.1007/978-3-030-61377-8_28. Modelo: https://huggingface.co/neuralmind/bert-base-portuguese-cased.
- Base MedPT: Farber et al., LREC 2026, DOI https://doi.org/10.63317/5mrjjtbsf6rv. Página: https://huggingface.co/datasets/AKCIT/MedPT.
- Referência de problema: Coelho, Gabriel A. *Transformação digital no SUS: explorando o uso de IA generativa no cuidado clínico primário*. TCC. Orientador: Murilo Gazzola. Universidade Presbiteriana Mackenzie, Faculdade de Computação e Informática, 2025. https://dspace.mackenzie.br/handle/10899/41755.

A licença MIT cobre o código escrito para a disciplina. Não relicencia o BERTimbau, o corpus MedPT nem o TCC.
