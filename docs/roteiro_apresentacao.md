# Roteiro para a apresentação de 11/11/2026

Tempo sugerido: 12 a 15 minutos. Quem fala pode dividir os blocos entre Gabriel Henrique Santana da Silva e Luiz Felipe dos Santos. Os números citados na hora devem ser lidos de `resultados/metricas.json` e dos gráficos, não de memória de um ensaio antigo.

## 1. Problema (1 minuto)

A atenção primária recebe texto o tempo todo: queixa, dúvida sobre exame, pergunta sobre medicamento, pedido de orientação. Antes de qualquer resposta, a unidade precisa saber **que tipo de mensagem chegou**. O TCC de Gabriel A. Coelho (Mackenzie, 2025) trata desse cenário com um chatbot de RAG e modelos de linguagem para o SUS. O resumo não usa BERTimbau. Neste trabalho, a disciplina pede o fine-tuning do BERTimbau, então a entrega é um classificador de intenção que organiza o texto. Ele não diagnostica e não prescreve.

## 2. Base (2 minutos)

Abrir `dados/base_saude.xlsx` no Excel e mostrar a aba `perguntas`. O algoritmo lê esse arquivo, não uma tabela solta na memória. A aba `origem` diz que as linhas vieram do MedPT.

Abrir a célula da base no notebook.

- Nome: MedPT, de Farber e colegas, LREC 2026.
- Link: https://huggingface.co/datasets/AKCIT/MedPT
- Origem descrita pelos autores: perguntas e respostas públicas do Doctoralia, em português.
- Coluna de texto: `question`.
- Coluna de rótulo: `question_type`.
- Dizer em voz alta o número de linhas que a célula imprimiu e as sete categorias, se a execução confirmar sete.
- Dizer a limitação sem enfeitar: o card do MedPT informa que a intenção foi rotulada por outro modelo (`gpt-oss-120b`). A pergunta é real; o rótulo não é uma anotação de prontuário feita por médico.
- A licença SPDX não estava declarada no card consultado em 11/11/2026. O uso aqui é acadêmico e a citação do artigo é obrigatória.

## 3. Por que não as categorias do enunciado (30 segundos)

Agendamento, sintomas, medicamentos, exames, orientações e encaminhamentos eram sugestões. A base real não usa esses nomes. Forçar esses rótulos seria inventar uma anotação. Os exemplos da demonstração ainda trazem frases desse tipo para mostrar o que o modelo faz com elas.

## 4. Pré-processamento e divisão (1 minuto)

- A limpeza não apaga acento nem palavra frequente, porque o BERTimbau precisa da frase.
- Saem texto vazio, duplicata e pergunta com dois rótulos diferentes.
- Divisão estratificada: 70% treino, 15% validação, 15% teste.
- A validação só serve para o early stopping. O teste só aparece na avaliação.

## 5. O que é o BERTimbau e o que é fine-tuning (2 minutos)

- BERTimbau é um BERT pré-treinado em português brasileiro por Souza, Nogueira e Lotufo (BRACIS 2020), no corpus BrWaC. O identificador é `neuralmind/bert-base-portuguese-cased`. São 12 camadas e cerca de 110 milhões de parâmetros na versão base.
- O tokenizer oficial transforma a frase em `input_ids` e `attention_mask`.
- Fine-tuning: os pesos já treinados continuam no modelo e uma camada nova, com uma saída por categoria, é treinada junto.
- A função de perda é a entropia cruzada. `loss.backward()` calcula o gradiente. O AdamW aplica o passo. Isso é o backpropagation pedido no enunciado.
- Early stopping devolve os pesos da época em que a validação foi melhor, para não ficar com um modelo que só decorou o treino.

## 6. Métricas (3 minutos)

A execução de referência, em CPU, no dia 11/11/2026, às 16:13 (horário de Brasília), ficou com acurácia 0,7381 e F1 macro 0,7372 no teste de 420 textos. A perda de treino da época 3 foi 0,4502 e a de validação foi 0,8310. Escolha de profissionais de saúde teve F1 0,9580. Outros teve F1 0,5370. Confirmem esses números em `resultados/metricas.json` antes de falar, se o notebook tiver sido executado de novo.

Mostrar, nesta ordem, os arquivos em `resultados/`:

1. `curva_perda.png`: as duas curvas. Se a de validação sobe enquanto a de treino cai, há sinal de overfitting; o early stopping existe para cortar isso.
2. `matriz_confusao.png`: onde o modelo confunde categorias parecidas.
3. `f1_por_categoria.png` e o relatório impresso.
4. Ler acurácia, precisão, revocação e F1 macro e ponderada. Dizer o tamanho do teste. Não arredondar para “quase 100%” se o arquivo não disser isso.

## 7. Demonstração (2 minutos)

No notebook, a função `classificar_texto` mostra a frase, a categoria e a confiança. Na interface Streamlit, digitar ou escolher um exemplo, clicar em classificar e abrir o histórico.

Deixar claro: esses exemplos não entram na conta das métricas.

## 8. Limitações e fechamento (1 minuto)

- CPU ou amostra reduzida, se for o caso da execução registrada: o número vale para aquela configuração.
- Textos do Doctoralia não são os textos de uma unidade do SUS.
- Rótulo de intenção gerado por modelo.
- Sem validação com equipe de saúde.
- O passo seguinte, se o curso continuasse, seria anotação humana de mensagens reais da atenção primária e, só então, comparação com o desenho de chatbot do TCC.

## Perguntas que o professor pode fazer

**Por que BERT e não regressão logística?** O BERT monta uma representação que depende do contexto da frase inteira. Uma regressão em cima de contagem de palavras trata cada termo de forma mais isolada. O trabalho não chega a treinar esse segundo modelo, então a comparação fica conceitual.

**O que é a attention mask?** É um vetor que diz ao modelo quais posições são palavra e quais são preenchimento para completar o tamanho fixo.

**Houve vazamento de teste?** A escolha do checkpoint usa a validação. As métricas finais usam o teste. Perguntas iguais foram unificadas antes da divisão.

**O modelo pode ser usado na UBS amanhã?** Não. Falta dado da própria unidade, revisão profissional dos rótulos e um protocolo de segurança. Serve para mostrar o método.
