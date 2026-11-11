# Análise da referência científica

Acesso em 11/11/2026. Esta leitura usa a página pública do repositório da Universidade Presbiteriana Mackenzie e o resumo depositado nela. O item lista um PDF (`Gabriel A. Coelho.pdf`, 6,81 MB). O conteúdo integral desse PDF não foi extraído nesta análise. Nada abaixo atribui ao trabalho um procedimento que o resumo e os metadados não registram.

## Identificação

| Campo | Registro público |
| --- | --- |
| Título | Transformação digital no SUS: explorando o uso de IA generativa no cuidado clínico primário |
| Autor | Gabriel A. Coelho |
| Orientador | Murilo Gazzola |
| Tipo | TCC |
| Instituição | Universidade Presbiteriana Mackenzie, Faculdade de Computação e Informática (FCI) |
| Data de emissão | 08/12/2025 |
| Disponibilização no repositório | 28/01/2026 |
| Identificador | https://dspace.mackenzie.br/handle/10899/41755 |
| Página do item | https://dspace.mackenzie.br/items/dbbc08ab-d9a9-40cf-9ba5-0bf19c308d1c/full |
| Idioma | pt_BR |
| Observação do repositório | Indicado para publicação |

Assuntos informados no item: atenção primária à saúde, APS, SUS, inteligência artificial, doenças crônicas, RAG, processamento de documentos médicos, classificação de imagens médicas, LGPD e chatbots de saúde.

## Objetivo da pesquisa

O resumo apresenta o desenvolvimento de um chatbot de inteligência artificial para apoiar a tomada de decisão clínica e oferecer informação acessível a pacientes da Atenção Primária à Saúde, com foco em pessoas com doenças crônicas.

## Problema abordado

O texto situa a transformação digital como oportunidade de ampliar acesso, agilidade e qualidade do cuidado. A gestão de doenças crônicas, descreve o autor, exige comunicação contínua e orientações confiáveis. Os trabalhos correlatos citados no resumo teriam limitações como fluxos de conversa pré-definidos, escopo clínico estreito ou ausência de personalização por perfil de usuário.

## Metodologia descrita no resumo

A solução proposta é um chatbot baseado em Retrieval-Augmented Generation (RAG) e Large Language Models (LLMs). O resumo também descreve processamento multimodal de texto e imagem, classificação de imagens radiológicas e dermatológicas, interpretação de exames, processamento de linguagem natural e adaptação do vocabulário ao perfil do usuário. A implementação, ainda segundo o resumo, inclui segurança, autenticação e auditoria, em conformidade com a LGPD e normas internacionais de privacidade.

O resumo caracteriza o sistema como prova de conceito.

## Modelo de inteligência artificial

O resumo nomeia a abordagem: RAG e LLMs. Não nomeia o LLM utilizado, não cita BERTimbau e não cita `neuralmind/bert-base-portuguese-cased`.

Há um resultado numérico de classificação de imagens: acurácia próxima de 90% em uma amostra de 1.000 registros do repositório DermNet. O resumo não informa a arquitetura desse classificador de imagens.

## Técnicas de treinamento

O resumo público não descreve função de perda, otimizador, taxa de aprendizado, número de épocas, divisão treino/validação/teste nem fine-tuning de um Transformer de texto. Este repositório não completa essas lacunas com suposições.

## Resultados e métricas apresentados

Conforme o resumo:

- desempenho consistente na recuperação semântica;
- adaptação da comunicação ao perfil do usuário;
- acurácia próxima de 90% na classificação de imagens, com 1.000 registros do DermNet.

O resumo não publica precisão, revocação, F1 nem matriz de confusão da tarefa de texto. Também não afirma que o chatbot substitui o profissional de saúde. A conclusão registrada é que a abordagem pode apoiar fluxos da APS, reduzir sobrecarga dos profissionais e ampliar o acesso a orientações, em alinhamento com diretrizes nacionais de transformação digital na saúde.

## Como este projeto usa o artigo

O TCC e este trabalho compartilham o problema: organizar a comunicação da atenção primária com apoio de processamento de linguagem natural, sem transformar o programa em um diagnóstico automático.

A disciplina pede outra técnica: fine-tuning do BERTimbau para classificação de textos. Este repositório implementa essa técnica. Ele não reproduz o chatbot RAG do TCC e não apresenta o BERTimbau como modelo do artigo.

A ponte entre os dois é prática. Um classificador de intenção pode organizar a mensagem que chega a uma unidade antes de qualquer resposta: separar dúvida sobre exame, pedido de orientação, escolha de profissional e outros tipos de texto. Isso é uma etapa de organização documental, não uma conduta clínica.

## O que a apresentação não deve afirmar

- Que o TCC de Coelho treinou o BERTimbau.
- Que as métricas deste repositório repetem as métricas do TCC.
- Que a acurácia de aproximadamente 90% do DermNet vale para a classificação de textos feita aqui.
- Que o sistema fecha diagnóstico, prescreve ou substitui a equipe da unidade.
