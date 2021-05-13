import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from os import listdir

PATH = r'C:\Users\heylu\Documents\github\A Study Case About Mental Health Tweets in Pandemic Times\streamlit\frequency.csv'

frequency = pd.read_csv(PATH)

frequency_per_day = frequency.groupby('date').sum()
frequency_per_day.index = pd.to_datetime(frequency_per_day.index)
frequency_per_day.index.freq = 'D'
frequency_per_month = frequency_per_day.groupby(pd.Grouper(freq='M')).sum()
frequency_per_month.drop(columns='labels', inplace=True)
frequency_per_day.drop(columns='labels', inplace=True)

fig_frequency_per_day = px.line(frequency_per_day)
fig_frequency_per_month = px.line(frequency_per_month)

st.set_page_config(page_title='Ansiedade e Pandemia')

st.title('Ansiedade e Pandemia: Análise Exploratória no Twitter ao longo de 4 anos.')

st.write(f"""Qualquer pesquisa, seja ela, vem por meio de uma curiosidade, um problema ou 
um interesse para com sociedade. Isto não foi tão diferente deste projeto. Por vezes,
desde que me isolei em casa, fiquei me questionando quanto a saúde mental minha e de
colegas. Como eu, por exemplo, trabalho ativamente [em um projeto sobre Covid-19](https://heylucasleao.github.io/), não é tão fácil manter a cabeça tão sã com um
projeto desse todos os dias, e nem imagino como seja para a linha de frente, mas enfim. Eu, que
tenho o privilégio de ter acompanhamento de um psiquiatra, comecei a me questionar quanto ao aumento de casos como
depressão ou ansiedade durante pandemia. Com [alguns artigos](https://www.nossasaude.com.br/dicas-de-saude/pandemia-aumenta-casos-de-depressao-e-ansiedade-no-brasil/), pude
perceber que de fato estava indo ao pensamento correto, mas queria trabalhar com que sei e com o que posso. Neste
caso, resolvi saber se houve um número de tweets, isto é, mensagens publicadas pela rede social Twitter,
com a palavra chave ansiedade, seguindo algumas regras na qual falarei ao próximo parágrafo, durante o 
período de janeiro de 2018 a março de 2021.

Como twitter há uma quantidade enorme de dados, resolvi seguir as seguintes regras para filtragem:

    1. Não haveria tweets a partir de retweets.
    2. Os mesmos, não poderiam contar links.
    3. Sem hashtags.
    4. Não haveria tweets iguais.
    
Seguindo estas regras, foi coletado cerca de 7,5 milhões de tweets, na qual podem ser baixados pelo [archive](https://archive.org/download/scraping-em-ansiedade/), um número que considerei atrativo
para ter noção de magnitude sobre ansiedade. Foram preservados os nomes dos usuários para tal.

{st.plotly_chart(fig_frequency_per_day)}

{st.plotly_chart(fig_frequency_per_month)}

Ainda sim, queria mais respostas, saber se em contextos específicos sobre o tema houve 
maior crescimento. Como 7,5 milhões é um número muito grande para se categorizar manualmente,
necessitava de um modelo de aprendizagem de máquina para processamento de linguagem natural, 
e um que vem se destacando atualmente por sua capacidade de extrair contextos 
por meio de Self-Attention é o modelo [BERT](https://huggingface.co/transformers/model_doc/bert.html).
Como não tenho capacidade computacional muito grande, tampouco um banco de dados grande para treinar do zero,
resolvi utilizar um modelo pré-treinado para [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/#:~:text=Transfer%20learning%20is%20a%20machine,model%20on%20a%20second%20task.&text=Common%20examples%20of%20transfer%20learning,your%20own%20predictive%20modeling%20problems.), 
e apenas fazer treinar camadas de um feed forward simples, com o intuito de:

    O modelo pré-treinado entenderá semântica brasileira, enquanto o feed forward aprenderá 
    o contexto atual.

O modelo pré-treinado que selecionei foi o [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased), disponível pelo site Hugging Face.
A arquitetura e construção do modelo se encontra no [repositório deste projeto.](https://github.com/HeyLucasLeao/A-Study-Case-About-Mental-Health-Tweets-in-Pandemic-Times)
 
Durante o treino, obteve 94% de acurácia nos dados de treino e 83% nos dados de validação. Tanto 
os dados de treino, teste e o logger estão dentro do repositório.

Como este estudo tem como objetivo uma dimensão de magnitude, considerei o percentual de acerto suficiente para satisfazer meus desejos.""")

