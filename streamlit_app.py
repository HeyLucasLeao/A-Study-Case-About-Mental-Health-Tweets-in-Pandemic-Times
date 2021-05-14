import streamlit as st
import pandas as pd
import plotly.express as px

PATH = r'frequency.csv'
TREND = r'trend.csv'

def fig_frequency_per_day():
    fig = px.line(frequency_per_day.rename(columns={'quantidade': 'Tweets'}), 
    labels={'value': 'Quantidade', 
    'date': 'Data',
    'quantidade': 'Tweets'},
    title='Tweets por Dia')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

def fig_frequency_per_year():
    frequency_per_year['date'] = [str(x) for x in frequency_per_year['date']]
    frequency_per_year['date'] = [x[:4] for x in frequency_per_year['date']]
    fig = px.bar(frequency_per_year, 
    x='date', 
    y='quantidade',
    labels={'quantidade': 'Tweets',
    'date': 'Date'})
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

def fig_frequency_per_month():
    fig = px.line(frequency_per_month.rename(columns={'quantidade': 'Tweets'}), 
    labels={'value': 'Quantidade', 'date': 'Data'},
    title='Tweets por Mês',
    line_shape='spline')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

def fig_diff_year():
    diff_per_year['date'] = [str(x) for x in diff_per_year['date']]
    diff_per_year['date'] = [x[:4] for x in diff_per_year['date']]
    fig = px.bar(diff_per_year, 
    x='date', 
    y='quantidade',
    labels={'quantidade': 'Percentual (%)',
    'date': 'Data'},
    title='Crescimento Anual de Tweets sobre Ansiedade')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:.2f} %")
    return fig

def fig_diff_month():
    diff_per_month.rename(columns={'quantidade': 'Percentual'}, inplace=True)
    fig = px.bar(diff_per_month.round(2),
    labels={'value': 'Percentual (%)',
    'date': 'Data'},
    title='Crescimento Mensal de Tweets sobre Ansiedade')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, 
    hovertemplate="%{y} %")
    return fig

def fig_total_labels():
    total_labels = round((frequency['labels'].value_counts() / len(frequency['labels'])) * 100, 2)
    total_labels = total_labels.reset_index()
    total_labels.rename(columns={'labels': 'value', 'index': 'labels'}, inplace=True)
    total_labels['categorias'] = ['Ansiedade (Geral)',
    'Crise',
    'Depressão',
    'Terapia',
    'Assuntos Graves']
    fig = px.bar(total_labels, 
    x='categorias',
    y='value',
    labels={'value': 'Percentual (%)', 'date': 'Data', 'categorias': "Categoria"},
    title='Distribuição de Categorias')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:2.f} %")
    return fig

def fig_frequency_per_label():
    fig = px.line(frequency_per_label, 
    x='date', 
    y='quantidade', 
    facet_row='labels', 
    height=1200, 
    width=800,
    labels={'quantidade': 'Tweets', 'date': 'Data'},
    title='Frequência por Categoria (Diária)')
    fig.update_yaxes(matches=None)
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_xaxes(showticklabels=True)
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1], textangle=45))
    return fig

def fig_frequency_per_label_per_month():
    df = frequency_per_label.copy()
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    frequency_per_label_per_month = df.groupby(['labels',pd.Grouper(freq='M')]).sum()
    frequency_per_label_per_month = frequency_per_label_per_month.reset_index()

    fig = px.line(frequency_per_label_per_month, 
    x='date', 
    y='quantidade', 
    facet_row='labels', 
    labels={'quantidade': 'Tweets', 'date': 'Data'},
    title='Frequência por Categoria (Mensal)')
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(matches=None)
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1], textangle=45))
    return fig

def fig_trend():
    trend.rename(columns={'trend': 'Tendência'}, inplace=True)
    fig = px.line(trend,
    labels={'value': 'Quantidade Média',
    'index': 'Data'},
    title='Predição de Tendência de Tweets')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

frequency = pd.read_csv(PATH)
frequency.drop(columns='Unnamed: 0', inplace=True)
frequency_per_day = frequency.groupby('date').sum()
frequency_per_day.index = pd.to_datetime(frequency_per_day.index)
frequency_per_day.index.freq = 'D'
frequency_per_year = frequency_per_day.groupby(pd.Grouper(freq='Y')).sum()
frequency_per_year.reset_index(inplace=True)
frequency_per_month = frequency_per_day.groupby(pd.Grouper(freq='M')).sum()
frequency_per_month.drop(columns='labels', inplace=True)
frequency_per_day.drop(columns='labels', inplace=True)

frequency_per_label = frequency.groupby(['labels', 'date']).sum()
frequency_per_label = frequency_per_label.reset_index()

diff_per_year = frequency_per_day[frequency_per_day.index < '2021-01-01'].groupby(pd.Grouper(freq='Y')).sum().pct_change()
diff_per_year.reset_index(inplace=True)
diff_per_year['quantidade'] = diff_per_year['quantidade'] * 100

diff_per_month = frequency_per_month.pct_change() * 100

dici = {0: 'Ansiedade (Geral)',
    1: 'Crise',
    3: 'Depressão',
    2: 'Terapia',
    4: 'Assuntos Graves'}
frequency_per_label['labels'] = frequency_per_label['labels'].apply(lambda x: dici[x])
trend = pd.read_csv(TREND, index_col='Unnamed: 0')

st.set_page_config(page_title='Ansiedade e Pandemia')

st.title('Ansiedade e Pandemia: Análise Exploratória no Tweets ao longo de 4 anos.')

st.write(f"""Toda pesquisa, seja ela, vem por meio de uma curiosidade, um problema ou 
um interesse para com sociedade. Isto não foi tão diferente deste projeto. Por vezes,
desde que me isolei em casa, fiquei me questionando quanto a saúde mental minha e de
colegas. Como eu, por exemplo, trabalho ativamente [em um projeto sobre Covid-19](https://heylucasleao.github.io/), não é tão fácil manter a cabeça tão sã com um
projeto desse todos os dias, e nem imagino como seja para a linha de frente. Eu, que tenho o privilégio de ter acompanhamento de um psiquiatra, comecei a me questionar quanto ao aumento de casos como
depressão ou ansiedade durante pandemia. Com [alguns artigos](https://www.nossasaude.com.br/dicas-de-saude/pandemia-aumenta-casos-de-depressao-e-ansiedade-no-brasil/), pude
perceber que de fato estava indo ao pensamento correto, e queria trabalhar com que sei e com o que posso. Neste
caso, resolvi saber se houve um crescimento de tweets, isto é, mensagens publicadas pela rede social Twitter,
com a palavra chave ansiedade, seguindo algumas regras na qual falarei no próximo parágrafo, durante o 
período de janeiro de 2018 a março de 2021.

Como twitter há uma quantidade enorme de dados, estabeleci as seguintes regras:

    1. Tweets a partir de retweets não serão contados.
    2. Os mesmos, não poderiam conter links, tampouco hashtags.
    3. Tweets iguais não serão contados.
    
Seguindo estas regras, foram coletados cerca de 7,5 milhões de tweets, os quais podem ser baixados pelo [archive](https://archive.org/download/scraping-em-ansiedade/), 
um número que considerei substancial para uma noção de magnitude sobre o tema. 
Por motivos de segurança, foram removidos os nomes dos usuários de cada tweet.""")

box1 = st.selectbox('Selecione período do gráfico',('Diário', 'Mensal', 'Anual'), key=1)

if 'Diário' in box1:
    st.plotly_chart(fig_frequency_per_day())
elif 'Mensal' in box1:
    st.plotly_chart(fig_frequency_per_month())
elif 'Anual' in box1:
    st.write("""Se tratando da frequência anual, é observado um aumento bem visual durante os 4 anos. 
Um detalhe que, apenas 4 meses de 2021 já foram maiores que o ano inteiro de 2019.""")
    st.plotly_chart(fig_frequency_per_year())

st.write("""Ainda sim, queria mais respostas. A partir disso, procurei saber se em contextos específicos sobre o tema houve 
maior crescimento. Como 7,5 milhões é um número muito grande para se categorizar manualmente,
necessitava de um modelo de aprendizagem de máquina para processamento de linguagem natural. 
Um que vem se destacando atualmente por sua capacidade de extrair contextos 
por meio de Self-Attention é o modelo [BERT](https://huggingface.co/transformers/model_doc/bert.html), baseado
nas camadas encoders dos [Transformers](https://huggingface.co/transformers/).
Como não tenho capacidade computacional muito grande, tampouco um banco de dados grande para treinar do zero,
resolvi utilizar um modelo pré-treinado para [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/#:~:text=Transfer%20learning%20is%20a%20machine,model%20on%20a%20second%20task.&text=Common%20examples%20of%20transfer%20learning,your%20own%20predictive%20modeling%20problems.), 
e apenas treinar camadas de um feed forward simples, com o intuito de:

    O modelo pré-treinado entenderá semântica brasileira, enquanto o feed forward aprenderá 
    o contexto atual.

O modelo pré-treinado selecionado foi o [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased), disponível pelo site Hugging Face.
A arquitetura e construção do modelo se encontra no [repositório deste projeto.](https://github.com/HeyLucasLeao/ansiedade-e-pandemia)
 
Durante o treino, o modelo chegou a 94% de acurácia nos dados de treino e 83% nos dados de validação. Tanto 
os dados de treino, teste e o logger estão dentro do repositório.

Devido ao objetivo deste projeto, considerei o percentual de acurácia suficiente para satisfazer meus desejos.""")

st.plotly_chart(fig_total_labels())

st.write("""Após processamento dos dados, foi identificado que 66,54% dos tweets coletados tratam-se sobre ansiedade 
em geral, 20,01% para crises, 6,01% sobre tópicos relacionados a depressão junto a ansiedade, apenas 
5,61% sobre tratamentos para tal e 1,83% sobre tweets relacionados a possíveis assuntos mais 
graves, como suicídio.""")

box2 = st.selectbox('Selecione período do gráfico',('Diário', 'Mensal'), key=2)

if 'Diário' in box2:
    st.plotly_chart(fig_frequency_per_label())
elif 'Mensal' in box2:
    st.plotly_chart(fig_frequency_per_label_per_month())

st.write("""Ao visualizar a frequência de cada categoria por dia e por mês, observou-se que 
seus aumentos foram similares e proporcionais ao longo do tempo, sem identificar um crescimento específico em 
uma delas em alguma época, o que responde minha dúvida quanto à existência 
de um pico específico em um tema comoparado aos outros.""")

box3 = st.selectbox('Selecione período do gráfico',('Anual', 'Mensal'), key=3)

if 'Anual' in box3:
    st.plotly_chart(fig_diff_year())
elif 'Mensal' in box3:
    st.plotly_chart(fig_diff_month())

st.write("""Observando pelo crescimento percentual, vemos que 2019 teve um aumento de 57% 
sobre tweets, e 100% em 2020 comparado 2019. Dividindo cada crescimento percentual por meses
a partir de 2018, conseguimos observar uma tendência de aumento ao redor dos meses, com o 
pico no mês de março de 2020. Para 2019, podemos correlacionar a falta de diretrizes para sanar
a crise econômica, enquanto para 2020, podemos correlacionar com a pandemia, os sentimentos
manifestados durante toda a questão com a saúde pública e as incertezas sobre o futuro do Brasil.""")

st.write("""A partir disto, procurei entender qual seria a predição para os próximos 
meses. Para isto, selecionei meu modelo preferido para dados sequencias, o [Light GBM](https://lightgbm.readthedocs.io/en/latest/), 
o qual foi selecionado devido aos seus ótimos resultados apresentados em diversas competições de Machine Learning.
A partir disto, ele foi treinado com a frequência de tweets por dias, e seus hiperparâmetros selecionados por [método Bayesiano](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html), 
atingindo uma média de 10% de erro percentual absoluto médio simétrico na validação cruzada por [Expanding Window Splitter](https://www.sktime.org/en/latest/api_reference/modules/auto_generated/sktime.forecasting.model_selection.ExpandingWindowSplitter.html), um dos
métodos de validação cruzada para séries temporais. Devido a sazonalidade diária e apenas meu 
desejo de observar a tendência de crescimento, suavizei exponencialmente esta predição por [Holt & Winters.](https://www.opservices.com.br/holt-winters/)
É possível observar a continuação desta tendência sobre o tema ao redor deste ano, com uma média de 
tweets por mês chegando a 539 mil, um valor substancialmente alto se tratando de meses passados.""")

st.plotly_chart(fig_trend())

st.write("""Com isso, me vi questionando a última vez que conversei abertamente sobre saúde mental
com meus amigos, após ver todos estes números. Fico feliz, até dizer, que temas sobre saúde 
mental têm sido mais debatido nos últimos anos, entretanto, isto não significa necessariamente
que a busca por auxílio psicológico também esteja aumentando na mesma proporção, o que fica um 
alerta ao leitor sobre questionar-se a última vez desde que procurou ajuda. É por este objetivo que acredito ser 
importante compartilhar este pequeno estudo, como uma forma de pedido para que caso tenha o 
privilégio ao acesso e a disponibilidade de algum sistema de saúde para acompanhamento ou 
grupo de apoio, que o faça. Dependendo do local, há também acessos gratuitos, como Unidade Básica de Saúde (UBS),
clínicas universitárias de psicologia aplicada, Centros de Atenção Psicossociais (CAPS), Centro de Valorização da Vida (CVV)
e pronto atendimentos designados pelas respectivas prefeituras e localidades. Saúde mental é um assunto importante na qual não pode passar 
desapercebido por ninguém, principalmente em plena pandemia.""")