from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from figures import *

app = Dash('Ansiedade & Pandemia')
app.layout = html.Div([
    html.Br(),
    html.H3("Ansiedade e Pandemia: Análise Exploratória no Tweets ao longo de 4 anos.", style={'text-align': 'center', 'size': 6, 'offset': 6}),
    html.Br(),
    html.P([f"""    Toda pesquisa, seja ela, vem por meio de uma curiosidade, um problema ou um interesse para com sociedade. 
    Isto não foi tão diferente deste projeto. Por vezes,desde que me isolei em casa, fiquei me questionando quanto a saúde mental minha e de colegas. 
    Como eu, por exemplo, trabalho ativamente """, html.A('em um projeto sobre Covid-19, ', href='https://share.streamlit.io/heylucasleao/covid-no-amazonas/main'), """não é tão fácil manter a cabeça tão sã 
    com um projeto desse todos os dias, e nem imagino como seja para a linha de frente. Eu, que tenho o privilégio de ter acompanhamento de um psiquiatra, 
    comecei a me questionar quanto ao aumento de casos como depressão ou ansiedade durante pandemia. """, html.A('Com alguns artigos ', href='https://www.nossasaude.com.br/dicas-de-saude/pandemia-aumenta-casos-de-depressao-e-ansiedade-no-brasil/'), 
    """pude perceber que de fato estava indo ao pensamento correto, e queria trabalhar com que sei e com o que posso. Neste caso, resolvi saber se houve um crescimento 
    de tweets, isto é, mensagens publicadas pela rede social Twitter, com a palavra chave ansiedade, seguindo algumas regras na qual falarei no próximo parágrafo, 
    durante o período de janeiro de 2018 a março de 2021."""
    ], style={'text-align': 'justify-all', 'text-indent': 50}),
    dbc.Row(dbc.Col(html.Div("""Como twitter há uma quantidade enorme de dados, estabeleci as seguintes regras:""")), style={'text-align': 'center'}),
    html.Br(),
    dbc.Row([
        dbc.Col(html.Div(" - 1. Derivados de retweets não serão contados."), style={'text-align': 'center'}),
        dbc.Col(html.Div(" - 2. Não poderiam conter links nem hashtags."), style={'text-align': 'center'}),
        dbc.Col(html.Div(" - 3. Tweets iguais não serão contados."), style={'text-align': 'center'}),
    ]),
    html.Br(),
    html.P([
        "Seguindo estas regras, foram coletados cerca de 7,5 milhões de ", html.A("tweets.", href="https://archive.org/download/scraping-em-ansiedade/"),  
        """ Este valor considerei substancial para uma noção de magnitude sobre o tema. 
        Por motivos de segurança, foram removidos os nomes dos usuários de cada tweet."""
    ], style={'text-align': 'justify-all', 'text-indent': 50}),
    html.Label(['Selecione o período gráfico']),
    dcc.Dropdown(
        id='frequency-dropdown',
        options=[
            {'label': 'Diário', 'value': 'day'},
            {'label': 'Mensal', 'value': 'month'},
            {'label': 'Anual', 'value': 'year'}
        ],
        placeholder='Selecione o período do gráfico',
        value='day',
        multi=False,
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id='frequency-graph'),
    html.P(["""Se tratando da frequência anual, é observado um aumento bem visual durante os 4 anos. 
            Um detalhe que, apenas 4 meses de 2021 já foram maiores que o ano inteiro de 2019."""
        ], style={'text-align': 'justify-all', 'text-indent': 50}),
    html.P(["""Ainda sim, queria mais respostas. A partir disso, procurei saber se em contextos específicos sobre o tema houve 
        maior crescimento. Como 7,5 milhões é um número muito grande para se categorizar manualmente,
        necessitava de um modelo de aprendizagem de máquina para processamento de linguagem natural. 
        Um que vem se destacando atualmente por sua capacidade de extrair contextos 
        por meio de Self-Attention é o modelo """,html.A('BERT', href="https://huggingface.co/transformers/model_doc/bert.html"),
        " baseado nas camadas encoders dos ", html.A("Transformers.", href="https://huggingface.co/transformers/"), 
        html.Br()], 
        style={'text-align': 'justify-all', 'text-indent': 50}),
    html.P(["""Como não tenho capacidade computacional muito grande, tampouco um banco de dados grande para treinar do zero,
        resolvi utilizar um modelo pré-treinado para """, html.A('Transfer Learning', href="https://machinelearningmastery.com/transfer-learning-for-deep-learning/#:~:text=Transfer%20learning%20is%20a%20machine,model%20on%20a%20second%20task.&text=Common%20examples%20of%20transfer%20learning,your%20own%20predictive%20modeling%20problems"),
        """ e apenas treinar camadas de um feed forward simples, com o intuito de: """,

        html.B("""O modelo pré-treinado entenderá semântica brasileira, enquanto o feed forward aprenderá
        o contexto atual. """),
        
        """O modelo pré-treinado selecionado foi o """, html.A('BERTimbau', href='https://huggingface.co/neuralmind/bert-base-portuguese-cased'), " disponível pelo site ",
        html.A("Hugging Face.", href='https://huggingface.co/'),
        " A arquitetura e construção do modelo se encontra no ", html.A('repositório deste projeto.', href='https://github.com/HeyLucasLeao/ansiedade-e-pandemia')], 
        style={'text-align': 'justify-all', 'text-indent': 50})


    ,
    html.P([
        """Durante o treino, o modelo chegou a 94% de acurácia nos dados de treino e 83% nos dados de validação. Tanto 
        os dados de treino, teste e o logger estão dentro do repositório.

        Devido ao objetivo deste projeto, considerei o percentual de acurácia suficiente para satisfazer meus desejos."""
    ], style={'text-align': 'justify-all', 'text-indent': 50}),
    dcc.Graph(figure=fig_total_labels()),
    html.P(["""Após processamento dos dados, foi identificado que 66,54% dos tweets coletados tratam-se sobre ansiedade 
            em geral, 20,01% para crises, 6,01% sobre tópicos relacionados a depressão junto a ansiedade, apenas 
            5,61% sobre tratamentos para tal e 1,83% sobre tweets relacionados a possíveis assuntos mais 
            graves, como suicídio."""], style={'text-align': 'justify-all', 'text-indent': 50}),
    html.Label(['Selecione o período gráfico']),
    dcc.Dropdown(
        id='per-label-dropdown',
        options=[
            {'label': 'Diário', 'value': 'day'},
            {'label': 'Mensal', 'value': 'month'},
        ],
        placeholder='Selecione o período do gráfico',
        value='day',
        multi=False,
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id='per-label-graph'),
    html.P(["""Ao visualizar a frequência de cada categoria por dia e por mês, observou-se que 
        seus aumentos foram similares e proporcionais ao longo do tempo, sem identificar um crescimento específico em 
        uma delas em alguma época, o que responde minha dúvida quanto à existência 
        de um pico específico em um tema comoparado aos outros."""], style={'text-align': 'justify-all', 'text-indent': 50}),
    html.Label(['Selecione o período gráfico']),
    dcc.Dropdown(
        id='diff-dropdown',
        options=[
            {'label': 'Mensal', 'value': 'month'},
            {'label': 'Anual', 'value': 'year'},
        ],
        value='month',
        multi=False,
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id='diff-graph'),
    html.P(["""Observando pelo crescimento percentual, vemos que 2019 teve um aumento de 57% 
        sobre tweets, e 100% em 2020 comparado 2019. Dividindo cada crescimento percentual por meses
        a partir de 2018, conseguimos observar uma tendência de aumento ao redor dos meses, com o 
        pico no mês de março de 2020. Para 2019, podemos correlacionar a falta de diretrizes para sanar
        a crise econômica, enquanto para 2020, podemos correlacionar com a pandemia, os sentimentos
        manifestados durante toda a questão com a saúde pública e as incertezas sobre o futuro do Brasil.""", 
        html.Br()], style={'text-align': 'justify-all', 'text-indent': 50}),

    html.P(["""A partir disto, procurei entender qual seria a predição para os próximos 
        meses. Para isto, selecionei meu modelo preferido para dados sequencias, """, html.A('Light GBM', href='https://lightgbm.readthedocs.io/en/latest/'), 
        """, o qual foi selecionado devido aos seus ótimos resultados apresentados em diversas competições de Machine Learning.
        A partir disto, ele foi treinado com a frequência de tweets por dias, e seus hiperparâmetros selecionados por """, html.A('método Bayesiano', href='https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html'), 
        ", atingindo uma média de 10% de erro percentual absoluto médio simétrico na validação cruzada por ", html.A('Expanding Window Splitter', href="https://www.sktime.org/en/latest/api_reference/modules/auto_generated/sktime.forecasting.model_selection.ExpandingWindowSplitter.html"), """, um dos
        métodos de validação cruzada para séries temporais. Devido a sazonalidade diária e apenas meu 
        desejo de observar a tendência de crescimento, suavizei exponencialmente esta predição por """, html.A("Holt & Winters.", href="https://www.opservices.com.br/holt-winters/"),
        """ É possível observar a continuação desta tendência sobre o tema ao redor deste ano, com uma média de 
        tweets por mês chegando a 539 mil, um valor substancialmente alto se tratando de meses passados."""], style={'text-align': 'justify-all', 'text-indent': 50}),
    dcc.Graph(figure=fig_trend()),
    html.P(["""Com isso, me vi questionando a última vez que conversei abertamente sobre saúde mental
        com meus amigos, após ver todos estes números. Fico feliz, até dizer, que temas sobre saúde 
        mental têm sido mais debatido nos últimos anos, entretanto, isto não significa necessariamente
        que a busca por auxílio psicológico também esteja aumentando na mesma proporção, o que fica um 
        alerta ao leitor sobre questionar-se a última vez desde que procurou ajuda. É por este objetivo que acredito ser 
        importante compartilhar este pequeno estudo, como uma forma de pedido para que caso tenha o 
        privilégio ao acesso e a disponibilidade de algum sistema de saúde para acompanhamento ou 
        grupo de apoio, que o faça. Dependendo do local, há também acessos gratuitos, como Unidade Básica de Saúde (UBS),
        clínicas universitárias de psicologia aplicada, Centros de Atenção Psicossociais (CAPS), Centro de Valorização da Vida (CVV)
        e pronto atendimentos designados pelas respectivas prefeituras e localidades. Saúde mental é um assunto importante na qual não pode passar 
        desapercebido por ninguém, principalmente em plena pandemia."""], style={'text-align': 'justify-all', 'text-indent': 50})
    ])

@app.callback(
    Output(component_id='frequency-graph', component_property='figure'),
    [Input(component_id='frequency-dropdown', component_property='value')]
)

def fig_frequency(x):
    return frequency_dict[x]

@app.callback(
    Output(component_id='per-label-graph', component_property='figure'),
    [Input(component_id='per-label-dropdown', component_property='value')]
)

def fig_label(x):
    return per_label_dict[x]

@app.callback(
    Output(component_id='diff-graph', component_property='figure'),
    [Input(component_id='diff-dropdown', component_property='value')]
)

def fig_diff(x):
    return diff_dict[x]

app.run_server(debug=True)