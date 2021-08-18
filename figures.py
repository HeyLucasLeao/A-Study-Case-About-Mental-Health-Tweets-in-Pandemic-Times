import pandas as pd
import plotly.express as px

PATH = r'frequency.zip'
TREND = r'trend.csv'

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
    
def fig_frequency_per_day():
    df = frequency_per_day
    fig = px.line(df.rename(columns={'quantidade': 'Tweets'}), 
    labels={'value': 'Quantidade', 
    'date': 'Data',
    'quantidade': 'Tweets'},
    title='Tweets por Dia')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

def fig_frequency_per_month():
    df = frequency_per_month
    fig = px.line(df.rename(columns={'quantidade': 'Tweets'}), 
    labels={'value': 'Quantidade', 'date': 'Data'},
    title='Tweets por Mês',
    line_shape='spline')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

def fig_frequency_per_year():
    df = frequency_per_year
    df['date'] = [str(x) for x in df['date']]
    df['date'] = [x[:4] for x in df['date']]
    fig = px.bar(df, 
    x='date', 
    y='quantidade',
    labels={'quantidade': 'Tweets',
    'date': 'Date'})
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

def fig_diff_year():
    df = diff_per_year
    df['date'] = [str(x) for x in df['date']]
    df['date'] = [x[:4] for x in df['date']]
    fig = px.bar(df, 
    x='date', 
    y='quantidade',
    labels={'quantidade': 'Percentual (%)',
    'date': 'Data'},
    title='Crescimento Anual de Tweets sobre Ansiedade')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:.2f} %")
    return fig

def fig_diff_month():
    df = diff_per_month
    df.rename(columns={'quantidade': 'Percentual'}, inplace=True)
    fig = px.bar(df.round(2),
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
    df = frequency_per_label
    fig = px.line(df, 
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
    df = frequency_per_label
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
    df = trend
    df.rename(columns={'trend': 'Tendência'}, inplace=True)
    fig = px.line(df,
    labels={'value': 'Quantidade Média',
    'index': 'Data'},
    title='Predição de Tendência de Tweets')
    fig.update_layout(hovermode='x', separators=",.")
    fig.update_traces(showlegend=False, hovertemplate="%{y:,f}")
    return fig

frequency_dict = {
    'day': fig_frequency_per_day(),
    'month': fig_frequency_per_month(),
    'year': fig_frequency_per_year()
}

per_label_dict = {
    'day': fig_frequency_per_label(),
    'month': fig_frequency_per_label_per_month()
}

diff_dict = {
    'month': fig_diff_month(),
    'year': fig_diff_year()
}