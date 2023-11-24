import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash_table import DataTable  # Importa DataTable desde dash_table

# Resto del código...


# Cargar el conjunto de datos
file_path = "C:/Users/sarit/OneDrive - Universidad de los Andes/SEMESTRE 7/Analítica/DatosFINALproyecto3.xlsx"
df = pd.read_excel(file_path)

# Convertir las columnas categóricas a tipo 'category'
categorical_columns = ['PERIODO', 'COLE_BILINGUE', 'COLE_JORNADA', 'COLE_NATURALEZA', 'ESTU_GENERO',
                        'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET']
df[categorical_columns] = df[categorical_columns].astype('category')

# Crear la aplicación de Dash
app = dash.Dash(__name__)

# Diseño del tablero
app.layout = html.Div([
    html.H1("Análisis de Datos - Dashboard"),

    # Gráfico de barras por periodo y puntaje promedio
    dcc.Graph(
        id='bar-chart',
        figure=px.bar(df, x='PERIODO', y='puntaje', color='ESTU_GENERO',
                      labels={'PERIODO': 'Periodo', 'puntaje': 'Puntaje Promedio'})
    ),

    # Gráfico de barras por naturaleza del colegio y conteo
    dcc.Graph(
        id='bar-chart-cole-naturaleza',
        figure=px.bar(df, x='COLE_NATURALEZA', color='COLE_NATURALEZA',
                      labels={'COLE_NATURALEZA': 'Naturaleza del Colegio'},
                      title='Distribución por Naturaleza del Colegio')
    ),

    # Gráfico de pastel por género
    dcc.Graph(
        id='pie-chart',
        figure=px.pie(df, names='ESTU_GENERO', title='Distribución por Género')
    ),

    # Gráfico de barras por estrato y conteo
    dcc.Graph(
        id='bar-chart-estrato',
        figure=px.bar(df, x='FAMI_ESTRATOVIVIENDA', color='FAMI_ESTRATOVIVIENDA',
                      labels={'FAMI_ESTRATOVIVIENDA': 'Estrato'},
                      title='Distribución por Estrato')
    ),

    # Tabla con datos
    html.Div([
        html.H3("Datos del DataSet"),
        DataTable(  # Cambia dcc.DataTable a DataTable
            id='data-table',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.to_dict('records'),
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
