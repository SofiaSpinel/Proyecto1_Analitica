import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# Inicializa la aplicación Dash
app = dash.Dash(__name__)

# Conexión a la base de datos en la nube
env_path = "C:\\Users\\sarit\\proy2\\app.env"
load_dotenv(dotenv_path=env_path)
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
DBNAME = os.getenv('DBNAME')

engine = create_engine("postgresql+psycopg2://postgres:xxxxxx@isabel.cvcju9aqozqq.us-east-1.rds.amazonaws.com:5432/datos2")
#engine = psycopg2.connect(
 #   dbname=DBNAME,
  #  user=USER,
   # password=PASSWORD,
    #host=HOST,
    #port=PORT
#)

# Leer datos desde la base de datos
#cursor = engine.cursor()
query = "SELECT * FROM variables"
df = pd.read_sql(query, engine)
print("conexión a la base de datos remota listo")

# Leer modelo desde un archivo BIF
reader = BIFReader("monty.bif")
modelo = reader.get_model()

# Inferencia de la probabilidad posterior
infer = VariableElimination(modelo)

# Define la disposición de la aplicación
app.layout = html.Div(
    [
        html.H6("Ingrese las variables para la inferencia"),
        html.Div(["Tuition: ", dcc.Dropdown(id='tuition', value='1', options=['1', '0'])]),
        html.Br(),
        html.Div(["Age: ", dcc.Dropdown(id='age', value='17-28', options=['17-28', '28-39', '39-50', '50-61', '61-70'])]),
        html.Br(),
        html.Div(["Course: ", dcc.Dropdown(id='course', value='discom', options=['discom', 'salud', 'soced', 'tec', 'turismo', 'vetequi'])]),
        html.Br(),
        html.Div(["Mquali: ", dcc.Dropdown(id='mquali', value='basic', options=['basic', 'higher', 'secundary', 'technical', 'unknown'])]),
        html.Br(),
        html.Div(["Grade: ", dcc.Dropdown(id='grade', value='0-4', options=['0-4', '12-16', '16-20', '8-12'])]),
        html.Br(),
        html.Div(["Mocup: ", dcc.Dropdown(id='mocup', value='admi', options=['admi', 'farm', 'industry', 'scientific', 'technicians'])]),
        html.Br(),
        html.H6("Probabilidad de Desertar:"),
        html.Br(),
        html.Div(["Target:", html.Div(id='output-target')]),
        html.Div(["Complemento de Target:", html.Div(id='output-complement')]),
        dcc.Graph(id='target-pie'),
        dcc.Checklist(
            id='target-checklist',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),  # Valor inicial (todos los valores seleccionados)
            labelStyle={'display': 'block'}  # Mostrar etiquetas en bloques para una mejor visualización
        ),
        dcc.Graph(id='target-graph'),
        dcc.Graph(id='target-graph-2'),
        dcc.Graph(id='target-graph-3')

    ])

# Callback para el Gráfico de Pastel
@app.callback(
    Output('target-pie', 'figure'),
    [Input('tuition', 'value'),
     Input('age', 'value'),
     Input('course', 'value'),
     Input('mquali', 'value'),
     Input('grade', 'value'),
     Input('mocup', 'value')]
)
def update_target_pie(tuition, age, course, mquali, grade, mocup):
    evidence = {
        "tuition": tuition,
        "age": age,
        "course": course,
        "mquali": mquali,
        "grade": grade,
        "mocup": mocup
    }
    posterior_p = infer.query(["target"], evidence=evidence)
    prob_target = posterior_p.values[0]
    complement = 1 - prob_target

    labels = ["Desertar", "Graduarse"]
    values = [prob_target, complement]
    fig = px.pie(names=labels, values=values, title="Distribución de Probabilidad de Desertar")

    return fig

# Callback para el Gráfico de Barras del primer Dash
@app.callback(
    Output('target-graph', 'figure'),
    Input('target-checklist', 'value')
)
def update_graph(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    fig = px.bar(filtered_df, x="course", color="target", title="Distribución de 'target' por Carrera")
    return fig
print("listo")
# Callback para el Gráfico de Barras del segundo Dash
@app.callback(
    Output('target-graph-2', 'figure'),
    Input('target-checklist', 'value')
)
def update_graph_2(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    fig = px.bar(filtered_df, x="age", color="target", title="Distribución de 'target' por Edad")
    return fig
print("listo")
# Callback para el Gráfico de Barras del tercer Dash
@app.callback(
    Output('target-graph-3', 'figure'),
    Input('target-checklist', 'value')
)
def update_graph_3(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    fig = px.bar(filtered_df, x="grade", color="target", title="Distribución de 'target' por Calificación")
    return fig
print("listo")
# Ejecuta la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)

app.run_server(host="0.0.0.0", debug=True, port=8001)