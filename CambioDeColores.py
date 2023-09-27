# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:43:16 2023

@author: sofia
"""

import plotly.express as px
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Supongamos que tienes un DataFrame llamado 'df' con tus datos
# Asegúrate de que 'df' contiene las columnas necesarias para todas las gráficas

# Reemplaza 'df' con tu propio DataFrame
df = pd.read_excel("C:/Users/sofia/OneDrive/Documents/GitHub/Proyecto1_Analitica/data_variables.xlsx")

df.head()
# Inicializa la aplicación Dash
app = dash.Dash(__name__)

# Define una paleta de colores personalizada con tonos más oscuros de "pink" y "purple"
color_palette = {
    "Graduate": "#FF1493",    # Pink oscuro
    "Dropout": "#800080",    # Purple oscuro
    # Agrega más colores aquí para otras categorías si es necesario
}

# Define la disposición de la aplicación
app.layout = html.Div([
    html.H1("Análisis de Datos con Dash"),

    # Gráfico 1: Gráfico de barras apiladas con checklist por target y curso
    html.Div([
        dcc.Checklist(
            id='target-checklist-1',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),
            labelStyle={'display': 'block'}
        ),
        dcc.Graph(id='target-course-graph')
    ]),

    # Gráfico 2: Gráfico de caja con checklist por target
    html.Div([
        dcc.Checklist(
            id='target-checklist-2',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),
            labelStyle={'display': 'block'}
        ),
        dcc.Graph(id='target-grade-boxplot')
    ]),

    # Gráfico 3: Gráfico de barras de conteo con checklist por target
    html.Div([
        dcc.Checklist(
            id='target-checklist-3',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),
            labelStyle={'display': 'block'}
        ),
        dcc.Graph(id='target-age-bar')
    ]),

    # Gráfico 4: Gráfico de barras apiladas con checklist por target, curso y matrícula
    html.Div([
        dcc.Checklist(
            id='target-checklist-4',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),
            labelStyle={'display': 'block'}
        ),
        dcc.Graph(id='target-course-tuition-bar')
    ]),

    # Gráfico 5: Gráfico de barras apiladas con checklist por target y nivel educativo
    html.Div([
        dcc.Checklist(
            id='target-checklist-5',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),
            labelStyle={'display': 'block'}
        ),
        dcc.Graph(id='target-mquali-bar')
    ]),

    # Gráfico 6: Gráfico de barras apiladas con checklist por target, curso y edad
    html.Div([
        dcc.Checklist(
            id='target-checklist-6',
            options=[{'label': target, 'value': target} for target in df['target'].unique()],
            value=df['target'].unique(),
            labelStyle={'display': 'block'}
        ),
        dcc.Graph(id='target-course-age-bar')
    ])
])

# Define la función de colores personalizados
def get_color_map(target_values):
    color_map = {}
    colors = px.colors.qualitative.Plotly
    for i, target in enumerate(target_values):
        color_map[target] = colors[i % len(colors)]
    return color_map

# Define la función de actualización del Gráfico 1
@app.callback(
    Output('target-course-graph', 'figure'),
    Input('target-checklist-1', 'value')
)
def update_course_graph(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    color_map = {target: color_palette.get(target, "green") for target in selected_targets}
    fig = px.bar(
        filtered_df,
        x="course",
        color="target",
        color_discrete_map=color_map,  # Colores personalizados
        title="Distribución de 'target' por Curso"
    )
    return fig

# Define la función de actualización del Gráfico 2
@app.callback(
    Output('target-grade-boxplot', 'figure'),
    Input('target-checklist-2', 'value')
)
def update_grade_boxplot(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    color_map = get_color_map(selected_targets)
    fig = px.box(
        filtered_df,
        x="target",
        y="grade",
        color="target",
        color_discrete_map=color_map,  # Colores personalizados
        title="Distribución de 'target' en función de la Puntuación de 'grade'"
    )
    return fig

# Define la función de actualización del Gráfico 3
@app.callback(
    Output('target-age-bar', 'figure'),
    Input('target-checklist-3', 'value')
)
def update_age_bar(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    color_map = get_color_map(selected_targets)
    fig = px.bar(
        filtered_df,
        x="target",
        color="age",
        color_discrete_map=color_map,  # Colores personalizados
        title="Distribución de 'target' con Colores Codificados por Edad"
    )
    return fig

# Define la función de actualización del Gráfico 4
@app.callback(
    Output('target-course-tuition-bar', 'figure'),
    Input('target-checklist-4', 'value')
)
def update_course_tuition_bar(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    color_map = get_color_map(selected_targets)
    fig = px.bar(
        filtered_df,
        x="course",
        color="target",
        facet_col="tuition",
        color_discrete_map=color_map,  # Colores personalizados
        title="Distribución de 'target' por Curso y Matrícula"
    )
    return fig

# Define la función de actualización del Gráfico 5
@app.callback(
    Output('target-mquali-bar', 'figure'),
    Input('target-checklist-5', 'value')
)
def update_mquali_bar(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    color_map = get_color_map(selected_targets)
    fig = px.bar(
        filtered_df,
        x="mquali",
        color="target",
        category_orders={"mquali": ["basic", "secundary"]},
        color_discrete_map=color_map,  # Colores personalizados
        title="Distribución de 'target' por Nivel Educativo"
    )
    return fig

# Define la función de actualización del Gráfico 6
@app.callback(
    Output('target-course-age-bar', 'figure'),
    Input('target-checklist-6', 'value')
)
def update_course_age_bar(selected_targets):
    filtered_df = df[df['target'].isin(selected_targets)]
    color_map = get_color_map(selected_targets)
    fig = px.bar(
        filtered_df,
        x="course",
        color="target",
        facet_col="age",
        color_discrete_map=color_map,  # Colores personalizados
        title="Distribución de 'target' por Curso y Edad"
    )
    return fig

# Ejecuta la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)