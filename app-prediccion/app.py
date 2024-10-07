from flask import Flask
import joblib
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go


model = joblib.load('precipitation_model.pkl')

historical_df = pd.read_csv('data.csv')
historical_df['Data Month'] = pd.to_datetime(historical_df['Data Month'], format='%Y%m')
historical_df = historical_df[['Data Month', 'Precipitation']]


historical_df['Precipitation'] = historical_df['Precipitation'] * 3600  # Conversión a mm/h

last_data = historical_df.iloc[-1]

# Iniciar la aplicación Flask
app = Flask(__name__)

# Iniciar la aplicación Dash
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

# Estilos CSS personalizados
styles = {
    'container': {
        'width': '90%',
        'margin': '0 auto',
        'fontFamily': 'Arial, sans-serif',
        'color': '#333',
        'backgroundColor': '#f9f9f9',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.1)'
    },
    'header': {
        'textAlign': 'center',
        'padding': '20px 0',
        'backgroundColor': '#007bff',
        'color': 'white',
        'borderRadius': '10px 10px 0 0',
        'marginBottom': '20px'
    },
    'section': {
        'margin': '20px 0',
        'padding': '20px',
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'backgroundColor': 'white'
    },
    'graph': {
        'textAlign': 'center',
        'marginTop': '20px',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'width': '70%'
    },
    'info': {
        'display': 'inline-block',
        'verticalAlign': 'top',
        'width': '25%',
        'padding': '20px',
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'backgroundColor': '#fafafa',
        'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)',
        'transition': '0.3s'
    },
    'infoTitle': {
        'fontWeight': 'bold',
        'marginBottom': '10px',
        'fontSize': '18px',
        'textAlign': 'left'
    },
    'infoText': {
        'fontSize': '16px',
        'lineHeight': '1.6',
        'textAlign': 'left'
    },
    'button': {
        'padding': '10px 15px',
        'fontSize': '16px',
        'backgroundColor': '#28a745',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginTop': '20px'
    },
    'buttonHover': {
        'backgroundColor': '#218838'
    },
    'accordion': {
        'margin': '10px 0',
        'border': '1px solid #007bff',
        'borderRadius': '5px',
        'overflow': 'hidden'
    },
    'accordionItem': {
        'borderBottom': '1px solid #ddd',
    },
    'accordionHeader': {
        'backgroundColor': '#007bff',
        'color': 'white',
        'padding': '10px',
        'cursor': 'pointer',
        'textAlign': 'left',
        'fontWeight': 'bold'
    },
    'accordionBody': {
        'padding': '10px',
        'backgroundColor': 'white'
    }
}

# Layout del Dashboard
dash_app.layout = html.Div(style=styles['container'], children=[
    html.Div(style=styles['header'], children=[
        html.H1("Predicción de Precipitaciones"),
        html.P("Predicción automática de precipitación basada en variables climáticas.")
    ]),

    html.Div(children=[
        # Gráfico de comparación
        html.Div(style=styles['graph'], children=[
            dcc.Graph(id='comparison-graph')
        ]),

        # Bot informativo
        html.Div(style=styles['info'], children=[
            html.H2("Información sobre precipitaciones", style=styles['infoTitle']),
            html.P(id='precipitation-info', style=styles['infoText']),
            html.Button("Actualizar Predicción", id='update-button', style=styles['button']),
        ])
    ]),

    # Sección de Accordeones
    html.Div(style=styles['section'], children=[
        html.Div(style=styles['accordion'], children=[
            html.Div(style=styles['accordionItem'], children=[
                html.Div(style=styles['accordionHeader'], children="¿Cómo interpretar el valor de la precipitación?", id='toggle-1', n_clicks=0),
                html.Div(style=styles['accordionBody'], id='content-1', children=[
                    html.P("La precipitación es mostrada en milímetros por hora (mm/h). Un valor superior a 10 mm/h se considera significativo y puede indicar lluvias intensas. Los valores más bajos suelen estar asociados con lluvias ligeras o ausencia de lluvia.", style=styles['infoText'])
                ], hidden=True)
            ]),
            html.Div(style=styles['accordionItem'], children=[
                html.Div(style=styles['accordionHeader'], children="Consejos para agricultores", id='toggle-2', n_clicks=0),
                html.Div(style=styles['accordionBody'], id='content-2', children=[
                    html.P("Conocer las predicciones de precipitación es fundamental para tomar decisiones informadas sobre el riego y la siembra. Si se pronostican lluvias intensas, considera evitar actividades que puedan ser perjudicadas por el agua en el suelo, como la cosecha. Por otro lado, si se anticipan lluvias ligeras o ausentes, asegúrate de que tus cultivos estén adecuadamente irrigados.", style=styles['infoText'])
                ], hidden=True)
            ]),
            html.Div(style=styles['accordionItem'], children=[
                html.Div(style=styles['accordionHeader'], children="Explicación de las variables utilizadas", id='toggle-3', n_clicks=0),
                html.Div(style=styles['accordionBody'], id='content-3', children=[
                    html.H3("1. Temperature_A (°C)", style=styles['infoTitle']),
                    html.P("Temperatura máxima registrada en la región. Se mide en grados Celsius. Es importante porque las temperaturas más altas pueden afectar la tasa de evaporación y la formación de nubes.", style=styles['infoText']),

                    html.H3("2. Temperature_D (°C)", style=styles['infoTitle']),
                    html.P("Temperatura mínima registrada en la región. También se mide en grados Celsius. Las temperaturas más bajas durante la noche pueden influir en la condensación y la acumulación de humedad en la atmósfera.", style=styles['infoText']),

                    html.H3("3. Humidity_A (%)", style=styles['infoTitle']),
                    html.P("Humedad relativa durante el día. Se expresa como un porcentaje y es crucial porque niveles altos de humedad significan mayor cantidad de vapor de agua en el aire, lo que puede llevar a precipitaciones.", style=styles['infoText']),

                    html.H3("4. Humidity_D (%)", style=styles['infoTitle']),
                    html.P("Humedad relativa durante la noche. También se mide en porcentaje. Ayuda a entender el nivel de saturación del aire en diferentes momentos del día, afectando la formación de precipitaciones.", style=styles['infoText']),

                    html.H3("5. Water Thickness GRACE (mm)", style=styles['infoTitle']),
                    html.P("Espesor del agua en la superficie terrestre, medido por satélites GRACE. Se mide en milímetros (mm). Representa cambios en el almacenamiento de agua, lo que puede influir en las precipitaciones futuras.", style=styles['infoText']),
                ], hidden=True)
            ]),
        ])
    ])
])

# Ruta para la raíz "/"
@app.route('/')
def home():
    return "Bienvenido al dashboard de predicción de precipitaciones. Accede a /dash para usar el modelo."

# Callback para generar la predicción automáticamente al cargar la página
@dash_app.callback(
    Output('comparison-graph', 'figure'),
    Output('precipitation-info', 'children'),
    Input('update-button', 'n_clicks')
)
def auto_prediction(n_clicks):
    # Asume que el modelo necesita ciertos datos históricos para predecir
    features = pd.DataFrame({
        'Temperature_A': [27.248285],  # Ejemplo, puedes extraerlos de tus datos
        'Temperature_D': [26.349115],
        'Humidity_A': [70.21221],
        'Humidity_D': [84.62871],
        'Water_Thickness_GRACE': [-8.835976601]
    })

    # Realizar la predicción del siguiente mes
    prediction_kg_m2_s = model.predict(features)[0]  # Predicción en kg m-2 s-1
    prediction_mm_h = prediction_kg_m2_s * 3600  # Conversión a mm/h

    # Obtener la precipitación del mes anterior
    previous_month = historical_df['Data Month'].max() - pd.DateOffset(months=1)
    previous_precipitation = historical_df.loc[historical_df['Data Month'] == previous_month, 'Precipitation']

    previous_precipitation_value = previous_precipitation.values[0] if not previous_precipitation.empty else 0  

    # Crear gráfico 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df['Data Month'], 
                             y=historical_df['Precipitation'],
                             mode='lines+markers',
                             name='Histórico',
                             line=dict(color='blue')))

    future_date = historical_df['Data Month'].max() + pd.DateOffset(months=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[prediction_mm_h], 
                             mode='markers', name='Predicción', 
                             marker=dict(size=10, color='red')))

    fig.update_layout(title='Predicción de Precipitaciones para el Próximo Mes',
                      xaxis_title='Fecha',
                      yaxis_title='Precipitación (mm/h)',
                      template='plotly_white')

    # Texto informativo sobre la precipitación anterior y la pronosticada
    info_text = (
        f"Se pronostica una precipitación de {prediction_mm_h:.2f} mm/h para el próximo mes."
    )

    return fig, info_text

@dash_app.callback(
    Output('content-1', 'hidden'),
    Output('content-2', 'hidden'),
    Output('content-3', 'hidden'),
    Input('toggle-1', 'n_clicks'),
    Input('toggle-2', 'n_clicks'),
    Input('toggle-3', 'n_clicks'),
)
def toggle_content(toggle1, toggle2, toggle3):
    return (toggle1 % 2 == 0, toggle2 % 2 == 0, toggle3 % 2 == 0)

if __name__ == '__main__':
    app.run(debug=True)
