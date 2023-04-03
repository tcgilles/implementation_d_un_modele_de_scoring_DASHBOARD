from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Loading the dataset already preprocessed
df = pd.read_csv("../input/data_cleaned.csv", 
                 index_col="SK_ID_CURR", 
                 nrows=1000)

# Loading the dataset source and limiting its values to the customers in df
df_source = pd.read_csv("../input/application_train.csv", 
                        index_col="SK_ID_CURR", 
                        nrows=2000)
df_source = df_source.loc[df.index, :].drop(columns=["TARGET"])
df_source = df[["TARGET"]].join(df_source)

# Types of features
continuous_feat = df.nunique()[df.nunique()>20].index.tolist()
categorical_feat_bar = df.nunique()[(df.nunique()<=20) & (df.nunique()>3)].index.tolist()
categorical_feat_pie = df.nunique()[df.nunique()<=3].index.tolist()

# List of customers
customers_list = df.index.tolist()

# probabilities
np.random.seed(seed=42)
df["probability"] = np.random.uniform(0, 1, len(customers_list))

# Threshold
threshold = 0.65

# Initialize the app - incorporate css
external_stylesheets=[dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)


header = dbc.Row(
            dbc.Col(
                html.H1(className="row", 
                        children="HOME CREDIT", 
                        style={"textAlign": "center", 
                               "color": "red", 
                               "fontsize": 30}),
                        width={"size":5, "offset":3}),
                justify="center"
                )


scoring = dbc.Row([       
            dbc.Col([
                html.H6("N° du client", style={"fontWeight": "bold", 
                                               "textAlign": "center"}), 
                html.Div(
                    dcc.Input(id="id_client", 
                              value=100002, 
                              type="number", 
                              style={"textAlign": "center"}),
                    style={"textAlign": "center"}
                        ),
                html.Br(),
                html.Div(html.H4(id="statut_credit")),
                html.Br(),
                html.Div(
                    daq.Gauge(id = "jauge",
                            showCurrentValue=True,
                            color={"gradient":True, 
                                "ranges":{"green":[0, threshold-0.25], 
                                          "yellow":[threshold-0.25, threshold], 
                                          "red":[threshold, 1]
                                            }
                                    },
                            label="SCORE",
                            min=0, max=1,)
                        )
            ], width=3),
            dbc.Col([
                dcc.Dropdown(options=["Tous les clients", "Clients acceptés",
                                      "Cliens rejetés"], 
                             value="Tous les clients",
                             placeholder="Base de comparaison",
                             clearable=False,
                             id="base_importance_locale",
                            ),
                html.Br(),
                dcc.Graph()
            ], width={"size": 6, "offset": 3})
        ])


feature_selection = dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(options=continuous_feat, 
                                        value="", 
                                        placeholder="variable 1", 
                                        id="feature_1"), 
                            width=2,),
                        dbc.Col(
                            dcc.Dropdown(value="", 
                                        placeholder="variable 2", 
                                        id="feature_2"), 
                            width=2),
                            ], justify="around"
                            )

axis_scale_selection =  dbc.Row([
                            dbc.Col(
                                dcc.RadioItems(options=['Linear', 'Log'], 
                                            value='Linear', 
                                            id='crossfilter-xaxis-type',
                                            inline=True,
                                            labelStyle={'marginTop': '5px'}), 
                                width=2,),
                            dbc.Col(
                                dcc.RadioItems(options=['Linear', 'Log'], 
                                            value='Linear',
                                            inline=True,
                                            id='crossfilter-yaxis-type',
                                            labelStyle={'marginTop': '5px'}), 
                                width=2,),
                                ], justify="around"
                                )

dist_plot_section = dbc.Row([
                        dbc.Col(dcc.Graph(id="dist_feature_1")),
                        dbc.Col(dcc.Graph(id="dist_feature_2")),
                            ])

bivariate_plot_section = dbc.Row(
                            dbc.Col(dcc.Graph(id="bivariate_graph", 
                                              style={'height': 600}),
                                    ),
                                )


app.layout = html.Div([
                dbc.Container([
                    header, 
                    dbc.Row(dbc.Col(html.Hr(), width=12)),
                    scoring,
                    dbc.Row(dbc.Col(html.Hr(), width=12)),
                    feature_selection,
                    axis_scale_selection,
                    dist_plot_section,
                    html.Br(),
                    bivariate_plot_section,   
                ], fluid=True,)
            ])


@app.callback(
    Output(component_id="statut_credit", component_property="children"),
    Output(component_id="statut_credit", component_property="style"),
    Input(component_id="id_client", component_property="value")
)
def update_credit_status(unique_id):
    if unique_id in customers_list:
        if df.loc[unique_id, "probability"] < threshold:
            result = "Crédit accordé"
            style = {"color": "green", "textAlign": "center"}
        else:
            result = "Crédit refusé"
            style = {"color": "red", "textAlign": "center"}
    else:
        result = "Identifiant incorrect"
        style={}
    return result, style


@app.callback(
    Output(component_id="jauge", component_property="value"),
    Input(component_id="id_client", component_property="value")
)
def set_value_gauge(unique_id):
    if unique_id not in customers_list:
        return 1
    else:
        return df.loc[unique_id, "probability"]


@app.callback(
    Output(component_id="feature_2", component_property="options"),
    Output(component_id="feature_2", component_property="disabled"),
    Input(component_id="feature_1", component_property="value")
)
def set_options_feature_2(feature_1):
    if feature_1 in continuous_feat:
        result = continuous_feat.copy()
        result.remove(feature_1)
        return result, False
    else:
        return continuous_feat, True
    

def plot_dist(value, feature):
    fig = px.histogram(data_frame=df, 
                       x=feature, 
                       color="TARGET", 
                       color_discrete_map={0:"blue", 1:"red"},
                        )
    fig.add_trace(
        go.Scatter(
                   x=[value, value],
                   y=[0, 20],
                   mode="lines",
                   line=go.scatter.Line(color="yellow"),
                   showlegend=False)
                )
    return fig


@app.callback(
    Output(component_id="dist_feature_1", component_property="figure"),
    Input(component_id="id_client", component_property="value"),
    Input(component_id="feature_1", component_property="value"),
)
def plot_dist_feature_1(unique_id, feature):
    if (unique_id in customers_list) and (feature != ""):
        value = df.loc[unique_id, feature]
        return plot_dist(value, feature)
    else:
        return {}


@app.callback(
    Output(component_id="dist_feature_2", component_property="figure"),
    Input(component_id="id_client", component_property="value"),
    Input(component_id="feature_2", component_property="value"),
)
def plot_dist_feature_2(unique_id, feature):
    if (unique_id in customers_list) and (feature != ""):
        value = df.loc[unique_id, feature]
        return plot_dist(value, feature)
    else:
        return {}

@app.callback(
    Output(component_id="bivariate_graph", component_property="figure"),
    Input(component_id="id_client", component_property="value"),
    Input(component_id="feature_1", component_property="value"),
    Input(component_id="feature_2", component_property="value"),
    Input(component_id="crossfilter-xaxis-type", component_property="value"),
    Input(component_id="crossfilter-yaxis-type", component_property="value")
)
def plot_bivariate_graph(unique_id, feature1, feature2, xaxis_type, yaxis_type):
    if (unique_id in customers_list) and (feature1 != "") and (feature2 != ""):
        data = df.copy().drop(index=unique_id)
        fig = px.scatter(data, x=feature1, y=feature2, 
                         color=data["TARGET"].map({0:"0", 1:"1"}),
                         hover_name=data.index.to_numpy(), 
                         color_discrete_map={"0":"blue", "1":"red"}, 
                         labels={"color": "Crédit rejeté :"}) 
        fig.add_trace(
            go.Scatter(
                    x=[df.loc[unique_id, feature1]],
                    y=[df.loc[unique_id, feature2]],
                    mode="markers",
                    marker=go.scatter.Marker(color="yellow", size=10, symbol="x"),
                    showlegend=False)
                    )
        fig.update_xaxes(title=feature1,
                         type='linear' if xaxis_type == 'Linear' else 'log')
        fig.update_yaxes(title=feature2,
                         type='linear' if yaxis_type == 'Linear' else 'log')
        fig.update_layout(legend=dict(orientation="h"))
        return fig
    else:
        return {}


if __name__ == '__main__':
    app.run_server(debug=True)