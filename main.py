from dash import Dash, html, dcc, Output, Input
import pandas as pd
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from shap_plots import ShapExplainer

# Loading the dataset of customers
filepath1 = "https://raw.githubusercontent.com/tcgilles/oc_projet7_dashboard/main/data/customers_data_1.csv"
df = pd.read_csv(filepath1).set_index("SK_ID_CURR").sort_index()

# Types of features
continuous_feat = df.nunique()[df.nunique()>10].index.tolist()
categorical_feat_bar = df.nunique()[(df.nunique()<=10) & (df.nunique()>3)]\
                         .index.tolist()
categorical_feat_pie = df.nunique()[df.nunique()<=3].index.tolist()

# List of customers
customers_list = df.index.tolist()

# Threshold
threshold = 0.658

# Adding a column TARGET
df["TARGET"] = df["SCORE"].apply(lambda x: 0 if x < threshold else 1)

# API url
api_url = 'https://api-home-credit-risk.herokuapp.com/predict'

# Initializing the SHAP explainer
shap_explainer = ShapExplainer()


# Initialize the app - incorporate css
external_stylesheets=[dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


header = dbc.Row(
            dbc.Col(
                html.H1(
                    className="row", 
                    children="HOME CREDIT", 
                    style={
                        "textAlign": "center", 
                        "color": "red",
                        }
                        ),
                width={"size":5, "offset":3}
                ),
            justify="center"
            )

scoring = dbc.Row([       
            dbc.Col([
                html.H6(
                    "N° du client", 
                    style={
                        "fontWeight": "bold", 
                        "textAlign": "center"
                        }
                        ), 
                html.Div(
                    dcc.Dropdown(
                        id="id_client",
                        options=df.index.tolist(), 
                        placeholder="select a customer", 
                        style={"textAlign": "center"}
                        ),
                    style={"textAlign": "center"}
                    ),
                html.Br(),
                html.Div(html.H4(id="statut_credit")),
                html.Br(),
                html.Div(
                    daq.Gauge(
                        id = "jauge",
                        showCurrentValue=True,
                        color={
                            "gradient":True, 
                            "ranges":{
                                "green":[0, threshold-0.25], 
                                "yellow":[threshold-0.25, threshold], 
                                "red":[threshold, 1]
                                }
                                },
                        label="SCORE",
                        min=0, 
                        max=1,
                        )
                    ),
                html.Div([
                    html.Ul(id="score_text"),
                    html.Ul(
                        f"Score critique : {threshold}", 
                        style={
                            "textAlign": "center", 
                            "color": "white", 
                            "margin-top": "-15px"
                            }
                        )], 
                    style={"margin-top": "-35px"}
                    )
            ], 
                width=2, 
                style={"height": "100%"}, 
                align="center"
            ),

            dbc.Col([
                dcc.Slider(5, 15, 1, value=10, id="nb_features_local"),
                dcc.Graph(id="feature_importance_local"),
            ], 
                width=6, 
                style={"height": "100%"}
            ), 

            dbc.Col([
                dcc.Slider(5, 15, 1, value=10, id="nb_features_global"),
                dcc.Graph(id="feature_importance_global"),
            ], 
                width=4, 
                style={"height": "100%"}
            ),
        ])

feature_selection = dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(
                                options=continuous_feat, value="", 
                                placeholder="variable 1", id="feature_1", 
                                style={"textAlign": "center"}
                            ), 
                            width=2,
                        ),

                        dbc.Col(
                            dcc.Dropdown( 
                                placeholder="variable 2", 
                                id="feature_2", 
                                style={"textAlign": "center"}
                            ), 
                            width=2
                        ),
                        
                        dbc.Col(
                            dcc.Dropdown(
                                options = categorical_feat_pie + \
                                            categorical_feat_bar,
                                placeholder="variable 3", 
                                id="feature_3", 
                                style={"textAlign": "center"}
                            ), 
                            width=2
                        ),
                    ], 
                    justify="around"
                    )

axis_scale_selection =  dbc.Row([
                            dbc.Col(
                                dcc.RadioItems(
                                    options=['Linear', 'Log'], 
                                    value='Linear', 
                                    id='crossfilter-xaxis-type',
                                    inline=True,
                                    labelStyle={'marginTop': '5px'},
                                    style={"textAlign": "center"}
                                ), 
                                width=2,
                            ),

                            dbc.Col(
                                dcc.RadioItems(
                                    options=['Linear', 'Log'], 
                                    value='Linear',
                                    inline=True,
                                    id='crossfilter-yaxis-type',
                                    labelStyle={'marginTop': '5px'},
                                    style={"textAlign": "center"}
                                ), 
                                width=2,
                            ),

                            dbc.Col(width=2),
                        ], 
                        justify="around"
                        )

dist_plot_section = dbc.Row([
                        dbc.Col(
                            dcc.Graph(id="dist_feature_1"),
                            width=4,
                            ),
                        dbc.Col(
                            dcc.Graph(id="dist_feature_2"),
                            width=4,
                            ),
                        dbc.Col(
                            dcc.Graph(id="box_feature_3"),
                            width=4,
                            ),
                        ],
                        justify="around"
                        )

bivariate_plot_section = dbc.Row([
                            dbc.Col(
                                dcc.Graph(
                                    id="bivariate_graph", 
                                    style={'height': 900}
                                ),
                                width=8
                            ),

                            dbc.Col([
                                dcc.Dropdown(
                                    options=categorical_feat_pie,
                                    placeholder="variable 4", 
                                    id="feature_4", 
                                    style={"textAlign": "center"}
                                ),
                                dcc.Graph(
                                    style={'height': 865},
                                    id="pie_feature_4"
                                ),  
                            ],
                            width = 4
                            )
                        ])


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
                ], 
                fluid=True,
                )
            ])



def api_call(input_values):
    response = requests.post(api_url, json=input_values).json()
    return response["score"]

@app.callback(
    Output("jauge", "value"),
    Input("id_client", "value"))
def set_value_gauge(unique_id):
    score = 0
    if unique_id in customers_list:
        data = df.copy().loc[unique_id, :]\
                        .fillna("_")\
                        .drop(columns=["SCORE", "TARGET"])\
                        .to_dict()
        score = api_call(data)
    return score


@app.callback(
    Output("statut_credit", "children"),
    Output("statut_credit", "style"),
    Input("id_client", "value"),
    Input("jauge", "value"))
def update_credit_status(unique_id, score):
    if unique_id in customers_list:
        if score < threshold:
            result = "Crédit accordé"
            style = {"color": "green", "textAlign": "center"}
        else:
            result = "Crédit refusé"
            style = {"color": "red", "textAlign": "center"}
    elif unique_id == None:
        result = ""
        style = {}
    else:
        result = "Identifiant incorrect"
        style = {"textAlign": "center", "font-size": 15}
    return result, style


@app.callback(
        Output("score_text", "children"),
        Output("score_text", "style"),
        Input("id_client", "value"),
        Input("jauge", "value"))
def display_score_text(customer_id, score):
    text = "" 
    style = {}
    if customer_id in customers_list:
        text = f"Score client : {score:.3f}"
        if score < threshold - 0.3:
            style = {"fontWeight": "bold", 
                     "textAlign": "center", 
                     "color": "green"}
        elif score < threshold - 0.07:
            style = {"fontWeight": "bold", 
                     "textAlign": "center", 
                     "color": "yellow"}
        elif score < threshold:
            style = {"fontWeight": "bold", 
                     "textAlign": "center", 
                     "color": "orange"}       
        else:
            style = {"fontWeight": "bold", 
                     "textAlign": "center", 
                     "color": "red"}       
    return text, style
    

@app.callback(
        Output("feature_importance_local", "figure"),
        Input("id_client", "value"),
        Input("nb_features_local", "value"))
def plot_feature_importance_local(customer_id, nb_features):
    if customer_id in customers_list:
        background_data = df.copy().drop(columns=["SCORE", "TARGET"])
        fig = shap_explainer.plot_local(background_data, 
                                        customer_id, 
                                        nb_features)
        return fig
    else:
        return {}


@app.callback(
        Output("feature_importance_global", "figure"),
        Input("nb_features_global", "value"))
def plot_feature_importance_global(nb_features):
    background_data = df.copy().drop(columns=["SCORE", "TARGET"])
    fig = shap_explainer.plot_global(background_data, 
                                         nb_features)
    return fig

    

@app.callback(
    Output("feature_2", "options"),
    Output("feature_2", "disabled"),
    Input("feature_1", "value"))
def set_options_feature_2(feature_1):
    if feature_1 in continuous_feat:
        result = continuous_feat.copy()
        result.remove(feature_1)
        return result, False
    else:
        return continuous_feat, True
    

def plot_dist(value, feature, xaxis_type):
    fig = px.histogram(data_frame=df, 
                       x=feature, 
                       color="TARGET",
                       color_discrete_map={0:"blue", 1:"red"},
                       barmode="overlay"
                        )
    fig.add_trace(
        go.Scatter(
                   x=[value, value],
                   y=[0, 100],
                   mode="lines",
                   line=go.scatter.Line(color="yellow"),
                   showlegend=False)
                )
    fig.update_xaxes(type='linear' if xaxis_type == 'Linear' else 'log')
    return fig

@app.callback(
    Output("dist_feature_1", "figure"),
    Output("dist_feature_2", "figure"),
    Output("bivariate_graph", "figure"),
    Input("id_client", "value"),
    Input("feature_1", "value"),
    Input("feature_2", "value"),
    Input("crossfilter-xaxis-type", "value"),
    Input("crossfilter-yaxis-type", "value"))
def plot_continuous_features(unique_id, feature1, feature2, 
                             xaxis_type, yaxis_type):
    
    fig1, fig2, fig3 = {}, {}, {}

    if unique_id in customers_list:
        if feature1:
            value1 = df.loc[unique_id, feature1]
            fig1 = plot_dist(value1, feature1, xaxis_type)

        if feature2:
            value2 = df.loc[unique_id, feature2]
            fig2 = plot_dist(value2, feature2, yaxis_type)

        if feature1 and feature2:        
            data = df.copy().drop(index=unique_id)
            fig3 = px.scatter(
                data, x=feature1, y=feature2, 
                color=data["SCORE"],
                color_continuous_scale="jet",
                range_color=[0, 1],
                hover_name=data.index.to_numpy(),
            ) 
            fig3.add_trace(
                go.Scatter(
                        x=[df.loc[unique_id, feature1]],
                        y=[df.loc[unique_id, feature2]],
                        mode="markers",
                        marker=go.scatter.Marker(color="green", 
                                                 size=15, 
                                                 symbol="x"),
                        name=f"client n° {unique_id}",
                        showlegend=True
                ),
            )
            fig3.update_xaxes(
                title=feature1,
                type='linear' if xaxis_type == 'Linear' else 'log'
            )
            fig3.update_yaxes(
                title=feature2, 
                type='linear' if yaxis_type == 'Linear' else 'log'
            )
            fig3.update_layout(legend=dict(orientation="h"))

            # Re-order the data:
            fig3.data = (fig3.data[1], fig3.data[0])

    return fig1, fig2, fig3


@app.callback(
    Output("box_feature_3", "figure"),
    Input("id_client", "value"),
    Input("feature_3", "value"))
def plot_box(customer_id, feature):
    fig = {}
    if customer_id in customers_list and feature :
        value = df.loc[customer_id, feature]
        fig = px.box(
            df,
            x="SCORE", 
            y=feature, 
            orientation="h",
            color = df[feature].apply(lambda x: f"{value:.1f}" 
                                             if x==value else "other"),
            color_discrete_map = {f"{value:.1f}":"blue", "other":"black"}
        )
    return fig

@app.callback(
    Output("pie_feature_4", "figure"),
    Input("id_client", "value"),
    Input("feature_4", "value"))
def plot_pie(customer_id, feature):
    fig = {}

    if customer_id in customers_list and feature:
        specs = [[{'type' : 'domain'}], [{'type' : 'domain'}]]
        titles = ['Tous les clients', 
                  'Pourcentage de clients défectueux par catégorie']
        fig = make_subplots(rows = 2, cols = 1, specs = specs, 
                            subplot_titles = titles)

        # plotting overall distribution of the category
        fig.add_trace( 
            go.Pie(
                values = df["SCORE"], 
                labels = df[feature], 
                hole = 0.3, 
                textinfo = 'label+percent', 
                textposition = 'inside'
            ), 
            row = 1, 
            col = 1
        )

        percentage_defaulter_per_category = \
            df[feature][df.TARGET == 1].value_counts() * 100 \
                                            / df[feature].value_counts()
        percentage_defaulter_per_category = \
                percentage_defaulter_per_category.dropna().round(2)

        fig.add_trace( 
            go.Pie(
                values=percentage_defaulter_per_category, 
                labels=percentage_defaulter_per_category.index, 
                hole=0.3, 
                textinfo='label+value', 
                hoverinfo='label+value'
            ), 
            row=2, 
            col=1
        )

        fig.update_layout(
            title = f'Distribution de {feature}', 
            showlegend = False
        )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)