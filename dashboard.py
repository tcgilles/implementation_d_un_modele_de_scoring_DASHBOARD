import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# Loading the dataset already preprocessed
df = pd.read_csv("../input/data_cleaned.csv", index_col="SK_ID_CURR")

# Deleting the initial target column
# df.drop(columns=["TARGET"], inplace=True)

# Types of features
continuous_feat = df.nunique()[df.nunique()>15].index.tolist()
categorical_feat_bar = df.nunique()[(df.nunique()<=15) & (df.nunique()>3)].index.tolist()
categorical_feat_pie = df.nunique()[df.nunique()<=3].index.tolist()

# List of customers
customers_list = df.index.tolist()

# Initialize the app - incorporate css
external_stylesheets=[dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        
    ],
    style=SIDEBAR_STYLE,
)

content = dbc.Container(
    [
        
    ],
    fluid=True, style=CONTENT_STYLE
)

app.layout = html.div([sidebar, content])

if __name__ == '__main__':
    app.run_server(debug=True)