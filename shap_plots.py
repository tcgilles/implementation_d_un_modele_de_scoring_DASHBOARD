import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib

# Threshold
threshold = 0.658

class ShapExplainer:
    def __init__(self) -> None:
        self.model = joblib.load("model.pkl")
        self.explainer = shap.TreeExplainer(self.model)

    
    def plot_global(self, 
                    background_data:pd.core.frame.DataFrame,
                    max_display:int):
        # Get the absolute values of the shap values
        shap_values = self.explainer.shap_values(background_data)
        shap_values = np.abs(shap_values[1])

        # Get the average shap value of each feature and sort them
        shap_values = pd.DataFrame(
            shap_values, 
            index=background_data.index,
            columns=background_data.columns
            )
        shap_values = pd.DataFrame({
            "feature": shap_values.mean(0).index, 
            "shap_values": shap_values.mean(0).values
                    })
        shap_values = shap_values.sort_values("shap_values").reset_index(drop=True)

        # Select only the top max_display features
        X = shap_values.iloc[-max_display:, :]

        # Plots the barplot of average shap values for the top max_display
        # features
        fig = px.bar(data_frame=X, 
                     x = "shap_values",
                     y = "feature",
                     title = "Importance globale des variables",
                     text_auto = ".3f",
                     labels = {"feature": "Variables", 
                              "shap_values": "Mean shap value (Log odds)"}
                    )
        return fig
    

    def plot_local(self, 
                   background_data:pd.core.frame.DataFrame, 
                   idx:int, 
                   max_display:int):
        # Get the shap values of the sample
        shap_values = self.explainer(background_data.loc[idx, :]\
                                     .to_numpy().reshape(1,-1))

        # Get the base score of the model and the sample final shap value
        raw_base_score = self.explainer.expected_value[1] # E(f(x))
        raw_customer_score = self.explainer.expected_value[1] \
                            + shap_values.values[:,:,1].sum() # f(x)

        # Store the shap values and the features values of the sample
        values = shap_values[0,:,1].values
        data = shap_values[0,:,1].data

        # Create a DataFrame with the 2 above arrays
        df_values = pd.DataFrame({"features": background_data.columns, 
                                  "shap_values": values, "data": data})
        
        # Sort the features by their absolute shap value
        df_values["abs_shap_values"] = np.abs(df_values["shap_values"])
        df_values = df_values.sort_values("abs_shap_values")\
                             .reset_index(drop=True)\
                             .drop(columns="abs_shap_values")

        # Select only the top max_display features
        Y = df_values.iloc[-max_display:, :].reset_index(drop=True)
        Y["features"] = [f"{val:.2f} = {feat}" for (val, feat) in 
                         Y[["data", "features"]].values]
        
        # Add a line in the DataFrame for the sum of shap values of 
        # the reamining features
        sum_others = df_values.iloc[:-max_display, 1].sum()
        name = f"Sum of {len(df_values) - max_display} other features"
        Y = pd.concat([pd.DataFrame([[name, sum_others, np.nan]], 
                                    columns=Y.columns), Y])
        
        # Cumsum of shap values
        Y["shap_cumsum"] = Y["shap_values"].cumsum() + raw_base_score
        min_ = min(raw_base_score, Y["shap_cumsum"].min())
        max_ = max(raw_base_score, Y["shap_cumsum"].max())

        # Waterfall plot of the shap values
        fig = go.Figure( go.Waterfall(
            x = Y["shap_values"],
            y = Y["features"],
            base = raw_base_score,
            text = Y["shap_values"].round(3),
            orientation = "h",
            measure = ["relative"]*len(Y),
            decreasing = {"marker":{"color":"green"}},
            increasing = {"marker":{"color":"red"}}
            )
        )

        # Add the title
        fig.update_layout(
            title = dict(text="Variables influençant le plus le score du client", 
                         x=0.5),
        )

        # Add ticks and title on the x-axis
        fig.update_xaxes(
            title = dict(text = "Shap value (Log odds)"),
            tickangle = 75,
            tickmode="array",
            tickvals=[raw_base_score, raw_customer_score] \
                    + list(np.arange(min_+0.05, max_-0.05, 0.2)),
            ticktext=[f"f(moy)={raw_base_score:.3f}", 
                      f"f(client)={raw_customer_score:.3f}"] \
                   + [str(round(i,1)) for i in np.arange(min_, max_, 0.2)]
        )

        # Add lines on the y-axis
        fig.update_yaxes(griddash = "solid")

        return fig