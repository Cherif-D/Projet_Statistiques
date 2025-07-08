import pandas as pd
import plotly.graph_objects as go
from typing import Literal

#IMPORT DU DATASET DE L INDEX DE QUALITE DE L AIR DE 6 VILLES
aqi_data = pd.read_csv("Air_Quality.csv", header=0, index_col=0)
aqi_data_shape = aqi_data.shape
aqi_data_head = aqi_data.head(10)
aqi_data_tail = aqi_data.tail(10)
nb_doublon = len(aqi_data[aqi_data.duplicated() == True])

#% DE REMPLISSAGE DE CHAQUE VARIABLE
def barplot_na() -> go.Figure:

    fig = go.Figure()

    nb_na = {}
    na_percent = {}

    for col in aqi_data.columns:

        x = len(aqi_data[col])
        y = len(aqi_data[col].dropna())
        nb_na[col] = x-y
        na_percent[col] = (nb_na[col]/x)

        fig.add_trace(
            trace=go.Bar(
                x=[col],
                y=[(1-na_percent[col])*100],
                marker_color="red"
                )
        )
    
    fig.update_layout(
        showlegend=False,
        yaxis_title="%"
    )

    return fig

#VILLES CONCERNEES
data_city = aqi_data["City"].unique()

#VALEUR MOYENNE POUR CHAQUE VARIABLE PAR VILLE
city_mean = aqi_data.groupby("City").mean(numeric_only=True)

def boxplot(col:Literal['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', "PM10", "AQI"]) -> go.Figure:

    fig = go.Figure()

    for city in aqi_data["City"].unique():

        df = aqi_data[col][aqi_data["City"] == city]

        fig.add_trace(
            trace=go.Box(
                y=df,
                name=city,
                boxpoints="outliers",
                boxmean=True
            )
        )

    fig.update_layout(
            yaxis_title=col,
            xaxis_title="Ville",
            showlegend=False,
            boxmode="group",
        )

    return fig

#MATRICE DE CORRELATION DES VARIABLES PAR VILLE
def matrice(city:Literal['Brasilia', 'Cairo', 'Dubai', 'London', 'New York', 'Sydney']=None) -> go.Figure:

    drop_col = [col for col in ["City", "Mois"] if col in aqi_data.columns]

    if city:

        corr = aqi_data[aqi_data["City"] == city].drop(columns=drop_col).corr()

    else :

        corr = aqi_data.drop(columns=drop_col).corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corrélation")
            )
        )

    return fig

#MOYENNE DES VARIABLES PAR MOIS
aqi_data.index = pd.to_datetime(aqi_data.index)
aqi_data["Mois"] = aqi_data.index.month_name()
mounth_mean = aqi_data.groupby("Mois").mean(numeric_only=True)