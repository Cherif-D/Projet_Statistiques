import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_white
from typing import Literal, List, Optional, Union, Tuple
from datetime import datetime
import time

#-----------------------------------------------------------------------------
# ANALYSE EXPLORATOIRE DES DONNEES
# ----------------------------------------------------------------------------

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

def boxplot(
        col:Literal['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', "PM10", "AQI"]
    ) -> go.Figure:

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

def data_filter(col=None, city=None):

    data = aqi_data.copy()
    data.index = pd.to_datetime(data.index)

    if col:

        for polluant in col:

            polluant = polluant.split(" ",1)[1]
            data.drop(columns=polluant,inplace=True)

    if city:

        data = data[data["City"] == city]
    
    #DROP VARIABLE CATEGORIELLE
    drop_col = [col for col in ["City", "Mois"] if col in data.columns]
    data.drop(columns=drop_col,inplace=True)

    #80% DE VALEUR NaN 
    data.drop(columns=["CO2"],inplace=True)

    return data

#DISPERTION ET DISTRIBUTION POUR CHAQUE VARIABLE
def disp_distr(col) -> go.Figure:

    data = data_filter(col)

    plt.figure(figsize=(12,6))
    sns.pairplot(data,corner=True,plot_kws={"s":2})
    
    return plt.gcf()

#MATRICE DE CORRELATION DES VARIABLES PAR VILLE
def matrice(
        city:Literal[
            'Brasilia', 'Cairo', 'Dubai', 'London', 'New York', 'Sydney'
        ]=None
    ) -> go.Figure:

    drop_col = [col for col in ["City", "Mois"] if col in aqi_data.columns]

    if city:

        corr = aqi_data[aqi_data["City"] == city].drop(
            columns=drop_col
        ).corr()

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
aqi_data.drop(columns=["Mois"],inplace=True)

# ----------------------------------------------------------------------------
# EXPLICATION DE L AQI
# ----------------------------------------------------------------------------

polluants = Literal[
    "Sans PM10", "Sans PM2.5", "Sans NO2", "Sans CO", "Sans O3", "Sans SO2"
]

villes = Literal[
    'Brasilia', 'Cairo', 'Dubai', 'London', 'New York', 'Sydney'
]

# VIF
def vif_aqi_variable(
        col:Optional[List[polluants]]=None,
        city:Optional[villes]=None
    ) -> pd.DataFrame:

    data = data_filter(col, city)
    
    X = data.iloc[:,:-1]
    X_const = sm.add_constant(X)

    vif = pd.DataFrame({
        "variable": X_const.columns,
        "VIF": [variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])]
    })

    return vif

#REGRESSION LINEAIRE
def reg_line_aqi(
        col:Optional[List[polluants]]=None,
        city:Optional[villes]=None,
    ) -> sm.regression.linear_model.RegressionResultsWrapper:

    data = data_filter(col, city)

    X = data.iloc[:,:-1]
    X = sm.add_constant(X)
    y = data.iloc[:,-1]

    model = sm.OLS(y, X).fit()

    return model

#HYPOTHESES
def reg_hypothese(
        model:sm.regression.linear_model.RegressionResultsWrapper,
        get:Literal["resid_hist", "acf","resid_homo","het_white"]
    ) -> Union[go.Figure, plt.Figure] :
    
    residuals = model.resid
    fitted = model.fittedvalues

    if get == "resid_hist":

        plt.figure(figsize=(12,6))
        x = np.linspace(residuals.min(), residuals.max(), 100)
        mu = residuals.mean()
        sigma = residuals.std()
        y = (
            (1 / (sigma * np.sqrt(2 * np.pi))) * 
            np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        )
        sns.histplot(x=residuals,stat="density",kde=True,bins=100)
        plt.plot(x, y, "--",label="Densité normale théorique",color="red")
        plt.xlabel("")
        plt.legend()

        return plt.gcf()
    
    if get == "acf":

        plot_acf(residuals, lags=40)
        plt.title("")

        return plt.gcf()
    
    if get == "resid_homo":

        plt.figure(figsize=(12,6))
        plt.scatter(x=fitted,y=residuals)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)

        return plt.gcf()

    if get == "het_white":

        class Het():

            def __init__(self):

                exog = model.model.exog
                white_test = het_white(residuals, exog)
                stat, p_value, f_stat, f_p_value = white_test

                self.stat = stat
                self.p_value = p_value
                self.f_stat = f_stat
                self.f_p_value = f_p_value

        het0 = Het()

        return het0
    
# ----------------------------------------------------------------------------
# Explication de PM2.5
# ----------------------------------------------------------------------------

map_target_threshold = {    "PM10":40, "PM2.5":20, "NO2":90, 
                            "O3":100, "SO2":200                 }

map_temporal = {    "Aucune"        :   None, 
                    "Saisonière"    :   "Season",
                    "Mensuelle"     :   "Month",
                    "Journalière"   :   "Day"   }
j_30 = pd.Timedelta(days=30)

def get_season_num(date:datetime) -> int:
    month = date.month
    day = date.day
    if (month == 12 and day >= 21) or (1 <= month <= 2) or (month == 3 and day < 20):
        return 1  # Winter
    elif (month == 3 and day >= 20) or (4 <= month <= 5) or (month == 6 and day < 21):
        return 2  # Spring
    elif (month == 6 and day >= 21) or (7 <= month <= 8) or (month == 9 and day < 22):
        return 3  # Summer
    else:
        return 4  # Autumn


#REGRESSION LOGISTIQUE
def reg_log(
        target:polluants,
        start:datetime,
        end:datetime,
        col:Optional[List[polluants]]=None,
        city:Optional[villes]=None,
        binaire:Optional[bool]=None,
        temp:Optional[Literal["Season","Month","Day"]]=None,
        balanced:bool=False
    ) -> Tuple[ BinaryResultsWrapper, Optional[int], pd.Series, 
                np.ndarray, np.ndarray  ]:

    target = "Sans " + target
    col.remove(target) if target in col else col
    target = target.split(" ",1)[1]
    start = pd.Timestamp(start).tz_localize("UTC")
    end = pd.Timestamp(end).tz_localize("UTC")

    def set_model(binaire:Optional[bool]=None) -> Tuple[
        BinaryResultsWrapper, Optional[int], np.ndarray, np.ndarray
    ]:
        threshold = None if binaire == (None or False) else 40
        list_threshold = [20, 40, 50, 100, 150]
        y_threshold = map_target_threshold[target] 
        e = True
        x = -1
        while e == True:
            data = data_filter(col, city)
            mask = (data.index>=start) & (data.index<=end)
            data = data.loc[mask]
            if temp:
                if temp == "Season":
                    data["Season"] = [get_season_num(d) for d in data.index]
                elif temp == "Month":
                    data["Month"] = data.index.month
                else:
                    data["Day"] = data.index.weekday
                data = pd.get_dummies(  data=data, columns=[temp],
                                        drop_first=True ).astype(float)
            if balanced:
                data[target] = np.where(data[target]>y_threshold, 0, 1)
                if len(data) > 10000:
                    count = data[target].value_counts().min()
                    replace = False
                else:
                    count = data[target].value_counts().max()
                    replace = True
                data = (
                    data.groupby(target)
                    .apply(
                        lambda x: x.sample( n=count, random_state=100,
                                            replace=replace )
                        )
                    .reset_index(drop=True)
                )
            X = data.copy()
            X = X.drop(columns=["AQI", target])
            try:
                if binaire:
                    if "PM10" in X.columns and binaire==True:
                        if x<= 4 :
                            threshold = (
                                list_threshold[x] if x >= 0 else threshold
                            )
                        else:
                            threshold -= 1
                        X["PM10"] = np.where(X["PM10"]>threshold, 0, 1)
                X = sm.add_constant(X, has_constant="add")
                y = data.loc[:,target]
                if not balanced:
                    y = np.where(y>y_threshold, 0, 1)
                class_0 = y[y==0]
                class_1 = y[y==1]
                model = sm.Logit(y, X).fit()
                e = False
            except :
                x += 1
                continue

        return model, threshold, class_0, class_1
    
    model, threshold, class_0, class_1 = set_model(binaire)

    return model, threshold, np.exp(model.params), class_0, class_1