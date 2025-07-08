import streamlit as st
import data_analyse as d_a

URL = "https://www.kaggle.com/datasets/youssefelebiary/air-quality-2024?resource=download"

def exploration():

    st.markdown("### :material/analytics: Analyse Exploratoire des Données")

    st.markdown("##### Dataset Global Air Quality (2024) - 6 Cities")
    st.dataframe(d_a.aqi_data)
    st.info(
        f"""**Nonbre de ligne** : {d_a.aqi_data_shape[0]} \n
**Nombre de colonne** : {d_a.aqi_data_shape[1]} \n
**Nombre de doublon** : {d_a.nb_doublon}"""
    )
    st.link_button(label="En voir plus", url=URL)

    st.markdown("##### Pourcentage de Remplissage de Chaque Variable")
    fig = d_a.barplot_na()
    st.plotly_chart(fig)

    st.markdown("##### Valeur moyenne pour chaque variable par ville")
    st.dataframe(d_a.city_mean)

    st.markdown("##### Valeur moyenne de chaque variable pour chaque mois")
    st.dataframe(d_a.mounth_mean)

    st.markdown("##### Boxplot par colonne")
    selected_col = st.selectbox(
        label="Sélectionner une colonne", 
        options=['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', "PM10", "AQI"]
    )
    st.plotly_chart(d_a.boxplot(selected_col))

    st.markdown("##### Matrice de corrélation des variables")
    selected_city = st.selectbox(
        label="Sélectionner une ville",
        options=["Aucune",'Brasilia', 'Cairo', 'Dubai', 'London', 'New York', 'Sydney']
    )
    selected_city = None if selected_city == "Aucune" else selected_city
    st.plotly_chart(d_a.matrice(selected_city))


def explication():

    st.markdown("")

def ouverture():

    st.markdown("")

pages = {
    "Pages":[
        st.Page(page=exploration,title="Exploration des données",icon=":material/analytics:"),
        st.Page(page=explication,title="Explication de l'AQI",icon=":material/bar_chart_4_bars:"),
        st.Page(page=ouverture,title="Ouverture",icon=":material/euro_symbol:"),
    ]
}

pg = st.navigation(pages=pages)
pg.run()