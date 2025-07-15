import streamlit as st
import data_analyse as d_a
from typing import Literal, Tuple, List

URL = """https://www.kaggle.com/datasets/youssefelebiary/
air-quality-2024?resource=download"""

def exploration():

    st.markdown(    "### :material/data_exploration: "
                    "Analyse Exploratoire des Données"  )

    st.markdown("##### Dataset Global Air Quality (2024) - 6 Cities")
    st.dataframe(d_a.aqi_data)
    st.info(
        f"**Nonbre de ligne** : {d_a.aqi_data_shape[0]}"
        f"\n\n**Nombre de colonne** : {d_a.aqi_data_shape[1]}"
        f"\n\n**Nombre de doublon** : {d_a.nb_doublon}"
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

    st.markdown("### Pairplots")
    sup_col = st.multiselect(
        label="Sélectionner une/des variable(s) à supprimer",
        options=[
            "Sans PM10", "Sans PM2.5", 
            "Sans NO2", "Sans CO", 
            "Sans O3", "Sans SO2"
        ],
        max_selections=5,
        default=["Sans PM10", "Sans NO2", "Sans CO","Sans SO2", "Sans O3"]
    )
 
    st.pyplot(d_a.disp_distr(sup_col),use_container_width=True)

    st.markdown("##### Matrice de corrélation des variables")
    selected_city = st.selectbox(
        label="Sélectionner une ville",
        options=[
            "Aucune",'Brasilia',
            'Cairo', 'Dubai',
            'London', 'New York',
            'Sydney'
            ]
    )
    selected_city = None if selected_city == "Aucune" else selected_city
    st.plotly_chart(d_a.matrice(selected_city))

def vif_presentation(presentation:Literal[1,2]) -> Tuple[List[str],str]:

    col1, col2 = st.columns(2)

    options = [
        "Sans PM10", "Sans PM2.5", "Sans NO2", 
        "Sans CO", "Sans O3", "Sans SO2"
    ]
    default = "Sans PM10"
    max_selections = 5

    with col1:

        del_col = st.multiselect(
            label="Sélectionner une/des variable(s) à supprimer",
            options=options,
            max_selections=max_selections,
            default=default
        )

    with col2:

        city = st.selectbox(
            label="   ",
            options=[
                "Toutes les villes", "Brasilia",
                'Cairo', 'Dubai', 'London', 
                'New York', 'Sydney'
            ],
        )
    
    st.markdown("##### Facteur d'inflation de la variance")

    city = None if city == "Toutes les villes" else city

    vif = d_a.vif_aqi_variable(del_col,city)
    st.dataframe(vif)

    return del_col, city


def prevision():

    st.markdown("### :material/straighten: Prédiction de l'AQI")

    del_col, city = vif_presentation(1)

    st.markdown("##### Régression linéaire")
    model = d_a.reg_line_aqi(del_col, city)
    st.write(model.summary())
    if st.toggle(
        label=":blue-badge[:material/info: Afficher l'analyse des résidus]"
        ):
        st.markdown("##### Histogramme des résidus")
        st.pyplot(d_a.reg_hypothese(model,"resid_hist"))
        st.info(f"**Moyenne des résidus :** {model.resid.mean()}")
        st.markdown("##### Autocorrélation des résidus")
        st.pyplot(d_a.reg_hypothese(model,"acf"))
        st.markdown("##### Homoscédasticité")
        het0 = d_a.reg_hypothese(model,"het_white")
        st.info(
            f"**stat** : {het0.stat}\n\n"
            f"**p_value** : {het0.p_value}\n\n"
            f"**f_stat** : {het0.f_stat}\n\n"
            f"**f_p_value** : {het0.f_p_value}\n\n"
        )
        st.pyplot(d_a.reg_hypothese(model,"resid_homo"))
    st.info(
        "**Linéarité :** ✓ \n\n"
        "**Distribution normale des résidus :** X \n\n"
        "**Moyenne des résidus nulles :** ✓ \n\n"
        "**Non autocorrélation des résidus :** X\n\n"
        "**Homoscédasticité :** X"
    )

def explication():

    st.markdown("### :material/search: Explication de PM2.5")
    target = st.segmented_control(  label="Choisir la target",
                                    options=d_a.map_target_threshold.keys(),
                                    selection_mode="single",
                                    default="PM2.5"   )
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(  label="Date de début",
                                value=d_a.aqi_data.index.min(),
                                min_value=d_a.aqi_data.index.min(),
                                max_value=d_a.aqi_data.index.max()-d_a.j_30   )
    with col2:
        end = st.date_input(    label="Date de fin",
                                value=d_a.aqi_data.index.max(),
                                min_value=start+d_a.j_30,
                                max_value=d_a.aqi_data.index.max()    )
    del_col, city = vif_presentation(2)
    st.markdown("##### Régression logistique")
    binaire = None
    def show_treshold(threshold:int):

        st.badge(  
            color="blue", 
            icon=":material/info:",
            label=  f"Threshold = {threshold}, "
                    f"if '{target}' > {threshold}, '{target}' = 0 else 1"
        )

    temporal = st.selectbox(    label="Ajout d'une variable temporelle",
                                options=[   "Aucune",
                                            "Saisonière", 
                                            "Mensuelle",
                                            "Journalière"    ],
                                placeholder="Aucune"    )
    temporal = d_a.map_temporal[temporal]
    if ("Sans PM10" not in del_col) and (target != "PM10"):
        binaire = st.selectbox( 
            label="Transformer la variable PM10 en binaire",
            options=["False","True"],
            placeholder="False"
        )
        binaire = True if binaire == "True" else False
    balanced =  st.toggle(label="Équilibrer les classes")
    model, threshold, model_params, c0, c1 = d_a.reg_log(   target=target,
                                                            start=start,
                                                            end=end,
                                                            col=del_col,
                                                            city=city,
                                                            binaire=binaire,
                                                            temp=temporal,
                                                            balanced=balanced)
    if binaire == True:
        show_treshold(threshold)
    
    st.write(model.summary())
    st.warning( f"**Nombre de valeur dans la classe 0 :** {len(c0)}\n\n"
                f"**Nombre de valeur dans la classe 1 :** {len(c1)}"    )
    st.dataframe(model_params)


def ouverture():
    
    st.markdown(
        "### :material/euro_symbol: "
        "Impact de l'activité économique sur l'AQI"
    )

pages = {
    "Pages":[
        st.Page(
            page=exploration,
            title="Exploration des données",
            icon=":material/data_exploration:"
        ),
        st.Page(
            page=prevision,
            title="Prédiction de l'AQI",
            icon=":material/straighten:"
        ),
        st.Page(
            page=explication,
            title="Explication de PM2.5",
            icon=":material/search:"
        ),
        st.Page(
            page=ouverture,
            title="Ouverture",
            icon=":material/euro_symbol:"
        ),
    ]
}

pg = st.navigation(pages=pages)
pg.run()