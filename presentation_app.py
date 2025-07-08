import streamlit as st
import data_analyse as d_a

def exploration():

    pass

def explication():

    pass

def ouverture():

    pass

pages = {
    "Pages":[
        st.Page(page=exploration,title="Exploration des données",icon=":material/:"),
        st.Page(page=explication,title="Explication de l'AQI",icon=":material/:"),
        st.Page(page=ouverture,title="Ouverture",icon=":material/:"),
    ]
}

pg = st.navigation(pages=pages)
pg.run()