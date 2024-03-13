import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#To run this file, use the following command in the terminal:
#streamlit run ./Streamlit/app.py
#Rasmus: cd C:\Users\rasmu\OneDrive\Skrivebord\4 sem\BI-Exam-COVID\


import streamlit as st
from streamlit_option_menu import option_menu


import pandas as pd
import numpy as np
import seaborn as sb
import folium

from PIL import Image



st.set_page_config(
    page_title="Covid-19 Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={        
    }
)


st.title("Streamlit Exam Project 2024")

# custom css
st.markdown("""
<style>
    .reduce-margin {
        margin-top: -5px;
        margin-bottom: -10px;
    }
    .small-font {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Introduktion
st.markdown("""
<div class='reduce-margin intro-text'>
    Dear reader,<br><br>
    In an era where data-driven decision making is paramount, our project presents a comprehensive analysis aimed at enhancing pandemic preparedness and response. Utilizing extensive data from Our World in Data, we delve into lessons learned from the COVID-19 pandemic, identifying opportunities for policy improvement and more efficient crisis management strategies.<br><br>
    This Streamlit dashboard offers an interactive platform for exploring various data dimensions, revealing insights into the effectiveness of different containment measures, the impact on healthcare systems, and the socio-economic consequences of pandemics. Our goal is to provide a tool that can inform future policy decisions, ensuring a more resilient societal structure against potential health crises.<br><br>
</div>
""", unsafe_allow_html=True)

# Formål med projektet
st.markdown("""
<div class='reduce-margin intro-text'>
    <strong>Purpose of the Project:</strong><br>
    In this Streamlit dashboard, our main goal is to explore and visualize the relationship between various factors and COVID-19 statistics to challenge common perceptions and inform future policy. Through in-depth analysis supported by a series of graphs, we aim to investigate three key hypotheses:<br>
    1. There is no correlation between a country's GDP and the number of COVID-19 cases.<br>
    2. Countries with a higher number of COVID-19 cases tend to have higher vaccination rates per hundred people.<br>
    3. Not all countries are equally exposed to the risk of COVID-19 infection.<br><br>
    By presenting data-driven insights on these hypotheses, we hope to provide valuable knowledge that could help in crafting more effective health policies and responses in the face of future pandemics. Our analysis is intended to empower policymakers, including you, by providing a nuanced understanding of the factors that influence pandemic dynamics and public health outcomes.<br><br>
</div>
""", unsafe_allow_html=True)


# Slutning og tak
st.markdown("""
<div class='small-font reduce-margin'>
    This examination project was crafted for the Business Intelligence 2024 course.<br>
    Kind regards,<br>
    Rasmus Arendt, Deniz Denson, Victor Christensen & Marcus Løbel
</div>
""", unsafe_allow_html=True)