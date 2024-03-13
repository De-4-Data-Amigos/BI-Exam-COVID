import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#To run this file, use the following command in the terminal:
#streamlit run ./Streamlit/Welcome.py
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
    In an era where data-driven decision making is important, our project presents a analysis aimed at pandemic preparedness and response. Utilizing extensive data from Our World in Data (OWID), we delve into lessons learned from the COVID-19 pandemic, identifying opportunities efficient crisis management strategies.<br><br>
    This Streamlit dashboard offers an interactive platform for exploring various data dimensions, revealing insights into the effectiveness of different containment measures and the impact on healthcare systems. Our goal is to provide a tool that can inform future policy decisions, ensuring resilient decision making towards potential health crisis.<br><br>
</div>
""", unsafe_allow_html=True)

# Formål med projektet
st.markdown("""
<div class='reduce-margin intro-text'>
    <strong>Purpose of the Project:</strong><br>
    In this Streamlit dashboard, our main goal is to explore and visualize the relationship between various factors and COVID-19 statistics to challenge common perceptions and inform future policy. Through in-depth analysis supported by a series of graphs, models and explanations, we aim to investigate three key hypotheses. <br>
            <br>
    <strong>Hypothesis 1</strong><br> We do not believe that there is a correlation between the number of infected individuals in relation to a country's Gross National Product (GNP) per capita.<br>
            <br>
    <strong>Hypothesis 2</strong><br> We do not believe there is a connection between a country's population density and the number of COVID-19 cases, where higher population density correlates with  more COVID-19 cases. That is to say, countries with more cases also had higher vaccination coverage.<br>
            <br>
    <strong>Hypothesis 3</strong><br>We do not believe that development of a country (HDI) correlates to how exposed a county is to infection.<br><br>
    By presenting data-driven insights on these hypotheses, we hope to provide valuable knowledge that could help in crafting more effective health policies and responses in the face of future pandemics. Our analysis is intended to empower policymakers, including you, by providing a nuanced understanding of the factors that influence pandemic dynamics and public health outcomes.<br><br>
</div>
""", unsafe_allow_html=True)


# Slutning og tak
st.markdown("""
<div class='small-font reduce-margin'>
    This examination project was crafted for the Business Intelligence 2024 course.<br>
    Kind regards,<br>
    Rasmus Tornby Arendt, Deniz Denson, Victor Christensen & Marcus Løbel
</div>
""", unsafe_allow_html=True)