import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle

# Streamlit run ./Streamlit/app.py

st.set_page_config(page_title="Data about Covid", page_icon="📊")

st.title("Covid Data")
st.markdown("This application is a Streamlit dashboard that can be used to analyze Covid data.")
st.markdown("This is our exam project for the course of Business Intelligence 2024.")

gdp_per_capita = st.number_input("GNP pr capita", step=1.0, format="%.2f")
st.write("The number is:", gdp_per_capita)
