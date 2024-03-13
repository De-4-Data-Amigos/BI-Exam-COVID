import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle

# Streamlit run ./Streamlit/app.py

st.set_page_config(page_title="GPD pr Country", page_icon="📊")

st.title("GPD pr Country affects Covid-19")
st.markdown("Based on our first hypothesis, we want to investigate the relationship between a country's GPD and the number of Covid-19 cases.")
st.markdown("Show the relationship between GPD and Covid-19 cases.")