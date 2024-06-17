import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import preprocessing as prep
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
# Streamlit run ./Streamlit/Welcome.py
#Rasmus: cd C:\Users\rasmu\OneDrive\Skrivebord\4 sem\BI-Exam-COVID\

st.set_page_config(page_title="Vaccination rates", page_icon="📊")

st.title("Exam changes")
st.markdown("Hypothesis 2:")
st.markdown("Extra page for the exam")

#import polynomial regression model
#via picture
st.image('../Data/polynomial.png', use_column_width=True)
