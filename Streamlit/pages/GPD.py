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
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import folium
#!pip install streamlit-folium
from streamlit_folium import st_folium


# Streamlit run ./Streamlit/app.py

st.set_page_config(page_title="Vaccination rates", page_icon="📊")

st.title("GPD pr Country affects Covid-19")
st.markdown("Based on our first hypothesis, we want to investigate the relationship between a country's GPD and the number of Covid-19 cases.")
st.markdown("Showing the relationship between GPD and Covid-19 cases.")

# Load the data. The data is from the Our World in Data's github (https://github.com/owid/covid-19-data/tree/master/public/data). downloaded on 10/03/2024
df = pd.read_csv("../Data/owid-covid-data.csv")

columns_to_keep_hypo1 = ['iso_code', 'location', 'total_cases', 'gdp_per_capita','date']

# new df
data_hypo1 = df[columns_to_keep_hypo1]


#Get percentage of missing values
missing_values = (data_hypo1.isnull().sum()/data_hypo1.shape[0])*100

# Remove rows
data_hypo1 = data_hypo1.dropna(subset=['total_cases'])

missing_values_hypo1 = (data_hypo1.isnull().sum()/data_hypo1.shape[0])*100

# Date column -> datetime
data_hypo1['date'] = pd.to_datetime(data_hypo1['date'])

# list of the iso codes
iso_codes_hypo1 = data_hypo1['iso_code'].unique()

# owid special codes
iso_codes_owid = data_hypo1[data_hypo1['iso_code'].str.contains('OWID')]['iso_code'].unique()

#Removing rows
data_hypo1 = data_hypo1[~data_hypo1['iso_code'].str.contains('OWID')]

# Check to see if it's done correct
iso_codes_owid = data_hypo1[data_hypo1['iso_code'].str.contains('OWID')]['iso_code'].unique()

#load dataset
pop_density = pd.read_csv("../Data/population-density.csv")


#dropping rows with missing values in gdp_per_capital column
data_hypo1 = data_hypo1.dropna(subset=['gdp_per_capita'])


def vacc_merge_datasets(dataset1, dataset2):
    # Convert dates to datetime objects
    dataset1['date'] = pd.to_datetime(dataset1['date'])
    dataset2['date'] = pd.to_datetime(dataset2['date'])

    # Merge datasets based on 'iso_code' and 'date'
    merged = pd.merge(dataset1, dataset2, on=['iso_code', 'date'], how='left', suffixes=('_1', '_2'))

    # Replace missing values in 'total_vaccinations_per_hundred_1' with values from 'total_vaccinations_per_hundred_2'
    merged['total_vaccinations_per_hundred'] = merged['total_vaccinations_per_hundred_1'].fillna(merged['total_vaccinations_per_hundred_2'])

    # Drop unnecessary columns
    merged.drop(['total_vaccinations_per_hundred_1', 'total_vaccinations_per_hundred_2'], axis=1, inplace=True)

    # Fill missing values in 'total_vaccinations_per_hundred' with most recent values from dataset2
    merged['total_vaccinations_per_hundred'].fillna(method='ffill', inplace=True)

    return merged

# Copy data
data_hypothesis_1 = data_hypo1[['location', 'total_cases', 'gdp_per_capita', 'date']]

last_row = data_hypothesis_1.groupby('location').last().reset_index()

#last_row['date'].max() == last_row['date'].min()

st.title("Cumulative Cases per Country")
st.markdown("Text here.")


# graph of the cumulative cases per country
data_hypothesis_1_subset = last_row.sort_values('total_cases', ascending=False)


# Brug plt.subplots for at oprette en figur og akse
fig, ax = plt.subplots(figsize=(20, 10))  # Juster størrelsen efter behov

# Brug seaborn på den specifikke akse for at lave et barplot
sns.barplot(x='location', y='total_cases', data=data_hypothesis_1_subset, ax=ax)

# Tilpasser plot for bedre visning
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#ax.set_title('Cumulative Cases per Country')
ax.set_xlabel('Country')
ax.set_ylabel('Number of Cumulative Cases')


# Viser plot i Streamlit
st.pyplot(fig)

st.title("Total cases vs. GDP per capita")
st.markdown("Based on the data, we can see that there is no correlation between a country's GDP and the number of Covid-19 cases.")

# Brug plt.subplots for at oprette en figur og akse
fig, ax = plt.subplots(figsize=(10, 6))  # Juster størrelsen efter behov

# Brug seaborn på den specifikke akse for at lave et scatterplot
sns.scatterplot(x='gdp_per_capita', y='total_cases', data=last_row, ax=ax)

# Tilføj titel og aksebetegnelser
#ax.set_title('Total Cases vs. GDP per Capita')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Total Cases')


# Viser plot i Streamlit
st.pyplot(fig)

st.title("Top 5 highest average total cases per country for each year")
st.markdown("Text here.")


data_hypo1['date'] = pd.to_datetime(data_hypo1['date'])

years = data_hypo1['date'].dt.year.unique()

# Bestem antallet af rækker og kolonner baseret på antallet af år. For simplicity, lader vi det være 3 kolonner.
rows = len(years) // 3 + (1 if len(years) % 3 else 0)

fig, axs = plt.subplots(rows, 3, figsize=(20, 20))  # Ajuster antallet af rækker og størrelsen efter behov

for i, y in enumerate(years):
    ax = axs[i // 3, i % 3] if rows > 1 else axs[i]  # Håndterer både enkelt og flere rækker af subplots
    
    # Beregn gennemsnittet af de totale tilfælde pr. land for året
    avg_total_cases = data_hypo1[data_hypo1['date'].dt.year == y].groupby('location')['total_cases'].mean()
    avg_total_cases = avg_total_cases.reset_index()
    avg_total_cases = avg_total_cases.sort_values(by='total_cases', ascending=False)
    
    # Vælg de top 5 lande
    top_5 = avg_total_cases.head(5)
    
    ax.pie(top_5['total_cases'], labels=top_5['location'], autopct='%.1f%%', startangle=90, shadow=True)
    ax.set_title(y, fontsize=15)

#fig.suptitle('Top 5 highest average total cases per country for each year', fontsize=20)

# Fjerner tomme subplot-pladser, hvis antallet af år ikke fylder alle subplot
for j in range(i + 1, rows * 3):
    fig.delaxes(axs.flatten()[j])

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Vis figuren i Streamlit
st.pyplot(fig)


class mul_lin_reg_model:
    r2_score_ = 0
    MAE = 0
    MSE = 0
    RMSE = 0

    eV = 0

    def __init__(self, X, y, name):
        self.country_name = name
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,  random_state=123, test_size=0.15)
        self.linreg = LinearRegression()
        self.linreg.fit(self.X_train, self.y_train)
    def predict(self, input):
        return self.linreg.predict(input)
    
    def do_scoring_(self, y_test, y_predicted):
        self.MAE = metrics.mean_absolute_error(y_test, y_predicted) 
        self.MSE = metrics.mean_squared_error(y_test, y_predicted)
        self.RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_predicted))
        self.eV = round(sm.explained_variance_score(y_test, y_predicted), 2)
        self.r2_score_ = r2_score(y_test, y_predicted)

def train_mul_lin_reg_by_name(country_name, dataset):
    country_subset = dataset[dataset['location'] == country_name]
    x_col = 'gdp_per_capita'
    y_col = 'total_cases'
    X = country_subset[x_col].values.reshape(-1, 1)
    y = country_subset[y_col].values.reshape(-1, 1)
    return mul_lin_reg_model(X,y, name)

import random
amount_of_countries = 5
countries_names = random.choices(data_hypo1['location'].unique(), k=amount_of_countries)
countries_and_models = {}
for name in countries_names:
    countries_and_models[name] = train_mul_lin_reg_by_name(name, data_hypo1)
    print(f"Added {name} to dict.")


countries_and_model_prediction = {}
for country_name in countries_and_models:
    model = countries_and_models[country_name]
    predictions = model.predict(model.y_test)
    countries_and_model_prediction[country_name] = predictions
    #print(predictions)
    model.do_scoring_(model.y_test, predictions)
    countries_and_models[country_name] = model
    print(f"{model.country_name}: mae: {model.MAE}, mse: {model.MSE}, rmse: {model.RMSE}, r2_score: {model.r2_score_}, eV: {model.eV}")


# Generer et Folium-kort
def generate_map(df):
    # Start kortet ved et globalt udsnit
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Tilføj markører for hvert land
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['country_name']}: {row['total_cases']} total cases",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Vis kortet i Streamlit
    st_folium(m, width=725, height=500)

# Du skal muligvis tilpasse din dataforberedelse for at matche dette eksempel
generate_map(df)