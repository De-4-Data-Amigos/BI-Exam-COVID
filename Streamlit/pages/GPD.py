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
import json


# Streamlit run ./Streamlit/app.py

st.set_page_config(page_title="GPD", page_icon="📊")

st.title("How GPD pr Country affects Covid-19 cases?")
st.markdown("Hypothesis 1:")
st.markdown("'We do not believe that there is a correlation between the number of infected individuals in relation to a country's Gross National Product (GNP) per capita.'")
 
st.markdown("Based on our first hypothesis, we want to investigate if there's a relationship between a country's GPD and the number of Covid-19 cases, as this could prove valuable information for authorities.")

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

# Create a map of the world based on the latitude and longitude of each country and use iso_code to define the country that we're hovering over dataset in streamlit

st.title("Interactive map of total cases per country")
# Aggregate total cases by iso_code
country_cases = data_hypo1.groupby('iso_code')['total_cases'].sum().reset_index()

# Formater nummeret til tusinder og fjern de sidste tre cifre og det sidste komma
def format_number(number):
    number_in_thousands = number // 1000  # Dividerer med 1000 og afrunder til nærmeste hele tal
    return f"{number_in_thousands:,.0f}".replace(",", ".")

# Korrekt funktion til at specificere UTF-8-kodning og tilføje formaterede total_cases
def load_and_merge_geojson(geojson_path, covid_data):
    with open(geojson_path, 'r', encoding='utf-8') as f:  # Angiv kodning her
        geojson = json.load(f)
    
    for feature in geojson['features']:
        iso_code = feature['properties'].get('iso_a3')  # Antager at ISO-koderne er gemt under 'iso_a3'
        if iso_code:
            total_cases = covid_data.loc[covid_data['iso_code'] == iso_code, 'total_cases'].values
            if total_cases.size > 0:
                feature['properties']['total_cases'] = format_number(int(total_cases[0]))
            else:
                feature['properties']['total_cases'] = "Data ikke tilgængelig"
    return geojson

# Initialiserer et Folium-kort centreret på en global visning
m = folium.Map(location=[20, 0], zoom_start=2)

# Funktion til at tilføje et GeoJSON-lag til kortet
def add_geojson_layer(geojson_data, map_object, layer_name):
    folium.GeoJson(
        data=geojson_data,
        name=layer_name,
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'total_cases'],
            aliases=['Land: ', 'Total tilfælde: '],
            localize=True
        )
    ).add_to(map_object)

# Indlæser og tilføjer hvert kontinents GeoJSON til kortet
geojson_paths = {
    'Europe': '../Data/GeoMaps/EU Map.json', 
    'North America': '../Data/GeoMaps/NA Map.json',
    'South America': '../Data/GeoMaps/SA Map.json', 
    'Africa': '../Data/GeoMaps/Africa Map.json',
    'Asia': '../Data/GeoMaps/Asia Map.json'
}

for continent, path in geojson_paths.items():
    geojson_data = load_and_merge_geojson(path, country_cases)
    add_geojson_layer(geojson_data, m, continent)

# Tilføjer et lagkontrolpanel til kortet
folium.LayerControl().add_to(m)

# Viser kortet i Streamlit
st_folium(m, width=725, height=500)







st.title("Total cases per country")
st.markdown("On the chart below, we get an overview of accumulative cases throughout the world.")
st.markdown("The top three countries with most cases are:")
st.markdown("1) United States")
st.markdown("2) China")
st.markdown("3) India.")
st.markdown("These countries are also the most populated countries in the world, so it's not surprising that they have the most cases. However, it's interesting to see that the United States has the most cases, as it's a wealthy country with a high gdp per capita.")



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
st.markdown("Based on this scatterplot, we can see that there's no correlation between total cases and gdp per capita, as the data shows there's a somewhat even amount of cases throughout the different gdp per capita values.")
st.markdown("One could argue that the countries with the highest gdp per capita have the lowest amount of cases, but the data shows that there's no correlation between the two variables. This is interesting, as one could argue that the wealthier countries would have better healthcare and therefore fewer cases, but this is not the case, as the data shows that there's no correlation between the two variables.")

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
st.markdown("Below we've divided the highest average total cases per year into each years own piechart. This should help visualize how hard some countries were affected by covid throughout the years.")


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

st.title("Analysis of the hypothesis")
st.markdown("Through exploratory data analysis and regression modeling, we did not find hard evidence suggesting a significant correlation between a country's GNP per capita and the number of COVID-19 cases. Higher GNP per capita countries tended to have higher numbers of COVID-19 cases, indicating a potential relationship between economic prosperity and virus transmission, but the data did not really support this enough.")
