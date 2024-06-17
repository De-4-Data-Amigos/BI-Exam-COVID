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
st.markdown("'We do not believe there is a correlation between a country's population density and its number of COVID-19 cases. That is, higher population density does not lead to more COVID-19 cases. That is to say, countries with more cases also had higher vaccination coverage.'")

st.markdown("Changed the column 'total_cases' to 'total_cases_per_million' to change the diagrams, and give a more realistic overview")

# Load the data. The data is from the Our World in Data's github (https://github.com/owid/covid-19-data/tree/master/public/data). downloaded on 10/03/2024
df = pd.read_csv("../Data/owid-covid-data.csv")

columns_to_keep_hypo2 = ['iso_code', 'location', 'total_cases_per_million', 'date', 'total_vaccinations_per_hundred', 'population_density']

data_hypo2 = df[columns_to_keep_hypo2]

data_hypo2 = data_hypo2.dropna(subset=['total_cases_per_million'])

data_hypo2['date'] = pd.to_datetime(data_hypo2['date'])

data_hypo2 = data_hypo2[~data_hypo2['iso_code'].str.contains('OWID')]

missing_population_density = data_hypo2[data_hypo2['population_density'].isnull()]['iso_code'].unique()

#load dataset
pop_density = pd.read_csv("../Data/population-density.csv")

first_year = data_hypo2['date'].min().year

last_year = data_hypo2['date'].max().year

#discard unusuable years
pop_density = pop_density[pop_density['Year'] >= first_year]
pop_density = pop_density[pop_density['Year'] <= last_year]

pop_density = pop_density.dropna(subset=['Code'])

len(pop_density['Entity'].unique())


pop_density.rename(columns={'Entity':'location', 'Code':'iso_code','Year':'year', 'Population density': 'population_density'}, inplace=True)

# Discover if it exists
doesnt_exists = []
for code in missing_population_density:
    if not code in pop_density['iso_code'].unique():
        doesnt_exists.append(code)

print(len(doesnt_exists), len(missing_population_density))

# this means that there is 5 countries in the covid dataset, with missing population_density, that are not in the population density dataset

#remove excess countries that are not there
data_hypo2 = data_hypo2[~data_hypo2['iso_code'].isin(doesnt_exists)]

rows_with_missing_pop_density = data_hypo2[data_hypo2['population_density'].isnull()]
df_with_pop_filled = pd.DataFrame(columns=data_hypo2.columns)
for row in rows_with_missing_pop_density.iterrows():
    index = row[0]
    row = row[1]
    year = row['date'].year
    location = row['location']
    iso_code = row['iso_code']
    year_condition = pop_density['year'] == year
    iso_code_condition = pop_density['iso_code'] == iso_code
    combined_condition = year_condition & iso_code_condition
    pop_density_row = pop_density[combined_condition]
    #print(row)
    df_with_pop_filled.loc[index] = [iso_code, location, row['total_cases_per_million'], pop_density_row['population_density'].values[0], row['date'], row['total_vaccinations_per_hundred']]

# put the data from intermediary df to original df
data_hypo2['population_density'] = data_hypo2['population_density'].fillna(df_with_pop_filled['population_density'])

#dropping rows with missing values in population_density
data_hypo2 = data_hypo2.dropna(subset=['population_density'])

# get the percentage of missing values
missing_values = (data_hypo2.isnull().sum()/data_hypo2.shape[0])*100

data_hypo2['date'].sort_values()

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


data_hypo2 = data_hypo2.dropna(subset=['total_vaccinations_per_hundred'])

data_hypo2.to_csv("../Data/cleaned_hypo2.csv", index=False)

# Histogram
fig, ax = plt.subplots()
ax.hist(data_hypo2['total_vaccinations_per_hundred'], bins=30, color='blue')
ax.set_title('Histogram of Total Vaccinations per Hundred')
ax.set_xlabel('Total Vaccinations per Hundred')
ax.set_ylabel('Frequency')
#st.pyplot(fig)

#import silhoutte picture at show it
st.image('../Data/Histogram.png', use_column_width=True)

last_row_subset = data_hypo2.groupby('location').last().reset_index()

# Matplotlib 3D Scatter Plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter3D(last_row_subset['population_density'], last_row_subset['total_vaccinations_per_hundred'], last_row_subset['total_cases_per_million'], color="green")
ax.set_title("Simple 3D Scatter Plot")
ax.grid()
ax.set_xlabel('Population Density')
ax.set_ylabel('Total Vaccinations per Hundred')
ax.set_zlabel('Total Cases per Million')

# Brug Streamlit til at vise matplotlib-plot
st.pyplot(fig)

# Plotly 3D Scatter Plot
fig_px = px.scatter_3d(last_row_subset,
                       x="population_density",
                       y="total_vaccinations_per_hundred",
                       z='total_cases_per_million',
                       color="population_density",
                       size='total_vaccinations_per_hundred',
                       hover_name='location',  # Dette sikrer, at landenavne vises, når du holder musen over datapunkter
                       size_max=40,
                       opacity=0.8,
                       title="3D Scatter Plot")
# Brug Streamlit til at vise Plotly-plot
st.plotly_chart(fig_px)


import streamlit as st
import pandas as pd
import plotly.express as px

# Forudsat at 'data_hypo2' er forberedt som tidligere beskrevet

# Liste over nordiske lande
nordic_countries = ['Denmark', 'Norway', 'Sweden', 'Finland', 'Greenland']

# Kopierer de nødvendige kolonner til dataframet for hypotese 2
data_hypothesis_2 = data_hypo2[['location', 'total_cases_per_million', 'total_vaccinations_per_hundred', 'population_density', 'date']]

# Filtrer dataene kun for nordiske lande
nordic_data = data_hypothesis_2[data_hypothesis_2['location'].isin(nordic_countries)]

# Opretter et scatterplot med plotly
fig = px.scatter(nordic_data, x='date', y='total_cases_per_million', color='location', hover_name='location',
                 labels={'date': 'Date', 'total_cases_per_million': 'Total cases pr million COVID-19'},
                 title='Relationship between COVID-19 Cases and the timeperiod 2020-2024 in the Nordic Countries')

# Bruger Streamlit til at vise plotly-plot
st.plotly_chart(fig)




# Filtrer dataene kun for nordic_countries
nordic_data = data_hypothesis_2[data_hypothesis_2['location'].isin(nordic_countries)]

# Opret et scatterplot med plotly for total vaccinations per hundred
fig_vaccinations = px.scatter(nordic_data, x='date', y='total_vaccinations_per_hundred', color='location', hover_name='location',
                 labels={'date': 'Date', 'total_vaccinations_per_hundred': 'Vaccinations per Hundred'},
                 title='Relationship between Vaccination Coverage and the timeperiod 2020-2024 in the Nordic Countries')

# Vis vaccinationsplot i Streamlit
st.plotly_chart(fig_vaccinations)



# Filtrere data_hypothesis_2 for kun nordiske lande
data_hypothesis_2_subset = data_hypothesis_2[data_hypothesis_2['location'].isin(nordic_countries)]

# De nordiske lande med det højeste gennemsnitlige antal sager for vaccinationsdækning per hundred
top_5_nordic_countries = data_hypothesis_2_subset.groupby('location')['total_vaccinations_per_hundred'].mean().nlargest(5).index

# Subset af data, der kun indeholder de nordiske lande med det højeste gennemsnitlige antal sager for vaccinationsdækning per hundred
top_5_nordic_data = data_hypothesis_2_subset[data_hypothesis_2_subset['location'].isin(top_5_nordic_countries)].sort_values('total_vaccinations_per_hundred', ascending=True)

# Søjlediagram med plotly for de nordiske lande med farvefulde søjler baseret på total_cases
fig = px.bar(top_5_nordic_data, x='location', y='total_vaccinations_per_hundred', color='total_vaccinations_per_hundred',
             labels={'total_vaccinations_per_hundred': 'Gennemsnitlig vaccinationsdækning per hundrede', 'location': 'Land'},
             title='Top 5 nordiske lande med det højeste gennemsnitlige antal sager for vaccinationsdækning per hundrede')

# Vis plottet på Streamlit side
st.plotly_chart(fig)



plt.figure(figsize=(10, 6))
for country in nordic_countries:
    country_data = data_hypothesis_2_subset[data_hypothesis_2_subset['location'] == country]
    plt.hist(country_data['total_cases_per_million'], bins=20, alpha=0.5, label=country)

plt.title('Distribution of the Number of Cases in the Nordic Countries')
plt.xlabel('Number of Cases')
plt.ylabel('Number of Observations')
plt.legend()

# Use Streamlit to display the plot
st.pyplot(plt)

st.image('../Data/heatmap.png', use_column_width=True)

st.image('../Data/linear3.png', use_column_width=True)

st.image('../Data/polynomial.png', use_column_width=True)
