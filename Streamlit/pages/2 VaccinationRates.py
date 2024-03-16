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



st.title("Vaccination rates")
st.markdown("Hypothesis 2:")
st.markdown("'We do not believe there is a connection between a country's population density and the number of COVID-19 cases, where higher population density correlates with more COVID-19 cases. That is to say, countries with more cases also had higher vaccination coverage.'")

st.markdown("Based on our second hypothesis, we want to investigate the relationship between the number of Covid-19 cases, vaccination rates and population density.")


# Load the data. The data is from the Our World in Data's github (https://github.com/owid/covid-19-data/tree/master/public/data). downloaded on 10/03/2024
df = pd.read_csv("../Data/owid-covid-data.csv")

columns_to_keep_hypo1 = ['iso_code', 'location', 'total_cases', 'gdp_per_capita','date']
columns_to_keep_hypo2 = ['iso_code', 'location', 'total_cases', 'date', 'total_vaccinations_per_hundred', 'population_density']
columns_to_keep_hypo3 = ['iso_code', 'location', 'total_cases', 'human_development_index','date']

# new df
data_hypo2 = df[columns_to_keep_hypo2]

#remove rows
data_hypo2 = data_hypo2.dropna(subset=['total_cases'])

# Date column -> datetime
data_hypo2['date'] = pd.to_datetime(data_hypo2['date'])

#removing rows
data_hypo2 = data_hypo2[~data_hypo2['iso_code'].str.contains('OWID')]

missing_population_density = data_hypo2[data_hypo2['population_density'].isnull()]['iso_code'].unique()
#missing_population_density

#load dataset
pop_density = pd.read_csv("../Data/population-density.csv")



first_year = data_hypo2['date'].min().year
#first_year

last_year = data_hypo2['date'].max().year
#last_year

#discard unusuable years
pop_density = pop_density[pop_density['Year'] >= first_year]
pop_density = pop_density[pop_density['Year'] <= last_year]

pop_density.rename(columns={'Entity':'location', 'Code':'iso_code','Year':'year', 'Population density': 'population_density'}, inplace=True)

# Discover if it exists
doesnt_exists = []
for code in missing_population_density:
    if not code in pop_density['iso_code'].unique():
        doesnt_exists.append(code)

#print(len(doesnt_exists), len(missing_population_density))
#print(doesnt_exists)
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
   # df_with_pop_filled.loc[index] = [iso_code, location, row['total_cases'], pop_density_row['population_density'].values[0], row['date'], row['total_vaccinations_per_hundred']]

# put the data from intermediary df to original df
data_hypo2['population_density'] = data_hypo2['population_density'].fillna(df_with_pop_filled['population_density'])

#dropping rows with missing values in population_density
data_hypo2 = data_hypo2.dropna(subset=['population_density'])

#(data_hypo2.isnull().sum()/data_hypo2.shape[0])*100

#load another dataset to fill data
#vacc_per_hundred_dataset = pd.read_csv("../Data/covid-vaccination-doses-per-capita.csv")

#vacc_per_hundred_dataset.rename(columns={'Entity':'location', 'Code':'iso_code','Day':'date'}, inplace=True)

#rows_with_missing_vacc_per_hundred = data_hypo2[data_hypo2['total_vaccinations_per_hundred'].isnull()]
#print(f"covid data is missing {len(rows_with_missing_vacc_per_hundred)} rows")

# get the percentage of missing values
missing_values = (data_hypo2.isnull().sum()/data_hypo2.shape[0])*100
#missing_values

data_hypo2['date'].sort_values()

#vacc_per_hundred_dataset['date'].sort_values()

#vacc_per_hundred_dataset['date'] = pd.to_datetime(vacc_per_hundred_dataset['date'])

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

# test_dataset = fill_missing_vaccination_data(data_hypo2, vacc_per_hundred_dataset)
#intermediary_dataset = vacc_merge_datasets(data_hypo2, vacc_per_hundred_dataset)

#data_hypo2['total_vaccinations_per_hundred'] = data_hypo2['total_vaccinations_per_hundred'].fillna(intermediary_dataset['total_vaccinations_per_hundred'])


data_hypo2 = data_hypo2.dropna(subset=['total_vaccinations_per_hundred'])

#data_hypo2.isnull().sum()/data_hypo2.shape[0]*100

st.title("Histogram of Total Vaccinations per Hundred")
st.markdown("In this section, we will explore the relationship between the number of Covid-19 cases and the vaccination rates.")
st.markdown("As a starting point, we will visualize the distribution of the total vaccinations per hundred people.")
st.markdown("The results will be presented in a histogram, showing that the majority of the countries have a vaccination rate of more ")

# Plot histogram for 'total_vaccinations_per_hundred'
fig, ax = plt.subplots()
ax.hist(data_hypo2['total_vaccinations_per_hundred'], bins=30, color='blue')
ax.set_title('Histogram of Total Vaccinations per Hundred')
ax.set_xlabel('Total Vaccinations per Hundred')
ax.set_ylabel('Frequency')
#st.pyplot(fig)

#import silhoutte picture at show it
st.image('../Data/Histogram.png', use_column_width=True)

# List of Nordic countries
nordic_countries = ['Denmark', 'Norway', 'Sweden', 'Finland', 'Greenland']

# Copy the necessary columns to the hypothesis 2 dataframe
data_hypothesis_2 = data_hypo2[['location', 'total_cases', 'total_vaccinations_per_hundred', 'population_density', 'date']]

denmark_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Denmark']
norge_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Norway']
sweden_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Sweden']
finland_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Finland']
greenland_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Greenland']
# Create empty dictionaries to store data for each Nordic country
nordic_data = {}

# Iterate over each Nordic country and extract the data
for country in nordic_countries:
    nordic_data[country] = data_hypothesis_2[data_hypothesis_2['location'] == country]

# Check the data for each Nordic country
for country, country_data in nordic_data.items():
    #print(f"{country} data:")
    print(country_data.head())


# Copy the necessary columns to the hypothesis 2 dataframe
data_hypothesis_2 = data_hypo2[['location', 'total_cases', 'total_vaccinations_per_hundred', 'population_density', 'date']]
denmark_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Denmark']
norge_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Norway']
sweden_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Sweden']
finland_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Finland']
greenland_data = data_hypothesis_2[data_hypothesis_2['location'] == 'Greenland']
# Check the data to see if it looks good
print(data_hypothesis_2.head())

data_hypothesis_2.isnull().sum()

data_hypothesis_2.dropna(subset=['total_cases'], inplace=True)

data_hypothesis_2.describe()

last_row = data_hypothesis_2.groupby('location').last().reset_index()
last_row.sample(5)

# get the last row for each country/ lastest observation
last_row = data_hypothesis_2.groupby('location').last().reset_index()
last_row.sample(5)


nordic_countries = ['Denmark', 'Norway', 'Sweden', 'Finland', 'Greenland']

# Create an empty list to store the plots
figures = []


st.title("Simple 3D Scatter Plt")
st.markdown("Text")

#import silhoutte picture at show it
st.image('../Data/simple3d.png', use_column_width=True)




from matplotlib import cm


# Din eksisterende kode til at forberede data
last_row_subset = data_hypo2.groupby('location').last().reset_index()

# Brug en colormap (f.eks. 'viridis') og normaliser dine data baseret på 'total_cases' for bedre farvedifferentiering
norm = plt.Normalize(last_row_subset['total_cases'].min(), last_row_subset['total_cases'].max())
colors = cm.viridis(norm(last_row_subset['total_cases']))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
# Brug 'colors' i stedet for en statisk farve
sc = ax.scatter3D(last_row_subset['population_density'], last_row_subset['total_vaccinations_per_hundred'], last_row_subset['total_cases'], c=colors)
ax.set_title("Simple 3D Scatter Plot")
ax.set_xlabel('Population Density')
ax.set_ylabel('Total Vaccinations per Hundred')
ax.set_zlabel('Total Cases')

# Tilføj en farvebar for at forklare, hvad farverne repræsenterer
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
cbar.set_label('Total Cases')

#st.pyplot(fig)

st.title("3D Scatter Plot")
st.markdown("Text")

fig = px.scatter_3d(last_row_subset, x="population_density", y="total_vaccinations_per_hundred", z='total_cases', color="population_density",
                    size='total_vaccinations_per_hundred', size_max=40, opacity=0.8,
                    color_continuous_scale=px.colors.sequential.Viridis) # Ændrer til et mere distinkt farveskema

st.plotly_chart(fig)




st.title("Relationship between COVID-19 Cases and Vaccination Coverage in Nordic Countries")
st.markdown("A scatterplot to visualize how the nordic countries were affected by covid from present to four years back.")
st.markdown("We can see how it first hit around winter 2020/2021 and how it evolved progressively during the first quarter of 2021.")
st.markdown("During the winter of 2021/2022 the second wave hit, and the cases increased again. How vaccination covered this, is shown in the scatterplot below.")

# Filtrer dataene kun for nordic_countries
nordic_data = data_hypothesis_2[data_hypothesis_2['location'].isin(nordic_countries)]

# Opret et scatterplot med plotly
fig = px.scatter(nordic_data, x='date', y='total_cases', color='location', hover_name='location',
                 labels={'date': 'Date', 'total_cases': 'Total COVID-19 Cases'})
                 #title='Relationship between COVID-19 Cases and the timeperiod 2020-2024 in the Nordic Countries')
st.plotly_chart(fig)

# Filtrer dataene kun for nordic_countries
nordic_data = data_hypothesis_2[data_hypothesis_2['location'].isin(nordic_countries)]

st.title("Relationship between Vaccination Coverage and the timeperiod 2020-2024 in the Nordic Countries")
st.markdown("This scatterplot gives us an overview of vaccination coverage in the same period.")
st.markdown("We can see how the vaccination coverage increased over time, and how the different nordic countries compare to each other in terms of vaccination coverage.") 
st.markdown("The scatterplot is interactive, so you can hover over the points to see the exact values for each country and date.")

# Opret et scatterplot med plotly
fig = px.scatter(nordic_data, x='date', y='total_vaccinations_per_hundred', color='location', hover_name='location',
                 labels={'date': 'Date', 'total_vaccinations_per_hundred': 'Vaccinations per Hundred'})
                 #title='Relationship between Vaccination Coverage and the timeperiod 2020-2024 in the Nordic Countries')

# Vis plottet
st.plotly_chart(fig)

st.title("Top 5 Nordic countries with the highest average number of cases for vaccination coverage per hundred")
st.markdown("In the bar chart below, you can see the top 5 nordic countries with the highest average number of cases for vaccination coverage per hundred.")
st.markdown("The countries are represented by colorful bars, where the color represents the total number of cases.")

# Filtrere data_hypothesis_2 for kun nordiske lande
data_hypothesis_2_subset = data_hypothesis_2[data_hypothesis_2['location'].isin(nordic_countries)]

# De nordiske lande med det højeste gennemsnitlige antal sager for vaccinationsdækning per hundred
top_5_nordic_countries = data_hypothesis_2_subset.groupby('location')['total_vaccinations_per_hundred'].mean().nlargest(5).index

# Subset af data, der kun indeholder de nordiske lande med det højeste gennemsnitlige antal sager for vaccinationsdækning per hundred
top_5_nordic_data = data_hypothesis_2_subset[data_hypothesis_2_subset['location'].isin(top_5_nordic_countries)].sort_values('total_vaccinations_per_hundred', ascending=True)

# Søjlediagram med plotly for de nordiske lande med farvefulde søjler baseret på total_cases
fig = px.bar(top_5_nordic_data, x='location', y='total_vaccinations_per_hundred', color='total_cases',
             labels={'total_vaccinations_per_hundred': 'Gennemsnitlig vaccinationsdækning per hundrede', 'location': 'Land'},
             title='Top 5 nordiske lande med det højeste gennemsnitlige antal sager for vaccinationsdækning per hundrede')

# Vis plottet på Streamlit side
st.plotly_chart(fig)


# Opretter en figur og en akse med matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Itererer gennem listen af nordiske lande for at tilføje hvert lands data til histogrammet
for country in nordic_countries:
    # Filtrer data for det aktuelle land
    country_data = data_hypothesis_2_subset[data_hypothesis_2_subset['location'] == country]
    
    # Tilføjer landets data til histogrammet
    ax.hist(country_data['total_cases'], bins=20, alpha=0.5, label=country)

st.title("Distribution of the number of cases in the Nordic countries")
st.markdown("In this chart we can see the frequency of cases of all the countries, as shown by color.")

ax.set_title('Distribution of the number of cases in the Nordic countries')
ax.set_xlabel('Cases')
ax.set_ylabel('Observations')
ax.legend()

# Tilføjer en legende for at identificere hvert land
ax.legend()

# Viser plottet i Streamlit ved hjælp af st.pyplot()
st.pyplot(fig)


# -liste over funktionernes navne
feature_cols = ['total_vaccinations_per_hundred', 'total_cases']

# Brug listen til at vælge en undermængde af det oprindelige datasæt kun for nordiske lande
X = data_hypo2[data_hypo2['location'].isin(nordic_countries)][feature_cols]

# Udskriv y for at forstå dens form
# print(y)

# Udskriv X
print(X)


# Filtrér datasættet for kun numeriske kolonner
numerical_data_hypo2 = data_hypo2.select_dtypes(include='number')

# Beregn korrelationskoefficienter
correlation_matrix = numerical_data_hypo2.corr()

st.title("Correlation Matrix for the Nordic countries")
st.markdown("In our correlation matrix, we have 3 variables: total_cases, total_vaccinations_per_hundred and population_density.")
st.markdown("The different variables are divied into colortones that indicate positive and negative correlation. The darker the color, the stronger the correlation.")
st.markdown("What we can see in our matrix is that the colors show that there isn't strong correlation between the three values, and thus our hypothesis is not supported by our data.")

# Lav en heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
ax.set_title('Correlation Matrix')

# Vis plottet på Streamlit side
st.pyplot(fig)


# Opret en Series for y (total_cases) for Danmark
y = data_hypo2[data_hypo2['location'] == 'Denmark']['total_cases']

# Opret et nyt DataFrame X med de valgte funktioner (total_vaccinations_per_hundred og population_density) for Danmark
X = data_hypo2[data_hypo2['location'] == 'Denmark'][['total_vaccinations_per_hundred']]

# Opret et nyt X-dataframe kun med nordiske lande
X =  X[X.index.isin(y.index)]

#print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#default split 75:25
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Opret en model
linreg = LinearRegression()

# Tilpas modellen til vores træningsdata
linreg.fit(X_train, y_train)

# Denne kode kombinerer navnene på funktionerne med deres tilsvarende koefficienter ved hjælp af zip(
list(zip(feature_cols, linreg.coef_))

y_predicted = linreg.predict(X_test)
#y_predicted

# Den beregnede værdi, som du får, er det gennemsnitlige absolutte fejl (Mean Absolute Error - MAE) mellem de faktiske salgsværdier (y_test) og de forudsagte salgsværdier (y_predicted).
print(metrics.mean_absolute_error(y_test, y_predicted))

# beregner gennemsnittet af de kvadrerede forskelle mellem de faktiske og forudsagte værdier.
print(metrics.mean_squared_error(y_test, y_predicted))

# RMSE tager kvadratroden af MSE for at give os en værdi, der er på samme skala som den oprindelige responsvariabel.
print(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))

# R-squared
r2_score(y_test, y_predicted)

st.title("Multiple Linear Regression for Denmark")
st.markdown("Like the correlation matrix, our MLR model shows that there isn't correlation between the number of cases and the vaccination coverage per hundred.") 
st.markdown("Had there been correlation, the two lines (if you can call them that), would have been more similar.")
st.markdown("But what we can see, is that they divert from each other and are no way near. Hence, no correlation and hypothesis not supported by our data.")

# Visualise the regression results
fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted, color='blue')
ax.set_title('Multiple Linear Regression for Denmark')
ax.set_xlabel('Total Cases')
ax.set_ylabel('Total Vaccinations per Hundred')

# Vis plottet på Streamlit side
st.pyplot(fig)

st.title("Analysis of the hypothesis")
st.markdown("Following our hypothesis, our analysis revealed no significant correlation between a country's population density and the number of COVID-19 cases. While population density is a factor in virus transmission, other variables such as government interventions, healthcare infrastructure, and public compliance with safety measures also play significant roles in shaping the spread of the virus.")


