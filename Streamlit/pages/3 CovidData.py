import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import tree


# Streamlit run ./Streamlit/app.py

st.set_page_config(page_title="Data about Covid", page_icon="📊")

st.title("Not all countries are equally exposed to the risk of COVID-19 infection.")
st.markdown("Hypothesis 3:")
st.markdown("'We do not believe that development of a  country (HDI) correlates to how exposed a county is to infection'")

st.markdown("In our third hypothesis, we want to try and predict clusters of countries based on Human Development Index and Total Cases. In other words, if two countries with close HDI values, will these also have the same amount of total cases?")

# Load the data. The data is from the Our World in Data's github (https://github.com/owid/covid-19-data/tree/master/public/data). downloaded on 10/03/2024
df = pd.read_csv("../Data/owid-covid-data.csv")

# Selecting the relevant columns for Hypothesis 3
columns_to_keep_hypo3 = ['iso_code', 'location', 'total_cases', 'human_development_index','date']

data_hypo3 = df[columns_to_keep_hypo3]

# Remove rows with missing values in 'total_cases' column
data_hypo3 = data_hypo3.dropna(subset=['total_cases'])

# Convert 'date' column to datetime format
data_hypo3['date'] = pd.to_datetime(data_hypo3['date'])

# Remove rows with 'iso_code' containing 'OWID'
data_hypo3 = data_hypo3[~data_hypo3['iso_code'].str.contains('OWID')]

# Read population density data
pop_density = pd.read_csv("../Data/population-density.csv")

# Read human development index data
hdi_dataset = pd.read_csv("../Data/human-development-index.csv")

# Filter human development index data for the first year
first_year = 2020
hdi_dataset = hdi_dataset[hdi_dataset['Year'] >= first_year]

# pop_density = pop_density[pop_density['Year'] >= first_year]
hdi_dataset.reset_index(drop=True, inplace=True)

# Create a list of years to add
additional_years = [2023, 2024]

# Repeat the last row for each additional year
for year in additional_years:
    last_row = hdi_dataset[hdi_dataset['Year'] == hdi_dataset['Year'].max()].copy()
    last_row['Year'] = year
    hdi_dataset = pd.concat([hdi_dataset, last_row], ignore_index=True)

hdi_dataset.rename(columns={'Code': 'iso_code', 'Entity':'location', 'Year':'year', 'Human Development Index':'human_development_index'}, inplace=True)

#data_hypo3['human_development_index'].isnull().sum()/data_hypo3.shape[0]*100

# Merge datasets based on the 'Code' and 'iso_code' columns
merged_dataset = pd.merge(data_hypo3, hdi_dataset, left_on='iso_code', right_on='iso_code', how='left')

# Fill missing HDI values with corresponding values from the second dataset
merged_dataset['human_development_index_x'] = merged_dataset['human_development_index_x'].fillna(merged_dataset['human_development_index_y'])

# Drop redundant columns and rename listed columns
data_hypo3 = merged_dataset.drop(columns=['human_development_index_y']).rename(columns={'human_development_index_x':'human_development_index', 'location_x':'location'})


data_hypo3 = data_hypo3.dropna(subset=['human_development_index'])

#data_hypo3['human_development_index'].isnull().sum()/data_hypo3.shape[0]*100


#Copy necessary data, the columns we want
print(data_hypo3.columns)
data_hypothesis_3 = data_hypo3[['human_development_index', 'total_cases', 'location', 'date']]


# Check the data to see if it looks good
#print(data_hypothesis_3.head())

# Group data by 'location' and select the last row for each country
last_row = data_hypothesis_3.groupby('location').last().reset_index()
last_row.sample(5)

# Define features and labels
X = last_row['human_development_index'].values.reshape(-1, 1)
y = last_row['total_cases'].values.reshape(-1, 1)

# Determine the optimal number of clusters using the Elbow Method
# Done by determining k (Kmeans) and minimizing the distortion - 
# the sum of the squared distances between each observation vector and its centroid


distortions = []  #Initialize an empty list to store distortion values
K = range(2,10)   #define our range of values for the number of clusters (K) from 2-9
for k in K:
    model = KMeans(n_clusters=k).fit(X) #Create a KMeans model with the current value of K and fit it to the data
    model.fit(X)  #Fit the model again (unnecessary, as the model has already been fitted above)

    #Calculate distortion for the current value of K and append it to the list
    #Distortion is defined as the average of the squared distances from the cluster centers to the data points
    distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 

    # Print the distortions for each value of K
print("Distortion: ", distortions)


st.title('Elbow Method for Optimal K')
st.markdown("In our elbow method, we're looking to find the most optimal amount of clusters of countries based on Human Development Index and Total Cases. We're looking for the 'elbow' in the graph, which is the point where the distortion begins to decrease at a slower rate.")
st.markdown("In our case, we'd recommend 5 clusters using this method, hence the 'elbow' is at 5. There's also an elbow at 3, but we prefer using 5 clusters, as it gives a better representation of the data.")

## ---------------- ##
# Things to notice:

# The Elbow Method is a heuristic process used to determine the optimal number of clusters in a dataset. It is based on the sum of squared distances between data points and their assigned clusters' centroids. The method is called the Elbow Method because the optimal number of clusters is at the "elbow" of the graph, where the distortion begins to decrease at a slower rate.

# Overfitting - Occurs if the model learns the training data too well. Will capture noise or random fluctuations in the training data and will not generalize well to new data.

# Underfitting - Occurs when the model is too simple to capture the underlying structure of the data. Will not capture the data well and will not generalize well to new data.


## ------------------ ## 
# Plot the Elbow Method graph
fig, ax = plt.subplots()
# Tilføjer titel, plotter punkterne og angiver aksetiketter
#ax.set_title('Elbow Method for Optimal K')
ax.plot(K, distortions, 'bx-')
ax.set_xlabel('K')
ax.set_ylabel('Distortion')

# Bruger Streamlit til at vise figuren i appen
st.pyplot(fig)



# Optimal number of clusters K
# Used to fit the Kmeans model with the optimal amount of clusters
num_clusters = 6

# next we create the KMeans model and fit it to the data 
# Determine the optimal number of clusters using the Silhouette Score Method
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)

kmeans.fit(X) # Fit the KMeans model to the data

# Calculate silhouette scores for different values of K
scores = [] # Initialize an empty list to store silhouette scores
K = range(2,10) # Range of values for number of clusters
for k in K:
    model = KMeans(n_clusters=k, n_init=10) # Create a KMeans model with the current value of K and perform multiple initializations
    model.fit(X)  # Fit the model to the data

     # Calculate the silhouette score for the current clustering
    # Silhouette score measures how similar an object is to its own cluster compared to other clusters
    # It ranges from -1 to 1, where a higher score indicates better separation between clusters
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    
    # Append the silhouette score to the list
    scores.append(score)


# Plot the Silhouette Score Method graph
st.title('Silhouette Score Method for Discovering the Optimal K')
st.markdown("Silhouette Score Method is a different method for this, but we're again looking for the optimal amount of clusters of countries based on Human Development. We've chosen 6, as 3 is too few, and 2 is not a good represenative of the data.")

# Create a new figure and axis
fig, ax = plt.subplots() # Create a new figure and axis

# Plot silhouette scores against number of clusters
# And add labels to the plot
ax.plot(K, scores, 'bx-')
ax.set_xlabel('K') #x-axis label
ax.set_ylabel('Silhouette Score') #y-axis label

# Showing plot through streamlit
st.pyplot(fig)

# Predict cluster labels for each data point using the fitted Kmeans model

predictions = kmeans.predict(X)
#print(predictions)

# Add cluster labels to the dataframe
last_row['cluster_label'] = kmeans.labels_


# Visualize clustering of countries by Human Development Index and Total Cases
st.title('Clustering of Countries by Human Development Index and Total Cases')
st.markdown("Now we're using the 6 clusters from Silhouette Score Method to cluster the countries based on Human Development.")

# Opretter en figur og et antal subplots baseret på antallet af klynger
fig, axs = plt.subplots(num_clusters, figsize=(10, num_clusters * 5), squeeze=False)

for i in range(num_clusters):
    ax = axs[i, 0] # Select the current subplot
    
    # Filter data for the current cluster
    cluster = last_row[last_row['cluster_label'] == i]
    
    # Plot data points for the current cluster
    ax.scatter(cluster['human_development_index'], cluster['total_cases'])
    
    # Title + labels
    ax.set_title(f'Cluster {i}')
    ax.set_xlabel('Human Development Index')
    ax.set_ylabel('Total Cases')
    ax.grid(True) # Add grid lines to the plot

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Show plot through Streamlit
st.pyplot(fig)


st.markdown("Here we can see all the clusters in different colors. We can see that the countries are clustered based on their Human Development Index and Total Cases. We can also see that the countries are not equally exposed to the risk of COVID-19 infection, as we can see that the clusters are not equally distributed. Lastly, we can see that the countries with the highest Human Development Index are in the same cluster, and the countries with the lowest Human Development.")

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot data points with different colors for each cluster
scatter = ax.scatter(last_row['human_development_index'], last_row['total_cases'], c=predictions, s=50, cmap='viridis')

# Add grid lines for better readability
ax.grid(True)

# Adds a color bar to represent clusters, if applicable
cb = plt.colorbar(scatter)
cb.set_label('Cluster label')

# Show the plot in Streamlit
st.pyplot(fig)

# Print cluster centers
#print(kmeans.cluster_centers_)

# Calculate the range for the first column
x_min = X.min()
x_max = X.max()

# Calculate the range for the second column
y_min = y.min()
y_max = y.max()


#TODO: VIRKER IKKE

from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score

# Instantiate a visualizer with the number of clusters
k = 6
model = KMeans(n_clusters=k, n_init=10)
model.fit_predict(X)

from sklearn.metrics import silhouette_score

# Calculate the silhouette score
score = silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# Visualize the silhouette scores of all points
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(X)
visualizer.show()  







# Convert the dataset into array
array = last_row[['human_development_index', 'total_cases', 'cluster_label']].values


# X - features, all rows, all columns but the last one
# y - labels, all rows, the last column
X, y = array[:, :-1], array[:, -1]

# Separate input data into classes based on labels
class0 = np.array(X[y==0])
class1 = np.array(X[y==1])
class2 = np.array(X[y==2])
class3 = np.array(X[y==3])
class4 = np.array(X[y==4])
class5 = np.array(X[y==5])

# Split the dataset into into training and testing sets in proportion 8:2 
#   80% of it as training data
#   20% as a validation dataset
set_prop = 0.2

#  Initialize seed parameter for the random number generator used for the split
seed = 7

# Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=set_prop, random_state=seed)

params = {'max_depth': 3}
classifier = DecisionTreeClassifier(**params)
# n_estimators - the number of trees in the forest
# classifier = RandomForestClassifier(n_estimators = 100, max_depth = 6)
classifier.fit(X_train, y_train)



# draw tree from the trained data by graphviz package
import graphviz
gr_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names=['human_development_index', 'total_cases'], class_names = True,        
                         filled=True, rounded=True, proportion = False, special_characters=True)  
dtree = graphviz.Source(gr_data) 
#dtree 

st.title("Silhouette Score")
st.markdown("Looking at this, we can see that the average silhouette score is about 0.58, which is an okay score. It means that the clusters are well apart from each other and are well clustered. We can also see that the clusters are equally exposed to the risk of COVID-19 infection, as we can see that the clusters are equally distributed.")

#import silhoutte picture at show it
st.image('../Data/silhouette.png', use_column_width=True)


st.title("Decision Tree")
st.markdown("Using the decision tree, we can try and predict what cluster a country should go into using the HDI and total cases. Using mathematics, it predicts the cluster, and as we can see, the countries with the highest Human Development Index are in the same cluster, and the countries with the lowest Human Development Index.")

# import tree picture at show it
st.image('../Data/tree.png', use_column_width=True)

st.title("Analysis of the hypothesis")
st.markdown("Through our analysis, we can gather that countries, no matter their HDI, are at a seemingly equal risk of covid-19 infection. Using our model, it's possible to predict what cluster a country should go into, based on the HDI and total cases. We can also see that the countries with the highest Human Development still have a high amount of cases, and the countries with the lowest Human Development still have a low amount of cases. This means that the HDI does not correlate to how exposed a country is to infection. Further investigation is required to evaluate the relationship between a country's Human Development Index (HDI) and its susceptibility to COVID-19 infection. What we've gathered so far, is that more regression analysis and correlation studies should be done to determine whether there truly is a significant association between HDI and COVID-19 transmission rates.")