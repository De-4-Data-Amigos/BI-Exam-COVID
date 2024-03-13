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
st.markdown("Text about the hypothesis")

# Load the data. The data is from the Our World in Data's github (https://github.com/owid/covid-19-data/tree/master/public/data). downloaded on 10/03/2024
df = pd.read_csv("../Data/owid-covid-data.csv")


columns_to_keep_hypo3 = ['iso_code', 'location', 'total_cases', 'human_development_index','date']

data_hypo3 = df[columns_to_keep_hypo3]

data_hypo3 = data_hypo3.dropna(subset=['total_cases'])

data_hypo3['date'] = pd.to_datetime(data_hypo3['date'])

data_hypo3 = data_hypo3[~data_hypo3['iso_code'].str.contains('OWID')]

pop_density = pd.read_csv("../Data/population-density.csv")

hdi_dataset = pd.read_csv("../Data/human-development-index.csv")

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
# Fill missing HDI values in dataset 2 with corresponding values from dataset 1
merged_dataset['human_development_index_x'] = merged_dataset['human_development_index_x'].fillna(merged_dataset['human_development_index_y'])

# Drop redundant columns
data_hypo3 = merged_dataset.drop(columns=['human_development_index_y']).rename(columns={'human_development_index_x':'human_development_index', 'location_x':'location'})


data_hypo3 = data_hypo3.dropna(subset=['human_development_index'])

#data_hypo3['human_development_index'].isnull().sum()/data_hypo3.shape[0]*100





#Copy necessary data
# Copy columns
print(data_hypo3.columns)
data_hypothesis_3 = data_hypo3[['human_development_index', 'total_cases', 'location', 'date']]


# Check the data to see if it looks good
#print(data_hypothesis_3.head())

# get the last row for each country
last_row = data_hypothesis_3.groupby('location').last().reset_index()
last_row.sample(5)

X = last_row['human_development_index'].values.reshape(-1, 1)
y = last_row['total_cases'].values.reshape(-1, 1)

# Determine k by minimizing the distortion - 
# the sum of the squared distances between each observation vector and its centroid
distortions = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k).fit(X)
    model.fit(X)
    distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 
print("Distortion: ", distortions)


st.title('Elbow Method for Optimal K')
st.markdown("Text")

# Opretter en ny figur og akse
fig, ax = plt.subplots()

# Tilføjer titel, plotter punkterne og angiver aksetiketter
#ax.set_title('Elbow Method for Optimal K')
ax.plot(K, distortions, 'bx-')
ax.set_xlabel('K')
ax.set_ylabel('Distortion')

# Bruger Streamlit til at vise figuren i appen
st.pyplot(fig)



# Optimal number of clusters K
num_clusters = 6

# next we create the KMeans model and fit it to the data 
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)

kmeans.fit(X)

scores = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    scores.append(score)



st.title('Silhouette Score Method for Discovering the Optimal K')
st.markdown("Text")
# Opretter en ny figur og akse
fig, ax = plt.subplots()

# Tilføjer titel, plotter punkterne og angiver aksetiketter
ax.plot(K, scores, 'bx-')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score')

# Bruger Streamlit til at vise figuren i appen
st.pyplot(fig)


predictions = kmeans.predict(X)
#print(predictions)


last_row['cluster_label'] = kmeans.labels_






st.title('Clustering of Countries by Human Development Index and Total Cases')
st.markdown("Text")

# Opretter en figur og et antal subplots baseret på antallet af klynger
fig, axs = plt.subplots(num_clusters, figsize=(10, num_clusters * 5), squeeze=False)

for i in range(num_clusters):
    # Vælger det aktuelle subplot
    ax = axs[i, 0]
    
    # Filtrerer data for den aktuelle klynge
    cluster = last_row[last_row['cluster_label'] == i]
    
    # Plotter punkter for den aktuelle klynge
    ax.scatter(cluster['human_development_index'], cluster['total_cases'])
    
    # Sætter titel og aksetiketter
    ax.set_title(f'Cluster {i}')
    ax.set_xlabel('Human Development Index')
    ax.set_ylabel('Total Cases')
    ax.grid(True)

# Justerer layout
plt.tight_layout()

# Viser den samlede figur i Streamlit
st.pyplot(fig)




# Opretter en ny figur
fig, ax = plt.subplots()

# Plotter datapunkter med forskellige farver for hver klynge
scatter = ax.scatter(last_row['human_development_index'], last_row['total_cases'], c=predictions, s=50, cmap='viridis')

# Tilføjer et grid for bedre læsbarhed
ax.grid(True)

# (Valgfrit) Tilføjer en farvebar for at repræsentere klyngerne, hvis det er relevant
cb = plt.colorbar(scatter)
cb.set_label('Cluster label')

# Viser plottet i Streamlit
st.pyplot(fig)

# Print cluster centers
#print(kmeans.cluster_centers_)



# first column
x_min = X.min()
x_max = X.max()

# second column
y_min = y.min()
y_max = y.max()


from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score



k = 5
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)  # Antager at X er defineret og er dit datasæt

# Opretter SilhouetteVisualizer med den fittede model
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

# Passer data til visualizeren
visualizer.fit(X)

# Streamlit's st.pyplot() funktion forventer en matplotlib figur.
# Yellowbrick's visualizer har en attribut 'fig', som er den figur, visualizeren tegner på.
# Vi kan vise denne figur direkte i Streamlit uden yderligere at skulle oprette figurer eller akser.
st.pyplot(visualizer.fig)




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


# Predict the labels of the test data
y_testp = classifier.predict(X_test)
#y_testp

# Calculated the accuracy of the model comparing the observed data and predicted data
print ("Accuracy is ", accuracy_score(y_test,y_testp))


# Create confusion matrix
confusion_mat = confusion_matrix(y_test,y_testp)
#confusion_mat

confusion = pd.crosstab(y_test,y_testp)
#confusion

# Classifier performance on training dataset
class_names = ['Class0', 'Class1', 'Class2','Class3', 'Class4', 'Class5']
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
plt.show()

model = GaussianNB()
model.fit(X_train, y_train)

# test the model on the test set
model.score(X_test, y_test)

