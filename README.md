# Exam project using Covid19 data for the course Business Intelligence

Members: Rasmus Arendt, Deniz Sonmez, Victor Christensen & Marcus Løbel

Datamatiker, 4 sem., Cph Business Lyngby


# Link to our miro-board and thought process:
[https://miro.com/app/board/uXjVNrBg1Yo=/](https://miro.com/app/board/uXjVNrBg1Yo=/?share_link_id=159207872113)https://miro.com/app/board/uXjVNrBg1Yo=/?share_link_id=159207872113

# Problem statement:
In the context of the COVID-19 pandemic, healthcare systems worldwide have been under unprecedented stress due to the dynamic and rapid spread of the virus. There is a critical need to optimize resource allocation, improve patient outcomes, and enhance the efficiency of public health responses. Our COVID-19 dataset contains detailed information on virus spread, healthcare system strain, and population demographics. We aim to leverage Business Intelligence (BI) tools to analyze this data to try and identify patterns, predict healthcare needs, and support decision-making processes and thus lessen the burden on society.

# Motivation
Early on we thought about covid19 for a project, as it's a topic that has affected us all for years. It was however not easy to come by data that we could use, which halted us for a bit. Then we went back and forth with a problem statement, before landing on above.
We find the topic very exciting as a future pandemic is a threat to everyone, and after covid-19, it's no longer a question whether pandemics will hit, but when.

# Theoretical foundation
Our research is based on real life, as we all experienced covid19 firsthand and therefore we think it's a very interesting matter to pursue. We do this by analyzing our dataset to provide meaningful insight to help combat a possible future pandemic.

# Argument of choices 
We have to admit that we were strapped for time at the end. We spent ALOT of time trying to clean data from different sources before we ended up on our chosen one.
Out of this and our problem statement, we came up with three hypothesis', which we then refined and found models fitting to help answer these. 

# Design
The design of our project is seen in Streamlit. Each hypothesis has it's own page with a "welcome" page welcoming the reader with a brief explanation including each hypothesis. Throughout each page, we have different figures offering an explanation which sums up our hypothesis and either rejects or confirms it.

# Implementation instructions
Make sure you have the right libraries imported installed (folium, seaborn, plotly etc). See below for a full list of libraries.
Make sure pathing to the .csv files are correct (they're inside the ./Data folder)
Make sure Streamlit is correctly installed (perhaps through Anaconda + keep 'environment' in mind)

# Artefacts
Code written in Python using Visual Studio Code
Different models and graphs launched through Streamlit
Two .csv files (part of the project)

Libraries:
- streamlit
- pandas
- numpy
- pydeck
- pickle
- math
- seaborn
- matplotlib.pyplot
- metrics
- KMeans
- cdist
- prep
- StandardScaler
- PCA
- stats
- LinearRegression
- r2_score
- ploty.express
- pickle
- pydeck
- DecisionTreeClassifier
- train_test_split
- tree
- classification_report
- confusion_matrix
- accuracy_score
- RandomForestClassifier
- GaussianNB

