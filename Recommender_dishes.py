import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
nltk.download('stopwords')

st.title("Restaurant Recommender System")

df = pd.read_csv(  # input dataset
    'https://raw.githubusercontent.com/smcri/ML_dataset/main/all_items2.csv')

df_rest = pd.read_csv(
    'all_rest.csv')

df.drop_duplicates(inplace=True)  # remove any duplicates
# preprocessing step for calculating similarity
df['Description'] = df['Description'].str.lower()
df['Description'] = df['Description'].apply(  # removing punctuation with empty string
    lambda text: text.translate(str.maketrans('', '', string.punctuation)))

STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
df['Description'] = df['Description'].apply(lambda text: " ".join(  # remving stopwords from description
    [word for word in str(text).split() if word not in STOPWORDS]))

# st.write(df)

# item_names = list(zip(df['Item Name'], df['Restaurant Index']))
# st.write(item_names)
# print(list(df_rest.iloc[:, 0]))
select_restaurant = st.selectbox(
    "Enter the name of restaurant from dropdown",
    list(df_rest.iloc[:, 0])
)

item_names = []


for i in range(df.shape[0]):
    item_names.append(df['Item Name'][i] + str(df['Restaurant Index']))
# print(df_rest.iloc[:, 0][0])
select_dish = st.selectbox(
    "Enter the name of resturant from dropdown",
    item_names
)
df_percent = df
df_percent.set_index('Item Name', inplace=True)  # setting Item name as index
# string all Item name into series indices
indices = pd.Series(df_percent.index)


tfidf = TfidfVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['Description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

name = 'Butter Naan'

rdishes = list()               # recommended dishses list

# retriving indiex of recommended item
idx = indices[indices == name].index[0]
# st.write(idx)
score_series = pd.Series(cosine_similarities[idx]).sort_values(
    ascending=False)  # retriving values with maximum cosine similarity

# string indices of top 10 dishes
# first position will be for dishes itself
top10 = list(score_series.iloc[1:31].index)

for each in top10:  # retriving the name of top10 dishes
    # if(rdishes.count(list(df_percent.index)[each]) <= 2):
    # if list(df_percent.index)[each] not in rdishes:
    rdishes.append(list(df_percent.index)[each])


# retring veg/nonveg of recommended list
rveg = df_percent['Veg/Non-veg'][df_percent.index == name]
rveg = rveg[0]
# retriving category of recommended list
rcat = df_percent['Category'][df_percent.index == name]
rcat = rcat[0]
rprice = df_percent['Price'][df_percent.index == name]
rprice = rprice[0]
score = list()
for dish in rdishes:
    # retriving veg/nonveg of dish
    veg = df_percent['Veg/Non-veg'][df_percent.index == dish]
    veg = veg[0]
    # retriving category of dish
    cat = df_percent['Category'][df_percent.index == dish]
    cat = cat[0]
    tempscore = 0
    if(veg == rveg):                                           # adding 3 if veg/nonveg matches
        tempscore = tempscore + 3
    if(rcat == cat):
        # adding 1 if category matches
        tempscore = tempscore + 1
    temprating = df_percent['Rating'][df_percent.index == dish]
    tempprice = df_percent['Price'][df_percent.index == dish]
    # sorting temprice acc to temprating in reverse manner
    tempsort = [x for _, x in sorted(zip(temprating, tempprice), reverse=True)]
    tempsort2 = sorted(df_percent['Rating']                        # retriving rating in reverse sorted manner
                       [df_percent.index == dish], reverse=True)
    tempscore = tempscore + 1.2*(tempsort2[0]/5)
    normprice = (tempsort[0]/830)
    tempscore = tempscore - 1.05*abs(normprice-rprice)/rprice
# print(df_percent['Price'][df_percent.index == dish])
    score.append(tempscore)

# sorting on the basis of score
rdishes = [x for _, x in sorted(zip(score, rdishes), reverse=True)]
newridshes = []
for dish in rdishes:
    if(newridshes.count(dish) <= 2):
        newridshes.append(dish)
rdishes = newridshes[0:10]
print(rdishes)
rindex = []

for dish in rdishes:
    rindex.append(int(df_percent['Restaurant Index']
                      [df_percent.index == dish][0]))

print(rindex)

rrest = []
dishes_details = []
i = 0
for index in rindex:
    templist = []
    templist.append(rdishes[i])
    templist.append(df_rest.iloc[index-1, [0]][0])
    templist.append(df_rest.iloc[index-1, [1]][0])
    templist.append(df_rest.iloc[index-1, [2]][0])
    templist.append(df_rest.iloc[index-1, [3]][0])
    i = i+1
    dishes_details.append(templist)
st.write(dishes_details)
print(rrest)
