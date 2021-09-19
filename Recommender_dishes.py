from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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


nltk.download('stopwords')

st.title("Restaurant Recommender System")

df = pd.read_csv(  # input dataset
    'all_items.csv')

df_rest = pd.read_csv(
    'all_rest.csv')
df_rest.columns = ['Name', 'Rating', 'Cuisine', 'Address', 'No. of Ratings']
df.drop_duplicates(inplace=True)  # remove any duplicates

# for i in range(df.shape[0]):
#     if(df.iloc[i, [1]][0] == "Aloo Cheese Roll"):
#         st.write(i)
# preprocessing step for calculating similarity
df['Description'] = df['Description'].str.lower()
df['Description'] = df['Description'].apply(  # removing punctuation with empty string
    lambda text: text.translate(str.maketrans('', '', string.punctuation)))

STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
df['Description'] = df['Description'].apply(lambda text: " ".join(  # remving stopwords from description
    [word for word in str(text).split() if word not in STOPWORDS]))


def index_dish():
    select_restaurant = st.selectbox(
        "Choose Restaurant",
        list(df_rest.iloc[:, 0])
    )
    selected_rest_index = df_rest[df_rest['Name']
                                  == select_restaurant].index[0]+1
    # st.write(selected_rest_index)
    item_names = []

    for i in range(df.shape[0]):
        if(df['Restaurant Index'][i] == selected_rest_index):
            item_names.append(df['Item Name'][i])
    select_dish = st.selectbox(
        "Choose the corresponding dishes",
        item_names
    )
    return df[df['Item Name'] == select_dish].index[0]


tfidf = TfidfVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
# vectorise the description and calculate tfidf values
tfidf_matrix = tfidf.fit_transform(df['Description'])

# calculte correlation matrix of cosine similarity on the basis of tf idf
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

rdishes = list()               # recommended dishes list

idx = index_dish()  # getting index number of row


score_series = pd.Series(cosine_similarities[idx]).sort_values(
    ascending=False)  # retriving values with maximum cosine similarity on the basis of index

# indices of top 30  dishes
# first position will be for dishes itself
top10 = list(score_series.iloc[1:31].index)

# print(top10)
ntop10 = []

for each in top10:
    if(each != idx):
        # appending tuple of (item name,restaurant index) to rdishes
        if (df.iloc[each, [1]][0], df.iloc[each, [6]][0]) not in rdishes:
            rdishes.append((df.iloc[each, [1]][0], df.iloc[each, [6]][0]))
            ntop10.append(each)

# st.write(ntop10)
# retrieving veg/nonveg of recommended list
rveg = df.iloc[idx, [4]][0]

# retrieving category of recommended list
rcat = df.iloc[idx, [0]][0]
rprice = df.iloc[idx, [2]][0]
score = list()
for nindex in ntop10:
    # retriving veg/nonveg of dish
    veg = df.iloc[nindex, [4]][0]
    # retriving category of dish
    cat = df.iloc[nindex, [0]][0]
    tempscore = 0
    if(veg == rveg):                                           # adding 3 if veg/nonveg matches
        tempscore = tempscore + 3
    if(rcat == cat):
        # adding 1 if category matches
        tempscore = tempscore + 1
    temprating = df.iloc[nindex, [5]][0]
    tempprice = df.iloc[nindex, [2]][0]

    # assigning score on the basis of rating
    tempscore = tempscore + 1.2*(temprating/5)
    normprice = (tempprice/830)
    # penalise on the basis of price
    tempscore = tempscore - 1.05*abs(normprice-rprice)/rprice

    score.append(tempscore)

# sorting on the basis of score
rdishes = [x for _, x in sorted(zip(score, rdishes), reverse=True)]
# sorting dish indices on the basis of score
ntop10 = [x for _, x in sorted(zip(score, ntop10), reverse=True)]

dishname = []

newname = []
newridshes = []
newntop10 = []


# loop to retrieve dishname
for dish in rdishes:
    dishname.append(dish[0])

i = 0

# loop to append dishes if frequency is 3
for name in dishname:
    if(newname.count(name) <= 2):
        newname.append(name)
        newridshes.append(rdishes[i])
        newntop10.append(ntop10[i])
    i = i+1


rdishes = newridshes[0:10]  # taking top 10 dishes
ntop10 = newntop10[0:10]
# st.write(rdishes)
rindex = []  # list for restaurant index

for dish in rdishes:  # appending restaurant index of dish
    rindex.append(dish[1])

print(rindex)


def show_image(searchtext):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(
        'D:\\Downloads\\chromedriver_win32\\chromedriver.exe', chrome_options=options)
    driver.get("https://www.google.com/imghp?hl=en")

    query_in = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
    query_in.send_keys(searchtext)
    query_in.send_keys(Keys.ENTER)

    # image_tag = driver.find_element_by_xpath('//*[@id="hdtb-msb"]/div[1]/div/div[2]/a')
    # image_tag.click()

    req_image = driver.find_element_by_xpath(
        '//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img')

    st.image(req_image.get_attribute("src"))


dishes_details = []
i = 0
for index in rindex:
    templist = []
    templist.append(rdishes[i][0])  # dishname
    templist.append(df.iloc[ntop10[i], [0]][0])
    # Restaurant name
    templist.append(df_rest.iloc[index-1, [0]][0])
    templist.append(df_rest.iloc[index-1, [1]][0])  # rating
    templist.append(df_rest.iloc[index-1, [2]][0])  # cuisine
    templist.append(df_rest.iloc[index-1, [3]][0])  # category
    # query_string = rdishes[i][0] + df_rest.iloc[index-1, [0]][0]
    # downloader.download(query_string, limit=1,  output_dir='dataset',
    # adult_filter_off=True, force_replace=False, timeout=60)

    i = i+1
    dishes_details.append(templist)
# st.write(dishes_details)
# for detail in dishes_details:
#     print(detail)


if(st.button('Show Recommendation')):
    rows = int((len(dishes_details)+1)/2)
    columns = st.columns(len(dishes_details))
    index = 0
    for row in range(0, rows):
        col1, col2 = st.columns(2)
        col1.header(dishes_details[index][0])
        with col1:
            st.text("Category: " + dishes_details[index][1])
            st.text("Restaurant Name: " + dishes_details[index][2])
            st.text("Rating " + dishes_details[index][3])
            st.text("Restaurant Type: " + dishes_details[index][4])
            # show_image(
            #     str(dishes_details[index][0])+str(dishes_details[index][1]))
        # st.write("\n")
        index += 1
        if(index != len(dishes_details)):
            col2.header(dishes_details[index][0])
            with col2:
                st.text("Category: " + dishes_details[index][1])
                st.text("Restaurant Name: " + dishes_details[index][2])
                st.text("Rating " + dishes_details[index][3])
                st.text("Restaurant Type: " + dishes_details[index][4])
                # show_image(
                #     str(dishes_details[index][0])+str(dishes_details[index][1]))
        index += 1
        st.write("\n")
